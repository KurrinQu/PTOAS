#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONLOADSTOREELISION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct TrackedStore {
  vector::MaskedStoreOp op;
  Value base;
  SmallVector<Value, 2> indices;
  Value mask;
  Value value;
};

struct FusionRegionStoreContext {
  Block *body = nullptr;
  Block *parentBlock = nullptr;
  Operation *regionOp = nullptr;
  llvm::DenseSet<Value> yieldedValues;
};

static bool areEquivalentValues(Value lhs, Value rhs);

static bool areEquivalentValueRanges(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
           return areEquivalentValues(std::get<0>(pair), std::get<1>(pair));
         });
}

static bool areEquivalentOperations(Operation *lhs, Operation *rhs) {
  if (!lhs || !rhs)
    return false;
  if (lhs->getName() != rhs->getName())
    return false;
  if (lhs->getNumRegions() != 0 || rhs->getNumRegions() != 0)
    return false;
  if (lhs->getNumResults() != rhs->getNumResults())
    return false;
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;
  if (lhs->getAttrDictionary() != rhs->getAttrDictionary())
    return false;
  if (!llvm::equal(lhs->getResultTypes(), rhs->getResultTypes()))
    return false;

  if (auto lhsDim = dyn_cast<memref::DimOp>(lhs)) {
    auto rhsDim = cast<memref::DimOp>(rhs);
    return lhsDim.getSource().getType() == rhsDim.getSource().getType() &&
           areEquivalentValues(lhsDim.getIndex(), rhsDim.getIndex());
  }

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (!areEquivalentValues(lhsOperand, rhsOperand))
      return false;
  }
  return true;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  if (!lhs || !rhs)
    return false;
  if (lhs.getType() != rhs.getType())
    return false;

  auto lhsArg = dyn_cast<BlockArgument>(lhs);
  auto rhsArg = dyn_cast<BlockArgument>(rhs);
  if (lhsArg || rhsArg) {
    return lhsArg && rhsArg && lhsArg.getOwner() == rhsArg.getOwner() &&
           lhsArg.getArgNumber() == rhsArg.getArgNumber();
  }

  return areEquivalentOperations(lhs.getDefiningOp(), rhs.getDefiningOp());
}

static bool areEquivalentMaskValues(Value lhs, Value rhs) {
  if (areEquivalentValues(lhs, rhs))
    return true;
  auto lhsMask = lhs.getDefiningOp<vector::ConstantMaskOp>();
  auto rhsMask = rhs.getDefiningOp<vector::ConstantMaskOp>();
  if (!lhsMask || !rhsMask)
    return false;
  return lhs.getType() == rhs.getType() &&
         lhsMask.getMaskDimSizesAttr() == rhsMask.getMaskDimSizesAttr();
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isSupportedLeafOp(Operation *op) {
  if (isa<vector::MaskedLoadOp, vector::MaskedStoreOp>(op))
    return true;
  return isPureNoRegionOp(op);
}

static Value getCanonicalTrackedValue(Value value) {
  while (value) {
    Operation *def = value.getDefiningOp();
    if (!def)
      break;

    if (auto bridge = dyn_cast<pto::SimdTileToMemrefOp>(def)) {
      value = bridge.getSrc();
      continue;
    }
    if (auto cast = dyn_cast<memref::CastOp>(def)) {
      value = cast.getSource();
      continue;
    }
    if (auto reshape = dyn_cast<memref::ReshapeOp>(def)) {
      value = reshape.getSource();
      continue;
    }
    if (auto reinterpretCast = dyn_cast<memref::ReinterpretCastOp>(def)) {
      value = reinterpretCast.getSource();
      continue;
    }
    if (auto collapse = dyn_cast<memref::CollapseShapeOp>(def)) {
      value = collapse.getSrc();
      continue;
    }
    if (auto expand = dyn_cast<memref::ExpandShapeOp>(def)) {
      value = expand.getSrc();
      continue;
    }
    if (auto memorySpaceCast = dyn_cast<memref::MemorySpaceCastOp>(def)) {
      value = memorySpaceCast.getSource();
      continue;
    }
    if (auto transpose = dyn_cast<memref::TransposeOp>(def)) {
      value = transpose.getIn();
      continue;
    }
    break;
  }
  return value;
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur->getBlock() == block)
      return cur;
  return nullptr;
}

static std::optional<FusionRegionStoreContext>
buildFusionRegionStoreContext(pto::FusionRegionOp fusionRegion) {
  Block &body = fusionRegion.getBody().front();
  auto yieldOp = dyn_cast<pto::YieldOp>(body.getTerminator());
  if (!yieldOp)
    return std::nullopt;

  FusionRegionStoreContext context;
  context.body = &body;
  context.parentBlock = fusionRegion->getBlock();
  context.regionOp = fusionRegion.getOperation();

  for (Value yielded : yieldOp.getValues()) {
    Value canonical = getCanonicalTrackedValue(yielded);
    if (canonical)
      context.yieldedValues.insert(canonical);
  }

  return context;
}

static Block *getLeafLoopBody(pto::SimdVecScopeOp scope) {
  Block &scopeBody = scope.getBody().front();
  bool seenOuterLoop = false;
  scf::ForOp outerLoop;
  for (Operation &op : scopeBody) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (seenOuterLoop)
        return nullptr;
      seenOuterLoop = true;
      outerLoop = loop;
      continue;
    }
    if (seenOuterLoop || !isPureNoRegionOp(&op))
      return nullptr;
  }

  if (!outerLoop || !outerLoop.getInitArgs().empty())
    return nullptr;

  bool seenInnerLoop = false;
  scf::ForOp innerLoop;
  for (Operation &op : outerLoop.getBody()->without_terminator()) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (seenInnerLoop)
        return nullptr;
      seenInnerLoop = true;
      innerLoop = loop;
      continue;
    }
    if (seenInnerLoop || !isPureNoRegionOp(&op))
      return nullptr;
  }

  Block *leafBody = innerLoop ? innerLoop.getBody() : outerLoop.getBody();
  if ((innerLoop && !innerLoop.getInitArgs().empty()) || !leafBody)
    return nullptr;

  for (Operation &op : leafBody->without_terminator())
    if (!isSupportedLeafOp(&op))
      return nullptr;

  return leafBody;
}

static int findTrackedStoreIndex(ArrayRef<TrackedStore> stores, Value base,
                                 ArrayRef<Value> indices, Value mask) {
  for (int index = static_cast<int>(stores.size()) - 1; index >= 0; --index) {
    const TrackedStore &store = stores[index];
    if (areEquivalentValues(store.base, base) &&
        areEquivalentValueRanges(store.indices, indices) &&
        areEquivalentMaskValues(store.mask, mask)) {
      return index;
    }
  }
  return -1;
}

static bool shouldElideTailStore(const TrackedStore &store,
                                 const FusionRegionStoreContext &context,
                                 Operation *scopeOp) {
  Value canonicalBase = getCanonicalTrackedValue(store.base);
  if (!canonicalBase)
    return false;
  // Yielded frontier is still region-observable in v1, so its final
  // materializing store must be preserved even if there is no reload.
  if (context.yieldedValues.contains(canonicalBase))
    return false;

  for (OpOperand &use : canonicalBase.getUses()) {
    Operation *owner = use.getOwner();
    if (context.regionOp->isProperAncestor(owner)) {
      // Uses nested under the current vec_scope are fine: erasing the tail store
      // only affects memory materialization, while SSA users still observe the
      // forwarded vector value. A later top-level op in the same fusion region
      // may still require the buffer to stay materialized, so keep the store.
      Operation *topLevelUser =
          getTopLevelAncestorInBlock(owner, context.body);
      if (!topLevelUser)
        return false;
      if (topLevelUser == scopeOp)
        continue;
      if (scopeOp->isBeforeInBlock(topLevelUser))
        return false;
      continue;
    }

    // Any observable use after the fusion_region means the buffer escapes the
    // region boundary, so the final store must remain.
    Operation *topLevelUser =
        getTopLevelAncestorInBlock(owner, context.parentBlock);
    if (!topLevelUser)
      return false;
    if (topLevelUser == context.regionOp)
      continue;
    if (context.regionOp->isBeforeInBlock(topLevelUser))
      return false;
  }
  return true;
}

static bool
elideLoadStoreRoundTripsInLeafBody(Block &body,
                                   const FusionRegionStoreContext *context,
                                   Operation *scopeOp) {
  SmallVector<Operation *, 8> eraseOrder;
  llvm::SmallPtrSet<Operation *, 8> scheduledForErase;
  SmallVector<TrackedStore, 8> trackedStores;
  bool changed = false;

  auto scheduleErase = [&](Operation *op) {
    if (scheduledForErase.insert(op).second)
      eraseOrder.push_back(op);
  };

  for (Operation &op : body.without_terminator()) {
    if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
      SmallVector<Value, 4> loadIndices(load.getIndices().begin(),
                                        load.getIndices().end());
      int matchIndex = findTrackedStoreIndex(trackedStores, load.getBase(),
                                             loadIndices, load.getMask());
      if (matchIndex >= 0) {
        load.getResult().replaceAllUsesWith(trackedStores[matchIndex].value);
        scheduleErase(load);
        changed = true;
      }
      continue;
    }

    if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
      SmallVector<Value, 4> storeIndices(store.getIndices().begin(),
                                         store.getIndices().end());
      int matchIndex = findTrackedStoreIndex(trackedStores, store.getBase(),
                                             storeIndices, store.getMask());
      if (matchIndex >= 0) {
        scheduleErase(trackedStores[matchIndex].op);
        trackedStores.erase(trackedStores.begin() + matchIndex);
        changed = true;
      }

      trackedStores.push_back(TrackedStore{
          store,
          store.getBase(),
          SmallVector<Value, 2>(store.getIndices().begin(),
                                store.getIndices().end()),
          store.getMask(),
          store.getValueToStore(),
      });
      continue;
    }

    if (!isPureNoRegionOp(&op))
      trackedStores.clear();
  }

  if (context) {
    for (const TrackedStore &store : trackedStores) {
      if (!shouldElideTailStore(store, *context, scopeOp))
        continue;
      scheduleErase(store.op);
      changed = true;
    }
  }

  for (Operation *op : eraseOrder)
    op->erase();
  return changed;
}

struct PTOFusionLoadStoreElisionPass
    : public pto::impl::PTOFusionLoadStoreElisionBase<
          PTOFusionLoadStoreElisionPass> {
  using pto::impl::PTOFusionLoadStoreElisionBase<
      PTOFusionLoadStoreElisionPass>::PTOFusionLoadStoreElisionBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    llvm::DenseMap<Operation *, FusionRegionStoreContext> regionContexts;
    func.walk([&](pto::FusionRegionOp fusionRegion) {
      std::optional<FusionRegionStoreContext> context =
          buildFusionRegionStoreContext(fusionRegion);
      if (!context)
        return;
      regionContexts.try_emplace(fusionRegion.getOperation(), std::move(*context));
    });

    func.walk([&](pto::SimdVecScopeOp scope) {
      Block *leafBody = getLeafLoopBody(scope);
      if (!leafBody)
        return;

      const FusionRegionStoreContext *context = nullptr;
      Operation *scopeOp = nullptr;
      if (auto fusionRegion = scope->getParentOfType<pto::FusionRegionOp>()) {
        auto it = regionContexts.find(fusionRegion.getOperation());
        if (it != regionContexts.end() &&
            scope->getBlock() == &fusionRegion.getBody().front()) {
          context = &it->second;
          scopeOp = scope.getOperation();
        }
      }

      (void)elideLoadStoreRoundTripsInLeafBody(*leafBody, context, scopeOp);
    });
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionLoadStoreElisionPass() {
  return std::make_unique<PTOFusionLoadStoreElisionPass>();
}
