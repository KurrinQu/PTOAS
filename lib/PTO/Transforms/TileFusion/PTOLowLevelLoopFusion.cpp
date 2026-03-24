#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOLOWLEVELLOOPFUSION
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct StageInfo {
  pto::SimdVecScopeOp scope;
  SmallVector<Operation *, 4> setupOps;
  SmallVector<Operation *, 4> scopePreludeOps;
  scf::ForOp outerLoop;
  SmallVector<Operation *, 4> outerLoopPreludeOps;
  scf::ForOp innerLoop;
  SmallVector<Operation *, 8> leafOps;
  vector::MaskedStoreOp store;

  bool hasInnerLoop() const { return static_cast<bool>(innerLoop); }
};

struct ForwardedStore {
  Value base;
  SmallVector<Value, 2> indices;
  Value mask;
  Value value;
};

static bool areEquivalentValues(Value lhs, Value rhs);

static bool areEquivalentValueRanges(ArrayRef<Value> lhs, ArrayRef<Value> rhs) {
  return lhs.size() == rhs.size() &&
         llvm::all_of(llvm::zip(lhs, rhs), [](auto pair) {
           return areEquivalentValues(std::get<0>(pair), std::get<1>(pair));
         });
}

static SmallVector<Value, 4> mapValues(ValueRange values, IRMapping &mapping) {
  SmallVector<Value, 4> mapped;
  mapped.reserve(values.size());
  for (Value value : values)
    mapped.push_back(mapping.lookupOrDefault(value));
  return mapped;
}

static Value mapValueOrSelf(Value value, IRMapping &mapping) {
  return mapping.lookupOrDefault(value);
}

static bool sameForHeader(scf::ForOp lhs, scf::ForOp rhs) {
  return areEquivalentValues(lhs.getLowerBound(), rhs.getLowerBound()) &&
         areEquivalentValues(lhs.getUpperBound(), rhs.getUpperBound()) &&
         areEquivalentValues(lhs.getStep(), rhs.getStep()) &&
         lhs.getInitArgs().empty() && rhs.getInitArgs().empty() &&
         lhs->getAttrs() == rhs->getAttrs();
}

static bool isPureNoRegionOp(Operation *op) {
  return op->getNumRegions() == 0 && isMemoryEffectFree(op);
}

static bool isSupportedLeafOp(Operation *op) {
  if (isa<vector::MaskedLoadOp, vector::MaskedStoreOp>(op))
    return true;
  return isPureNoRegionOp(op);
}

static bool isInterstageSetupOp(Operation *op) {
  if (isa<pto::SimdTileToMemrefOp>(op))
    return true;
  return isPureNoRegionOp(op);
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

static LogicalResult analyzeLeafLoopBody(Block &body,
                                         SmallVectorImpl<Operation *> &leafOps,
                                         vector::MaskedStoreOp &store) {
  int storesSeen = 0;
  for (Operation &op : body.without_terminator()) {
    if (!isSupportedLeafOp(&op))
      return failure();
    if (auto maskedStore = dyn_cast<vector::MaskedStoreOp>(op)) {
      if (++storesSeen > 1)
        return failure();
      store = maskedStore;
    }
    leafOps.push_back(&op);
  }

  if (storesSeen != 1)
    return failure();

  for (Operation *op : leafOps)
    if (auto load = dyn_cast<vector::MaskedLoadOp>(op))
      if (!areEquivalentMaskValues(load.getMask(), store.getMask()))
        return failure();

  return success();
}

static LogicalResult analyzeStage(pto::SimdVecScopeOp scope, StageInfo &stage) {
  stage.scope = scope;

  Block &scopeBody = scope.getBody().front();
  bool seenOuterLoop = false;
  for (Operation &op : scopeBody) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (seenOuterLoop)
        return failure();
      seenOuterLoop = true;
      stage.outerLoop = loop;
      continue;
    }
    if (seenOuterLoop || !isPureNoRegionOp(&op))
      return failure();
    stage.scopePreludeOps.push_back(&op);
  }

  if (!stage.outerLoop || !stage.outerLoop.getInitArgs().empty())
    return failure();

  bool seenInnerLoop = false;
  for (Operation &op : stage.outerLoop.getBody()->without_terminator()) {
    if (auto loop = dyn_cast<scf::ForOp>(op)) {
      if (seenInnerLoop)
        return failure();
      seenInnerLoop = true;
      stage.innerLoop = loop;
      continue;
    }
    if (seenInnerLoop || !isPureNoRegionOp(&op))
      return failure();
    stage.outerLoopPreludeOps.push_back(&op);
  }

  if (stage.innerLoop) {
    if (!stage.innerLoop || !stage.innerLoop.getInitArgs().empty())
      return failure();
    return analyzeLeafLoopBody(*stage.innerLoop.getBody(), stage.leafOps,
                               stage.store);
  }

  return analyzeLeafLoopBody(*stage.outerLoop.getBody(), stage.leafOps,
                             stage.store);
}

static SmallVector<StageInfo, 8>
collectStageRunFrom(pto::SimdVecScopeOp firstScope) {
  SmallVector<StageInfo, 8> stages;

  StageInfo firstStage;
  if (failed(analyzeStage(firstScope, firstStage)))
    return stages;
  stages.push_back(std::move(firstStage));

  SmallVector<Operation *, 4> pendingSetup;
  for (Operation *op = firstScope->getNextNode(); op; op = op->getNextNode()) {
    if (auto nextScope = dyn_cast<pto::SimdVecScopeOp>(op)) {
      StageInfo nextStage;
      nextStage.setupOps = pendingSetup;
      pendingSetup.clear();
      if (failed(analyzeStage(nextScope, nextStage)))
        break;
      stages.push_back(std::move(nextStage));
      continue;
    }

    if (!isInterstageSetupOp(op))
      break;
    pendingSetup.push_back(op);
  }

  return stages;
}

static const ForwardedStore *findForwardedStore(ArrayRef<ForwardedStore> stores,
                                                Value base,
                                                ArrayRef<Value> indices,
                                                Value mask) {
  for (const ForwardedStore &store : llvm::reverse(stores))
    if (areEquivalentValues(store.base, base) &&
        areEquivalentValueRanges(store.indices, indices) &&
        areEquivalentMaskValues(store.mask, mask))
      return &store;
  return nullptr;
}

static bool sameLoopNestShape(const StageInfo &lhs, const StageInfo &rhs) {
  if (lhs.hasInnerLoop() != rhs.hasInnerLoop())
    return false;
  if (!sameForHeader(lhs.outerLoop, rhs.outerLoop))
    return false;
  if (lhs.hasInnerLoop() && !sameForHeader(lhs.innerLoop, rhs.innerLoop))
    return false;
  return true;
}

static void cloneOpAndMapResults(OpBuilder &builder, Operation *op,
                                 IRMapping &mapping) {
  Operation *cloned = builder.clone(*op, mapping);
  for (auto [oldRes, newRes] :
       llvm::zip(op->getResults(), cloned->getResults()))
    mapping.map(oldRes, newRes);
}

static void
cloneStageLeafOps(OpBuilder &builder, const StageInfo &stage,
                  IRMapping &mapping, ArrayRef<unsigned> lastStoreStage,
                  unsigned stageIndex,
                  SmallVectorImpl<ForwardedStore> &forwardedStores) {
  for (Operation *op : stage.leafOps) {
    if (auto load = dyn_cast<vector::MaskedLoadOp>(op)) {
      Value base = mapping.lookupOrDefault(load.getBase());
      SmallVector<Value, 4> indices = mapValues(load.getIndices(), mapping);
      Value mask = mapping.lookupOrDefault(load.getMask());
      Value passThru = mapping.lookupOrDefault(load.getPassThru());

      if (const ForwardedStore *forwarded =
              findForwardedStore(forwardedStores, base, indices, mask)) {
        mapping.map(load.getResult(), forwarded->value);
        continue;
      }

      auto cloned = builder.create<vector::MaskedLoadOp>(
          load.getLoc(), load.getResult().getType(), base, indices, mask,
          passThru);
      cloned->setAttrs(load->getAttrs());
      mapping.map(load.getResult(), cloned.getResult());
      continue;
    }

    if (auto store = dyn_cast<vector::MaskedStoreOp>(op)) {
      Value base = mapping.lookupOrDefault(store.getBase());
      SmallVector<Value, 4> indices = mapValues(store.getIndices(), mapping);
      Value mask = mapping.lookupOrDefault(store.getMask());
      Value value = mapping.lookupOrDefault(store.getValueToStore());

      if (lastStoreStage[stageIndex] == stageIndex) {
        auto cloned = builder.create<vector::MaskedStoreOp>(
            store.getLoc(), base, indices, mask, value);
        cloned->setAttrs(store->getAttrs());
      }

      forwardedStores.push_back(ForwardedStore{
          base, SmallVector<Value, 2>(indices.begin(), indices.end()), mask,
          value});
      continue;
    }

    cloneOpAndMapResults(builder, op, mapping);
  }
}

static bool fuseStageRun(SmallVectorImpl<StageInfo> &stages) {
  if (stages.size() < 2)
    return false;

  StageInfo &first = stages.front();
  for (StageInfo &stage : llvm::drop_begin(stages)) {
    if (stage.scope->getAttrs() != first.scope->getAttrs())
      return false;
    if (!sameLoopNestShape(first, stage))
      return false;
  }

  SmallVector<unsigned, 8> lastStoreStage(stages.size());
  for (auto [index, stage] : llvm::enumerate(stages)) {
    unsigned last = index;
    for (unsigned next = index + 1; next < stages.size(); ++next) {
      if (areEquivalentValues(stage.store.getBase(),
                              stages[next].store.getBase()))
        last = next;
    }
    lastStoreStage[index] = last;
  }

  OpBuilder blockBuilder(first.scope);
  auto fusedScope =
      blockBuilder.create<pto::SimdVecScopeOp>(first.scope.getLoc());
  fusedScope->setAttrs(first.scope->getAttrs());

  for (StageInfo &stage : llvm::drop_begin(stages))
    for (Operation *setupOp : stage.setupOps)
      setupOp->moveBefore(fusedScope);

  Block *scopeBody = new Block();
  fusedScope.getBody().push_back(scopeBody);
  OpBuilder scopeBuilder = OpBuilder::atBlockBegin(scopeBody);

  SmallVector<IRMapping, 8> stageMappings(stages.size());
  for (auto [index, stage] : llvm::enumerate(stages))
    for (Operation *op : stage.scopePreludeOps)
      cloneOpAndMapResults(scopeBuilder, op, stageMappings[index]);

  auto fusedOuterLoop = scopeBuilder.create<scf::ForOp>(
      first.outerLoop.getLoc(),
      mapValueOrSelf(first.outerLoop.getLowerBound(), stageMappings.front()),
      mapValueOrSelf(first.outerLoop.getUpperBound(), stageMappings.front()),
      mapValueOrSelf(first.outerLoop.getStep(), stageMappings.front()));
  fusedOuterLoop->setAttrs(first.outerLoop->getAttrs());

  for (auto [index, stage] : llvm::enumerate(stages))
    stageMappings[index].map(stage.outerLoop.getInductionVar(),
                             fusedOuterLoop.getInductionVar());

  OpBuilder outerBodyBuilder =
      OpBuilder::atBlockBegin(fusedOuterLoop.getBody());
  for (auto [index, stage] : llvm::enumerate(stages))
    for (Operation *op : stage.outerLoopPreludeOps)
      cloneOpAndMapResults(outerBodyBuilder, op, stageMappings[index]);

  SmallVector<ForwardedStore, 8> forwardedStores;
  if (first.hasInnerLoop()) {
    auto fusedInnerLoop = outerBodyBuilder.create<scf::ForOp>(
        first.innerLoop.getLoc(),
        mapValueOrSelf(first.innerLoop.getLowerBound(), stageMappings.front()),
        mapValueOrSelf(first.innerLoop.getUpperBound(), stageMappings.front()),
        mapValueOrSelf(first.innerLoop.getStep(), stageMappings.front()));
    fusedInnerLoop->setAttrs(first.innerLoop->getAttrs());

    for (auto [index, stage] : llvm::enumerate(stages))
      stageMappings[index].map(stage.innerLoop.getInductionVar(),
                               fusedInnerLoop.getInductionVar());

    OpBuilder innerBodyBuilder =
        OpBuilder::atBlockBegin(fusedInnerLoop.getBody());
    for (auto [index, stage] : llvm::enumerate(stages))
      cloneStageLeafOps(innerBodyBuilder, stage, stageMappings[index],
                        lastStoreStage, index, forwardedStores);
  } else {
    for (auto [index, stage] : llvm::enumerate(stages))
      cloneStageLeafOps(outerBodyBuilder, stage, stageMappings[index],
                        lastStoreStage, index, forwardedStores);
  }

  for (StageInfo &stage : llvm::reverse(stages))
    stage.scope.erase();

  return true;
}

static bool fuseStageRunsInBlock(Block &block) {
  bool changed = false;
  bool localChange = true;

  while (localChange) {
    localChange = false;
    for (Operation &op : block) {
      auto firstScope = dyn_cast<pto::SimdVecScopeOp>(op);
      if (!firstScope)
        continue;

      SmallVector<StageInfo, 8> stages = collectStageRunFrom(firstScope);
      if (!fuseStageRun(stages))
        continue;

      changed = true;
      localChange = true;
      break;
    }
  }

  return changed;
}

static bool fuseStageRunsInRegion(Region &region) {
  bool changed = false;

  for (Block &block : region.getBlocks()) {
    SmallVector<Region *, 4> nestedRegions;
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        nestedRegions.push_back(&nested);

    for (Region *nested : nestedRegions)
      if (fuseStageRunsInRegion(*nested))
        changed = true;

    if (fuseStageRunsInBlock(block))
      changed = true;
  }

  return changed;
}

struct PTOLowLevelLoopFusionPass
    : public pto::impl::PTOLowLevelLoopFusionBase<PTOLowLevelLoopFusionPass> {
  using pto::impl::PTOLowLevelLoopFusionBase<
      PTOLowLevelLoopFusionPass>::PTOLowLevelLoopFusionBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    int fusedFuncs = 0;
    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;
      if (func.empty())
        continue;

      if (fuseStageRunsInRegion(func.getBody()))
        ++fusedFuncs;
    }

    if (debug) {
      llvm::errs() << "[op-fusion] low-level loop fusion changed " << fusedFuncs
                   << " function(s)\n";
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options) {
  return std::make_unique<PTOLowLevelLoopFusionPass>(options);
}
