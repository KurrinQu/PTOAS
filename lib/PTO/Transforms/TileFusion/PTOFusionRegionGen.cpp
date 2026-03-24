#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOFUSIONREGIONGEN
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

struct GroupSpanMember {
  Operation *op = nullptr;
  int64_t order = 0;
};

struct GroupSpan {
  Block *block = nullptr;
  int64_t groupId = -1;
  SmallVector<GroupSpanMember, 8> members;
};

struct GroupSpanInterface {
  // Values that remain visible outside the fusion_region after encapsulation.
  // The final pto.yield/result list is derived exactly from this set so
  // downstream passes can treat it as the region's external visibility
  // frontier.
  SmallVector<Value, 8> externallyVisibleValues;
  SmallVector<Operation *, 8> localDefs;
};

static std::optional<int64_t> getRequiredI64Attr(Operation *op,
                                                 StringRef attrName) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(attrName))
    return attr.getInt();
  return std::nullopt;
}

static bool hasIncompleteFusionMetadata(Operation *op) {
  const bool hasGroupId = op->hasAttr(kFusionGroupIdAttr);
  const bool hasOrder = op->hasAttr(kFusionOrderAttr);
  return hasGroupId != hasOrder;
}

static LogicalResult collectGroupSpansInBlock(
    Block &block, SmallVectorImpl<GroupSpan> &spans) {
  DenseMap<int64_t, unsigned> spanIndexByGroupId;

  GroupSpan current;

  auto flush = [&]() -> LogicalResult {
    if (current.members.empty())
      return success();

    current.block = &block;
    auto [it, inserted] =
        spanIndexByGroupId.try_emplace(current.groupId, spans.size());
    if (!inserted) {
      current.members.front().op->emitError(
          "expected one contiguous span per pto.fusion.group_id within a basic "
          "block");
      return failure();
    }

    spans.push_back(std::move(current));
    current = GroupSpan();
    return success();
  };

  for (Operation &op : block) {
    if (hasIncompleteFusionMetadata(&op)) {
      op.emitError("expected pto.fusion.group_id and pto.fusion.order to "
                   "either both exist or both be absent");
      return failure();
    }

    std::optional<int64_t> groupId = getRequiredI64Attr(&op, kFusionGroupIdAttr);
    if (!groupId) {
      if (failed(flush()))
        return failure();
      continue;
    }

    std::optional<int64_t> order = getRequiredI64Attr(&op, kFusionOrderAttr);
    if (!order) {
      op.emitError("missing required pto.fusion.order attribute");
      return failure();
    }

    if (current.members.empty()) {
      current.groupId = *groupId;
      current.members.push_back(GroupSpanMember{&op, *order});
      continue;
    }

    if (current.groupId != *groupId) {
      if (failed(flush()))
        return failure();
      current.groupId = *groupId;
    }

    if (!current.members.empty() && current.members.back().order >= *order) {
      op.emitError("expected contiguous fusion span to follow increasing "
                   "pto.fusion.order");
      return failure();
    }

    current.members.push_back(GroupSpanMember{&op, *order});
  }

  return flush();
}

static bool isNestedInOp(Operation *op, Operation *ancestor) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur == ancestor)
      return true;
  return false;
}

static bool isNestedInSpan(Operation *op, const DenseSet<Operation *> &spanOps) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (spanOps.contains(cur))
      return true;
  return false;
}

static void appendUniqueValue(SmallVectorImpl<Value> &values,
                              DenseSet<Value> &seen, Value value) {
  if (seen.insert(value).second)
    values.push_back(value);
}

static Operation *getTopLevelAncestorInBlock(Operation *op, Block *block) {
  for (Operation *cur = op; cur; cur = cur->getParentOp())
    if (cur->getBlock() == block)
      return cur;
  return nullptr;
}

static bool canReplaceUseWithRegionResult(OpOperand &use, Operation *boundary) {
  Operation *topLevel = getTopLevelAncestorInBlock(use.getOwner(),
                                                   boundary->getBlock());
  if (!topLevel || topLevel == boundary)
    return false;
  return boundary->isBeforeInBlock(topLevel);
}

static bool hasReplaceableUseOutsideSpan(Value value,
                                         const DenseSet<Operation *> &spanOps,
                                         Operation *boundary) {
  for (OpOperand &use : value.getUses()) {
    if (isNestedInSpan(use.getOwner(), spanOps))
      continue;
    if (canReplaceUseWithRegionResult(use, boundary))
      return true;
  }
  return false;
}

static bool canSinkAllocTileDefToRegion(Value value, const GroupSpan &span,
                                        const DenseSet<Operation *> &spanOps) {
  auto alloc = dyn_cast_or_null<pto::AllocTileOp>(value.getDefiningOp());
  if (!alloc || alloc->getBlock() != span.block)
    return false;

  Operation *firstOp = span.members.front().op;
  if (!alloc->isBeforeInBlock(firstOp))
    return false;

  for (OpOperand &use : value.getUses()) {
    if (isNestedInSpan(use.getOwner(), spanOps))
      continue;
    if (!canReplaceUseWithRegionResult(use, firstOp))
      return false;
  }

  return true;
}

static GroupSpanInterface buildGroupSpanInterface(const GroupSpan &span) {
  GroupSpanInterface iface;
  DenseSet<Value> seenOutputs;
  DenseSet<Operation *> spanOps;
  DenseSet<Operation *> seenLocalDefs;
  Operation *boundary = span.members.front().op;

  for (const GroupSpanMember &member : span.members)
    spanOps.insert(member.op);

  for (const GroupSpanMember &member : span.members) {
    if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(member.op)) {
      for (Value init : dpsIface.getDpsInits()) {
        if (!canSinkAllocTileDefToRegion(init, span, spanOps))
          continue;
        Operation *defOp = init.getDefiningOp();
        if (seenLocalDefs.insert(defOp).second) {
          iface.localDefs.push_back(defOp);
        }
      }
    }
  }

  // Keep only values that are still used outside the scheduled span. Values
  // without replaceable outside uses stay internal to the region and must not
  // be yielded. Enumeration follows span order so the pto.yield/result order
  // is stable.
  for (const GroupSpanMember &member : span.members) {
    for (Value result : member.op->getResults())
      if (hasReplaceableUseOutsideSpan(result, spanOps, boundary))
        appendUniqueValue(iface.externallyVisibleValues, seenOutputs, result);

    if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(member.op)) {
      for (Value init : dpsIface.getDpsInits())
        if (hasReplaceableUseOutsideSpan(init, spanOps, boundary))
          appendUniqueValue(iface.externallyVisibleValues, seenOutputs, init);
    }
  }

  return iface;
}

static void replaceEscapingUsesOutsideRegion(pto::FusionRegionOp fusionRegion,
                                             ArrayRef<Value> oldValues) {
  for (auto [oldValueRef, newValue] :
       llvm::zip(oldValues, fusionRegion.getOutputs())) {
    Value oldValue = oldValueRef;
    oldValue.replaceUsesWithIf(newValue, [&](OpOperand &use) {
      return !isNestedInOp(use.getOwner(), fusionRegion.getOperation()) &&
             canReplaceUseWithRegionResult(use, fusionRegion.getOperation());
    });
  }
}

static void clearSpanFusionMetadata(const GroupSpan &span) {
  for (const GroupSpanMember &member : span.members) {
    member.op->removeAttr(kFusionGroupIdAttr);
    member.op->removeAttr(kFusionOrderAttr);
  }
}

static LogicalResult encapsulateGroupSpan(const GroupSpan &span) {
  if (span.members.empty())
    return success();

  GroupSpanInterface iface = buildGroupSpanInterface(span);

  SmallVector<Type, 8> outputTypes;
  outputTypes.reserve(iface.externallyVisibleValues.size());
  for (Value output : iface.externallyVisibleValues)
    outputTypes.push_back(output.getType());

  Operation *firstOp = span.members.front().op;
  Location loc = firstOp->getLoc();
  OpBuilder builder(firstOp);
  auto fusionRegion = builder.create<pto::FusionRegionOp>(loc,
                                                          TypeRange(outputTypes));
  fusionRegion->setAttr(kFusionGroupIdAttr,
                        builder.getI64IntegerAttr(span.groupId));

  Block *body = new Block();
  fusionRegion.getBody().push_back(body);

  for (Operation *localDef : iface.localDefs)
    localDef->moveBefore(body, body->end());
  for (const GroupSpanMember &member : span.members)
    member.op->moveBefore(body, body->end());

  clearSpanFusionMetadata(span);

  SmallVector<Value, 8> yieldValues;
  yieldValues.reserve(iface.externallyVisibleValues.size());
  for (Value output : iface.externallyVisibleValues)
    yieldValues.push_back(output);

  OpBuilder bodyBuilder = OpBuilder::atBlockEnd(body);
  bodyBuilder.create<pto::YieldOp>(loc, ValueRange(yieldValues));

  if (failed(verify(fusionRegion.getOperation())))
    return failure();

  replaceEscapingUsesOutsideRegion(fusionRegion, iface.externallyVisibleValues);
  return success();
}

static LogicalResult processRegion(Region &region) {
  for (Block &block : region.getBlocks()) {
    SmallVector<Region *, 4> nestedRegions;
    for (Operation &op : block)
      for (Region &nestedRegion : op.getRegions())
        nestedRegions.push_back(&nestedRegion);

    for (Region *nestedRegion : nestedRegions)
      if (failed(processRegion(*nestedRegion)))
        return failure();

    SmallVector<GroupSpan, 8> spans;
    if (failed(collectGroupSpansInBlock(block, spans)))
      return failure();

    for (const GroupSpan &span : spans)
      if (failed(encapsulateGroupSpan(span)))
        return failure();
  }
  return success();
}

struct PTOFusionRegionGenPass
    : public pto::impl::PTOFusionRegionGenBase<PTOFusionRegionGenPass> {
  using pto::impl::PTOFusionRegionGenBase<
      PTOFusionRegionGenPass>::PTOFusionRegionGenBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    if (failed(processRegion(func.getRegion())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOFusionRegionGenPass() {
  return std::make_unique<PTOFusionRegionGenPass>();
}
