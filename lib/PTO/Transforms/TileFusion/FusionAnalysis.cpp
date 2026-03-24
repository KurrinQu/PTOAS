#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {

namespace {

static int64_t getConstantIndexOrDynamic(Value value) {
  if (!value)
    return ShapedType::kDynamic;
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  return ShapedType::kDynamic;
}

static SmallVector<int64_t, 4> getValidShapeVec(Type type) {
  if (auto tileType = dyn_cast<pto::TileBufType>(type)) {
    return SmallVector<int64_t, 4>(tileType.getValidShape().begin(),
                                   tileType.getValidShape().end());
  }
  if (auto shapedType = dyn_cast<ShapedType>(type)) {
    return SmallVector<int64_t, 4>(shapedType.getShape().begin(),
                                   shapedType.getShape().end());
  }
  return {};
}

static SmallVector<int64_t, 4> getValidShapeVec(Value value) {
  SmallVector<int64_t, 4> validShape = getValidShapeVec(value.getType());
  if (auto bind = value.getDefiningOp<pto::BindTileOp>()) {
    if (validShape.size() >= 1 && bind.getValidRow())
      validShape[0] = getConstantIndexOrDynamic(bind.getValidRow());
    if (validShape.size() >= 2 && bind.getValidCol())
      validShape[1] = getConstantIndexOrDynamic(bind.getValidCol());
  }
  return validShape;
}

struct Rank2IterationSpace {
  int64_t rows = ShapedType::kDynamic;
  int64_t cols = ShapedType::kDynamic;
};

static std::optional<Rank2IterationSpace> getRank2IterationSpace(Value value) {
  SmallVector<int64_t, 4> validShape = getValidShapeVec(value);
  if (validShape.size() < 2)
    return std::nullopt;
  return Rank2IterationSpace{validShape[0], validShape[1]};
}

static void mergeIterationDim(int64_t &mergedDim, int64_t dim,
                              IterationDomainInfo &info) {
  if (mergedDim == ShapedType::kDynamic || dim == ShapedType::kDynamic) {
    mergedDim = ShapedType::kDynamic;
    if (info.unprovenReason == IterationDomainUnprovenReason::None)
      info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
    return;
  }

  if (mergedDim != dim) {
    mergedDim = ShapedType::kDynamic;
    info.unprovenReason = IterationDomainUnprovenReason::InconsistentShape;
  }
}

static IterationDomainInfo inferConsensusIterationDomain(
    ArrayRef<Value> anchorValues) {
  IterationDomainInfo info;
  info.unprovenReason = IterationDomainUnprovenReason::None;

  if (anchorValues.empty())
    return info;

  std::optional<Rank2IterationSpace> firstSpace =
      getRank2IterationSpace(anchorValues.front());
  if (!firstSpace)
    return info;

  info.vRow = firstSpace->rows;
  info.vCol = firstSpace->cols;

  if (info.vRow == ShapedType::kDynamic || info.vCol == ShapedType::kDynamic)
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;

  for (Value value : ArrayRef<Value>(anchorValues).drop_front()) {
    std::optional<Rank2IterationSpace> space = getRank2IterationSpace(value);
    if (!space) {
      info.vRow = ShapedType::kDynamic;
      info.vCol = ShapedType::kDynamic;
      info.unprovenReason = IterationDomainUnprovenReason::MissingTileDomain;
      return info;
    }
    mergeIterationDim(info.vRow, space->rows, info);
    mergeIterationDim(info.vCol, space->cols, info);
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None &&
      info.vRow != ShapedType::kDynamic && info.vCol != ShapedType::kDynamic) {
    info.proof = IterationDomainProof::Proven;
    return info;
  }

  if (info.unprovenReason == IterationDomainUnprovenReason::None)
    info.unprovenReason = IterationDomainUnprovenReason::DynamicShape;
  return info;
}

static IterationDomainInfo
inferIterationDomainInfo(const FusionOpSemantics &semantics) {
  // Current pre-fusion analysis only extracts an op's iteration domain from
  // family-aware tile valid-shape anchors; it does not try to prove symbolic
  // equality for dynamic v_row/v_col. As a result, when all relevant tiles are
  // dynamic-shape, the domain stays conservatively Unproven until a later
  // shape-inference stage refines it.
  switch (semantics.computeFamily) {
  case FusionComputeFamily::Elementwise: {
    SmallVector<Value, 6> anchors;
    anchors.append(semantics.tileInputs.begin(), semantics.tileInputs.end());
    anchors.append(semantics.tileOutputs.begin(), semantics.tileOutputs.end());
    return inferConsensusIterationDomain(anchors);
  }
  case FusionComputeFamily::ScalarExpand:
  case FusionComputeFamily::RowBroadcastBinary:
    return inferConsensusIterationDomain(semantics.tileOutputs);
  case FusionComputeFamily::ReduceRow:
  case FusionComputeFamily::ReduceCol:
    return inferConsensusIterationDomain(semantics.tileInputs);
  case FusionComputeFamily::Unknown:
    return IterationDomainInfo();
  }
  return IterationDomainInfo();
}

static unsigned assignIterationDomainClass(
    SmallVectorImpl<IterationDomainClass> &classes,
    DenseMap<std::pair<int64_t, int64_t>, unsigned> &provenClassByKey,
    const IterationDomainInfo &info, unsigned nodeId) {
  if (info.proof == IterationDomainProof::Proven) {
    std::pair<int64_t, int64_t> key{info.vRow, info.vCol};
    auto it = provenClassByKey.find(key);
    if (it != provenClassByKey.end()) {
      classes[it->second].members.push_back(nodeId);
      return it->second;
    }

    unsigned classId = classes.size();
    IterationDomainClass klass;
    klass.id = classId;
    klass.info = info;
    klass.members.push_back(nodeId);
    classes.push_back(std::move(klass));
    provenClassByKey.try_emplace(key, classId);
    return classId;
  }

  unsigned classId = classes.size();
  IterationDomainClass klass;
  klass.id = classId;
  klass.info = info;
  klass.members.push_back(nodeId);
  classes.push_back(std::move(klass));
  return classId;
}

struct MutableLiveness {
  FusionValueLiveness live;
};

static unsigned getOrCreateLivenessSlot(DenseMap<Value, unsigned> &slotByValue,
                                        SmallVectorImpl<MutableLiveness> &slots,
                                        Value value) {
  auto [it, inserted] = slotByValue.try_emplace(value, slots.size());
  if (inserted) {
    MutableLiveness state;
    state.live.value = value;
    slots.push_back(std::move(state));
  }
  return it->second;
}

static void appendUniqueNode(SmallVectorImpl<unsigned> &nodes, unsigned nodeId) {
  if (!llvm::is_contained(nodes, nodeId))
    nodes.push_back(nodeId);
}

static void finalizeBlockLiveness(
    Block &block, DenseMap<Operation *, FusionOpKind> &kindByOp,
    DenseMap<Operation *, unsigned> &computeNodeByOp,
    SmallVectorImpl<MutableLiveness> &mutableLiveness) {
  for (MutableLiveness &state : mutableLiveness) {
    for (OpOperand &use : state.live.value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getBlock() != &block) {
        state.live.hasExternalUsers = true;
        state.live.escapesBlock = true;
        continue;
      }

      auto kindIt = kindByOp.find(user);
      if (kindIt == kindByOp.end())
        continue;

      if (user->hasTrait<OpTrait::IsTerminator>())
        state.live.escapesBlock = true;

      switch (kindIt->second) {
      case FusionOpKind::Compute: {
        auto nodeIt = computeNodeByOp.find(user);
        if (nodeIt == computeNodeByOp.end())
          continue;
        appendUniqueNode(state.live.consumerNodes, nodeIt->second);
        state.live.lastLocalConsumer = nodeIt->second;
        break;
      }
      case FusionOpKind::LocalBoundary:
        state.live.hasLocalBoundaryUsers = true;
        break;
      case FusionOpKind::HardBoundary:
        state.live.hasLocalHardBoundaryUsers = true;
        break;
      }
    }
  }
}

static FailureOr<FusionBlockAnalysis> analyzeBlock(Block &block) {
  FusionBlockAnalysis analysis;
  analysis.block = &block;

  DenseMap<Value, unsigned> producerByValue;
  DenseMap<Value, unsigned> livenessSlotByValue;
  SmallVector<MutableLiveness, 8> mutableLiveness;
  DenseMap<Operation *, FusionOpKind> kindByOp;
  DenseMap<Operation *, unsigned> computeNodeByOp;
  DenseMap<std::pair<int64_t, int64_t>, unsigned> provenClassByKey;

  unsigned blockOrder = 0;
  for (Operation &op : block) {
    FailureOr<FusionOpSemantics> semanticsOr = getFusionOpSemantics(&op);
    if (failed(semanticsOr)) {
      op.emitError("failed to normalize fusion op semantics");
      return failure();
    }
    kindByOp[&op] = semanticsOr->kind;

    if (semanticsOr->kind == FusionOpKind::LocalBoundary) {
      // Keep local boundaries out of the compute DFG while still materializing
      // their input/output values in liveness so planning can see that the
      // dependency chain stops here without globally blocking unrelated ops.
      for (Value input : semanticsOr->tileInputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      for (Value output : semanticsOr->tileOutputs)
        getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      ++blockOrder;
      continue;
    }

    if (semanticsOr->kind != FusionOpKind::Compute) {
      ++blockOrder;
      continue;
    }

    FusionComputeNode node;
    node.id = analysis.computeNodes.size();
    node.blockOrder = blockOrder;
    node.op = &op;
    node.semantics = *semanticsOr;
    computeNodeByOp[&op] = node.id;

    // Recover the node's iteration-domain signature from normalized semantics
    // and bucket it into a reusable proven/unproven equivalence class.
    IterationDomainInfo domainInfo = inferIterationDomainInfo(node.semantics);
    node.iterationDomainClass = assignIterationDomainClass(
        analysis.iterationDomainClasses, provenClassByKey, domainInfo, node.id);

    for (Value output : node.semantics.tileOutputs) {
      producerByValue[output] = node.id;
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, output);
      mutableLiveness[liveSlot].live.producerNode = node.id;
    }

    for (Value input : node.semantics.tileInputs) {
      unsigned liveSlot =
          getOrCreateLivenessSlot(livenessSlotByValue, mutableLiveness, input);
      appendUniqueNode(mutableLiveness[liveSlot].live.consumerNodes, node.id);
      mutableLiveness[liveSlot].live.lastLocalConsumer = node.id;

      auto producerIt = producerByValue.find(input);
      if (producerIt == producerByValue.end())
        continue;

      FusionDFGEdge edge;
      edge.producerNode = producerIt->second;
      edge.consumerNode = node.id;
      edge.value = input;

      unsigned edgeId = analysis.edges.size();
      analysis.edges.push_back(edge);
      node.incomingEdges.push_back(edgeId);
      if (edge.producerNode < analysis.computeNodes.size())
        analysis.computeNodes[edge.producerNode].outgoingEdges.push_back(edgeId);
    }

    analysis.computeNodes.push_back(std::move(node));
    ++blockOrder;
  }

  finalizeBlockLiveness(block, kindByOp, computeNodeByOp, mutableLiveness);

  analysis.liveness.reserve(mutableLiveness.size());
  for (MutableLiveness &state : mutableLiveness)
    analysis.liveness.push_back(std::move(state.live));

  return std::move(analysis);
}

static LogicalResult analyzeRegion(Region &region,
                                   SmallVectorImpl<FusionBlockAnalysis> &blocks) {
  for (Block &block : region.getBlocks()) {
    FailureOr<FusionBlockAnalysis> blockAnalysis = analyzeBlock(block);
    if (failed(blockAnalysis))
      return failure();
    blocks.push_back(std::move(*blockAnalysis));
    // Analyze nested regions recursively, but keep each nested block as its own
    // block-local analysis unit instead of merging graphs across region edges.
    for (Operation &op : block)
      for (Region &nested : op.getRegions())
        if (failed(analyzeRegion(nested, blocks)))
          return failure();
  }
  return success();
}

} // namespace

FailureOr<PreFusionAnalysisResult> buildPreFusionAnalysis(func::FuncOp func) {
  PreFusionAnalysisResult result;
  if (failed(analyzeRegion(func.getRegion(), result.blocks)))
    return failure();
  return std::move(result);
}

} // namespace pto
} // namespace mlir
