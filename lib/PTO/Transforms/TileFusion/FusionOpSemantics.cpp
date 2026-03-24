#include "PTO/Transforms/TileFusion/FusionOpSemantics.h"

#include "llvm/ADT/StringSwitch.h"

namespace mlir {
namespace pto {

static FusionComputeFamily getFusionComputeFamily(StringRef opName) {
  return llvm::StringSwitch<FusionComputeFamily>(opName)
      .Cases("tadd", "tsub", "tmul", "tdiv", "tmax", "tmin",
             FusionComputeFamily::Elementwise)
      .Cases("tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins",
             FusionComputeFamily::Elementwise)
      .Case("texp", FusionComputeFamily::Elementwise)
      .Case("texpands", FusionComputeFamily::ScalarExpand)
      .Cases("trowexpandmul", "trowexpanddiv",
             FusionComputeFamily::RowBroadcastBinary)
      .Cases("trowsum", "trowmax", "trowmin",
             FusionComputeFamily::ReduceRow)
      .Cases("tcolsum", "tcolmax", "tcolmin",
             FusionComputeFamily::ReduceCol)
      .Default(FusionComputeFamily::Unknown);
}

bool isSupportedPreFusionComputeOp(StringRef opName) {
  return getFusionComputeFamily(opName) != FusionComputeFamily::Unknown;
}

static bool isTileFusionTileValue(Value value) {
  return isa<pto::TileBufType>(value.getType());
}

static SmallVector<Value, 2> collectNormalizedTileOutputs(Operation *op) {
  SmallVector<Value, 2> outputs;

  if (auto dpsIface = dyn_cast<pto::PTO_DpsInitOpInterface>(op)) {
    for (Value init : dpsIface.getDpsInits()) {
      if (isTileFusionTileValue(init))
        outputs.push_back(init);
    }
    if (!outputs.empty())
      return outputs;
  }

  for (Value result : op->getResults()) {
    if (isTileFusionTileValue(result))
      outputs.push_back(result);
  }
  return outputs;
}

static std::optional<unsigned>
getDescriptorOutputSuffixStart(const OpLibMatchDescriptor &desc,
                               ArrayRef<Value> normalizedOutputs) {
  if (normalizedOutputs.empty() ||
      desc.operands.size() < normalizedOutputs.size())
    return std::nullopt;

  unsigned start = desc.operands.size() - normalizedOutputs.size();
  for (auto [idx, output] : llvm::enumerate(normalizedOutputs)) {
    unsigned operandIndex = start + idx;
    if (desc.operandRoles[operandIndex] !=
        static_cast<int64_t>(OpLibArgRole::Tile))
      return std::nullopt;
    if (desc.operands[operandIndex] != output)
      return std::nullopt;
  }
  return start;
}

FailureOr<FusionOpSemantics> getFusionOpSemantics(Operation *op) {
  FusionOpSemantics semantics;
  semantics.op = op;
  semantics.opName = op->getName().getStringRef().str();

  if (auto reshape = dyn_cast<pto::TReshapeOp>(op)) {
    semantics.kind = FusionOpKind::LocalBoundary;
    semantics.opName = "treshape";
    semantics.tileInputs.push_back(reshape.getSrc());
    semantics.tileOutputs.push_back(reshape.getResult());
    return semantics;
  }

  auto oplibIface = dyn_cast<pto::OpLibOpInterface>(op);
  if (!oplibIface) {
    semantics.kind = FusionOpKind::HardBoundary;
    return semantics;
  }

  FailureOr<pto::OpLibMatchDescriptor> descOr =
      oplibIface.getOpLibMatchDescriptor();
  if (failed(descOr))
    return failure();

  const pto::OpLibMatchDescriptor &desc = *descOr;
  if (desc.operands.size() != desc.operandRoles.size())
    return failure();

  semantics.opName = desc.opName;
  semantics.computeFamily = getFusionComputeFamily(desc.opName);
  if (semantics.computeFamily == FusionComputeFamily::Unknown) {
    semantics.kind = FusionOpKind::HardBoundary;
    return semantics;
  }

  semantics.kind = FusionOpKind::Compute;
  semantics.tileOutputs = collectNormalizedTileOutputs(op);

  std::optional<unsigned> outputSuffixStart =
      getDescriptorOutputSuffixStart(desc, semantics.tileOutputs);

  for (auto [idx, operand, role] : llvm::enumerate(desc.operands,
                                                   desc.operandRoles)) {
    const bool isOutputOperand =
        outputSuffixStart && idx >= *outputSuffixStart &&
        idx < *outputSuffixStart + semantics.tileOutputs.size();

    switch (static_cast<pto::OpLibArgRole>(role)) {
    case pto::OpLibArgRole::Tile:
      if (!isOutputOperand)
        semantics.tileInputs.push_back(operand);
      break;
    case pto::OpLibArgRole::Scalar:
      semantics.scalarInputs.push_back(operand);
      break;
    default:
      return failure();
    }
  }

  if (semantics.tileOutputs.empty())
    return failure();

  return semantics;
}

} // namespace pto
} // namespace mlir
