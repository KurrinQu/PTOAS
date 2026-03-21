#ifndef PTO_TRANSFORMS_FUSIONANALYSIS_H
#define PTO_TRANSFORMS_FUSIONANALYSIS_H

#include "FusionOpSemantics.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Support/LLVM.h"

#include <optional>

namespace mlir {
namespace pto {

enum class IterationDomainProof {
  Proven,
  Unproven,
};

enum class IterationDomainUnprovenReason {
  None,
  MissingTileDomain,
  DynamicShape,
  InconsistentShape,
};

struct IterationDomainInfo {
  int64_t vRow = ShapedType::kDynamic;
  int64_t vCol = ShapedType::kDynamic;
  IterationDomainProof proof = IterationDomainProof::Unproven;
  IterationDomainUnprovenReason unprovenReason =
      IterationDomainUnprovenReason::MissingTileDomain;
};

struct IterationDomainClass {
  unsigned id = 0;
  IterationDomainInfo info;
  SmallVector<unsigned, 4> members;
};

struct FusionDFGEdge {
  unsigned producerNode = 0;
  unsigned consumerNode = 0;
  Value value;
};

struct FusionValueLiveness {
  Value value;
  std::optional<unsigned> producerNode;
  SmallVector<unsigned, 4> consumerNodes;
  std::optional<unsigned> lastLocalConsumer;
  bool hasExternalUsers = false;
  bool escapesBlock = false;
  bool hasLocalBoundaryUsers = false;
  bool hasLocalHardBoundaryUsers = false;
};

struct FusionComputeNode {
  unsigned id = 0;
  unsigned blockOrder = 0;
  Operation *op = nullptr;
  FusionOpSemantics semantics;
  unsigned iterationDomainClass = 0;
  SmallVector<unsigned, 4> incomingEdges;
  SmallVector<unsigned, 4> outgoingEdges;
};

struct FusionBlockAnalysis {
  Block *block = nullptr;
  SmallVector<FusionComputeNode, 8> computeNodes;
  SmallVector<IterationDomainClass, 4> iterationDomainClasses;
  SmallVector<FusionDFGEdge, 8> edges;
  SmallVector<FusionValueLiveness, 8> liveness;
};

struct PreFusionAnalysisResult {
  SmallVector<FusionBlockAnalysis, 8> blocks;
};

FailureOr<PreFusionAnalysisResult> buildPreFusionAnalysis(func::FuncOp func);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_FUSIONANALYSIS_H
