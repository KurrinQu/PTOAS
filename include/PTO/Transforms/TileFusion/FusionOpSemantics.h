#ifndef PTO_TRANSFORMS_FUSIONOPSEMANTICS_H
#define PTO_TRANSFORMS_FUSIONOPSEMANTICS_H

#include "PTO/IR/PTO.h"

#include "mlir/Support/LLVM.h"

#include <string>

namespace mlir {
namespace pto {

// Classification used by pre-fusion analysis to decide whether an op becomes
// a compute node, a local non-through boundary (for example `pto.treshape`),
// or a hard block-local boundary.
enum class FusionOpKind {
  Compute,
  LocalBoundary,
  HardBoundary,
};

enum class FusionComputeFamily {
  Unknown,
  Elementwise,
  ScalarExpand,
  RowBroadcastBinary,
  ReduceRow,
  ReduceCol,
};

struct FusionOpSemantics {
  FusionOpKind kind = FusionOpKind::HardBoundary;
  FusionComputeFamily computeFamily = FusionComputeFamily::Unknown;
  Operation *op = nullptr;
  std::string opName;
  SmallVector<Value, 4> tileInputs;
  SmallVector<Value, 2> tileOutputs;
  SmallVector<Value, 2> scalarInputs;
};

bool isSupportedPreFusionComputeOp(StringRef opName);
FailureOr<FusionOpSemantics> getFusionOpSemantics(Operation *op);

} // namespace pto
} // namespace mlir

#endif // PTO_TRANSFORMS_FUSIONOPSEMANTICS_H
