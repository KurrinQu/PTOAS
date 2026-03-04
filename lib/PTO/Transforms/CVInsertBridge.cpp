#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

/// A cross-domain data dependency that needs bridging.
struct BridgePoint {
  Value producerValue;         // SSA value produced in one section
  Operation *producerSection;  // SectionCubeOp or SectionVectorOp
  SmallVector<OpOperand *> consumerUses; // uses in the other section
  Operation *consumerSection;
};

class CVInsertBridgePass
    : public PassWrapper<CVInsertBridgePass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVInsertBridgePass)

  StringRef getArgument() const override {
    return "pto-cv-insert-bridge";
  }
  StringRef getDescription() const override {
    return "Insert GM workspace bridges for cross-domain data dependencies";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    // TODO: implement in subsequent tasks
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createCVInsertBridgePass() {
  return std::make_unique<CVInsertBridgePass>();
}
} // namespace pto
} // namespace mlir
