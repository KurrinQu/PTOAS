#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::pto;

namespace {

/// Compute domain classification for an operation.
enum class ComputeDomain { CUBE, VECTOR, SHARED };

class CVClassifyAndSplitPass
    : public PassWrapper<CVClassifyAndSplitPass,
                         OperationPass<func::FuncOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CVClassifyAndSplitPass)

  StringRef getArgument() const override {
    return "pto-cv-classify-and-split";
  }
  StringRef getDescription() const override {
    return "Classify ops into Cube/Vector domains and split into sections";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<pto::PTODialect>();
    registry.insert<func::FuncDialect>();
    registry.insert<scf::SCFDialect>();
  }

  void runOnOperation() override {
    // TODO: implement in subsequent tasks
  }
};

} // namespace

namespace mlir {
namespace pto {
std::unique_ptr<Pass> createCVClassifyAndSplitPass() {
  return std::make_unique<CVClassifyAndSplitPass>();
}
} // namespace pto
} // namespace mlir
