#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"
#include "PTO/Transforms/TileFusion/FusionAnalysis.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PREFUSIONANALYSIS
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PreFusionAnalysisPass
    : public pto::impl::PreFusionAnalysisBase<PreFusionAnalysisPass> {
  using pto::impl::PreFusionAnalysisBase<
      PreFusionAnalysisPass>::PreFusionAnalysisBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    const auto &analysis = getAnalysis<pto::PreFusionAnalysis>();
    if (!analysis.isValid()) {
      signalPassFailure();
      return;
    }

    markAllAnalysesPreserved();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPreFusionAnalysisPass() {
  return std::make_unique<PreFusionAnalysisPass>();
}
