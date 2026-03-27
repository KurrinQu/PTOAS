#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOA5VMIFCANONICALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PTOA5VMIfCanonicalizePass
    : public pto::impl::PTOA5VMIfCanonicalizeBase<
          PTOA5VMIfCanonicalizePass> {
  using pto::impl::PTOA5VMIfCanonicalizeBase<
      PTOA5VMIfCanonicalizePass>::PTOA5VMIfCanonicalizeBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<Operation *, 8> ifOps;
    func.walk([&](scf::IfOp ifOp) { ifOps.push_back(ifOp.getOperation()); });
    if (ifOps.empty())
      return;

    RewritePatternSet patterns(&getContext());
    scf::IfOp::getCanonicalizationPatterns(patterns, &getContext());
    FrozenRewritePatternSet frozenPatterns(std::move(patterns));

    GreedyRewriteConfig config;
    config.strictMode = GreedyRewriteStrictness::ExistingAndNewOps;
    config.scope = &func.getBody();

    if (failed(applyOpPatternsAndFold(ifOps, frozenPatterns, config)))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOA5VMIfCanonicalizePass() {
  return std::make_unique<PTOA5VMIfCanonicalizePass>();
}
