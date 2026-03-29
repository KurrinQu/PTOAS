#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOIFCANONICALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

struct PTOVPTOIfCanonicalizePass
    : public pto::impl::PTOVPTOIfCanonicalizeBase<PTOVPTOIfCanonicalizePass> {
  using pto::impl::PTOVPTOIfCanonicalizeBase<
      PTOVPTOIfCanonicalizePass>::PTOVPTOIfCanonicalizeBase;

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

std::unique_ptr<Pass> mlir::pto::createPTOVPTOIfCanonicalizePass() {
  return std::make_unique<PTOVPTOIfCanonicalizePass>();
}
