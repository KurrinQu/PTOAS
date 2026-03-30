#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVPTOTRIVIALLOOPCANONICALIZE
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static std::optional<int64_t> computeConstDiff(Value lowerBound,
                                               Value upperBound) {
  IntegerAttr constLowerBound;
  IntegerAttr constUpperBound;
  if (matchPattern(lowerBound, m_Constant(&constLowerBound)) &&
      matchPattern(upperBound, m_Constant(&constUpperBound))) {
    llvm::APInt lowerBoundValue = constLowerBound.getValue();
    llvm::APInt upperBoundValue = constUpperBound.getValue();
    return (upperBoundValue - lowerBoundValue).getSExtValue();
  }

  llvm::APInt diff;
  if (matchPattern(
          upperBound,
          m_Op<arith::AddIOp>(matchers::m_Val(lowerBound), m_ConstantInt(&diff))) ||
      matchPattern(
          upperBound,
          m_Op<arith::AddIOp>(m_ConstantInt(&diff), matchers::m_Val(lowerBound))))
    return diff.getSExtValue();
  return std::nullopt;
}

static void replaceOpWithRegion(IRRewriter &rewriter, Operation *op,
                                Region &region, ValueRange blockArgs = {}) {
  assert(llvm::hasSingleElement(region) && "expected a single-block region");
  Block *block = &region.front();
  Operation *terminator = block->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(block, op, blockArgs);
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

static LogicalResult simplifyTrivialLoop(scf::ForOp loop,
                                         IRRewriter &rewriter) {
  if (!loop || isa_and_nonnull<pto::VecScopeOp>(loop->getParentOp()))
    return failure();

  if (loop.getLowerBound() == loop.getUpperBound()) {
    rewriter.replaceOp(loop, loop.getInitArgs());
    return success();
  }

  std::optional<int64_t> diff =
      computeConstDiff(loop.getLowerBound(), loop.getUpperBound());
  if (!diff)
    return failure();

  if (*diff <= 0) {
    rewriter.replaceOp(loop, loop.getInitArgs());
    return success();
  }

  std::optional<llvm::APInt> maybeStepValue = loop.getConstantStep();
  if (!maybeStepValue)
    return failure();

  if (maybeStepValue->sge(*diff)) {
    SmallVector<Value, 4> blockArgs;
    blockArgs.reserve(loop.getInitArgs().size() + 1);
    blockArgs.push_back(loop.getLowerBound());
    llvm::append_range(blockArgs, loop.getInitArgs());
    replaceOpWithRegion(rewriter, loop, loop.getRegion(), blockArgs);
    return success();
  }

  Block &body = loop.getRegion().front();
  if (!llvm::hasSingleElement(body))
    return failure();

  if (llvm::any_of(loop.getYieldedValues(), [&](Value value) {
        return !loop.isDefinedOutsideOfLoop(value);
      }))
    return failure();

  rewriter.replaceOp(loop, loop.getYieldedValues());
  return success();
}

struct PTOVPTOTrivialLoopCanonicalizePass
    : public pto::impl::PTOVPTOTrivialLoopCanonicalizeBase<
          PTOVPTOTrivialLoopCanonicalizePass> {
  using pto::impl::PTOVPTOTrivialLoopCanonicalizeBase<
      PTOVPTOTrivialLoopCanonicalizePass>::PTOVPTOTrivialLoopCanonicalizeBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.isExternal())
      return;

    SmallVector<scf::ForOp, 16> loops;
    func.walk([&](scf::ForOp loop) { loops.push_back(loop); });

    IRRewriter rewriter(func.getContext());
    for (scf::ForOp loop : llvm::reverse(loops)) {
      if (!loop || !loop->getBlock())
        continue;
      rewriter.setInsertionPoint(loop);
      (void)simplifyTrivialLoop(loop, rewriter);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOVPTOTrivialLoopCanonicalizePass() {
  return std::make_unique<PTOVPTOTrivialLoopCanonicalizePass>();
}
