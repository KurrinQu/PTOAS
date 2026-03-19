//===- PTOToA5VM.cpp - PTO to A5VM pass wiring ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"
#include "PTO/Transforms/Passes.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DEF_PTOTOA5VM
#include "PTO/Transforms/Passes.h.inc"

namespace {

template <typename OpTy, LogicalResult (*LowerFn)(OpTy, PatternRewriter &)>
struct LowerPTOOpPattern : public OpRewritePattern<OpTy> {
  using OpRewritePattern<OpTy>::OpRewritePattern;

  LogicalResult matchAndRewrite(OpTy op,
                                PatternRewriter &rewriter) const override {
    if (failed(LowerFn(op, rewriter)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct PTOToA5VMPass : public impl::PTOToA5VMBase<PTOToA5VMPass> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PTOToA5VMPass)

  void runOnOperation() override {
    ModuleOp module = getOperation();
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerPTOOpPattern<TLoadOp, lowerTLOAD>,
                 LowerPTOOpPattern<TAbsOp, lowerTABS>,
                 LowerPTOOpPattern<TStoreOp, lowerTSTORE>>(&getContext());

    ConversionTarget target(getContext());
    target.addLegalDialect<a5vm::A5VMDialect, arith::ArithDialect,
                           func::FuncDialect, memref::MemRefDialect,
                           scf::SCFDialect>();
    target.addLegalDialect<pto::PTODialect>();
    target.addIllegalOp<TLoadOp, TAbsOp, TStoreOp>();
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createLowerPTOToA5VMPass() {
  return std::make_unique<PTOToA5VMPass>();
}

} // namespace pto
} // namespace mlir
