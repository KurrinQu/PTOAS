// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "VPTOLLVMEmitterInternal.h"

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"

namespace mlir::pto {

class LowerTrapOpPattern final : public OpConversionPattern<pto::TrapOp> {
public:
  explicit LowerTrapOpPattern(TypeConverter &typeConverter,
                              MLIRContext *context, LoweringState &state)
      : OpConversionPattern<pto::TrapOp>(typeConverter, context),
        state(state) {}

  LogicalResult
  matchAndRewrite(pto::TrapOp op, pto::TrapOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    constexpr StringLiteral calleeName = "llvm.hivm.TRAP";
    auto funcType = rewriter.getFunctionType({}, {});
    rewriter.create<func::CallOp>(op.getLoc(), calleeName, TypeRange{},
                                   ValueRange{});
    state.plannedDecls.push_back(PlannedDecl{calleeName.str(), funcType});
    rewriter.eraseOp(op);
    return success();
  }

private:
  LoweringState &state;
};

void populateVPTOBasicPatterns(TypeConverter &typeConverter,
                                RewritePatternSet &patterns,
                                LoweringState &state) {
  patterns.add<LowerTrapOpPattern>(typeConverter, patterns.getContext(),
                                   state);
}

} // namespace mlir::pto
