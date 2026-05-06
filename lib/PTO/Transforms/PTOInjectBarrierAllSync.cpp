// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===- PTOInjectBarrierAllSync.cpp - Conservative sync barriers -----------===//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/Passes.h"
#include "PTO/IR/PTO.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir {
namespace pto {
namespace func = ::mlir::func;

#define GEN_PASS_DEF_PTOINJECTBARRIERALLSYNC
#include "PTO/Transforms/Passes.h.inc"

} // namespace pto
} // namespace mlir

using namespace mlir;
using namespace mlir::pto;

namespace {

static bool hasReadOrWriteMemoryEffect(Operation *op) {
  auto memEffect = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffect)
    return false;

  SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>, 4> effects;
  memEffect.getEffects(effects);
  return llvm::any_of(effects, [](const auto &effect) {
    return isa<MemoryEffects::Read, MemoryEffects::Write>(effect.getEffect());
  });
}

static bool shouldInjectBarrierAllBefore(Operation *op) {
  Dialect *dialect = op->getDialect();
  if (!dialect ||
      dialect->getNamespace() != pto::PTODialect::getDialectNamespace())
    return false;

  return isa<pto::OpPipeInterface>(op) && hasReadOrWriteMemoryEffect(op);
}

struct PTOInjectBarrierAllSyncPass
    : public mlir::pto::impl::PTOInjectBarrierAllSyncBase<
          PTOInjectBarrierAllSyncPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    MLIRContext *ctx = func.getContext();
    SmallVector<Operation *> insertionPoints;

    func.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (shouldInjectBarrierAllBefore(op))
        insertionPoints.push_back(op);
      return WalkResult::advance();
    });

    IRRewriter rewriter(ctx);
    auto pipeAll = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_ALL);
    for (Operation *op : insertionPoints) {
      rewriter.setInsertionPoint(op);
      rewriter.create<pto::BarrierOp>(op->getLoc(), pipeAll);
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInjectBarrierAllSyncPass() {
  return std::make_unique<PTOInjectBarrierAllSyncPass>();
}
