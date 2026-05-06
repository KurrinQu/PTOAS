// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "PTO/IR/PTO.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static ModuleOp buildVecAddKernel2D(MLIRContext &ctx) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  ModuleOp module = ModuleOp::create(loc);

  Type f32 = builder.getF32Type();
  auto ptrF32 = pto::PtrType::get(&ctx, f32);
  auto tv2F32 = pto::TensorViewType::get(&ctx, /*rank=*/2, f32);
  auto tileView32 =
      pto::PartitionTensorViewType::get(&ctx, llvm::ArrayRef<int64_t>{32, 32},
                                        f32);

  auto vec = pto::AddressSpaceAttr::get(&ctx, pto::AddressSpace::VEC);
  auto cfg = pto::TileBufConfigAttr::getDefault(&ctx);
  auto tileBuf32 = pto::TileBufType::get(
      &ctx, llvm::ArrayRef<int64_t>{32, 32}, f32, vec,
      llvm::ArrayRef<int64_t>{32, 32}, cfg);

  auto fnType = builder.getFunctionType({ptrF32, ptrF32, ptrF32}, {});
  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(loc, "vec_add_kernel_2d", fnType);

  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c32 = builder.create<arith::ConstantIndexOp>(loc, 32);

  Value arg0 = entry->getArgument(0);
  Value arg1 = entry->getArgument(1);
  Value arg2 = entry->getArgument(2);

  auto tv0 = builder.create<pto::MakeTensorViewOp>(
      loc, tv2F32, arg0, ValueRange{c32, c32}, ValueRange{c32, c1},
      pto::LayoutAttr{});
  auto tv1 = builder.create<pto::MakeTensorViewOp>(
      loc, tv2F32, arg1, ValueRange{c32, c32}, ValueRange{c32, c1},
      pto::LayoutAttr{});
  auto tv2 = builder.create<pto::MakeTensorViewOp>(
      loc, tv2F32, arg2, ValueRange{c32, c32}, ValueRange{c32, c1},
      pto::LayoutAttr{});

  auto sv0 = builder.create<pto::PartitionViewOp>(
      loc, tileView32, tv0.getResult(), ValueRange{c0, c0},
      ValueRange{c32, c32});
  auto sv1 = builder.create<pto::PartitionViewOp>(
      loc, tileView32, tv1.getResult(), ValueRange{c0, c0},
      ValueRange{c32, c32});
  auto sv2 = builder.create<pto::PartitionViewOp>(
      loc, tileView32, tv2.getResult(), ValueRange{c0, c0},
      ValueRange{c32, c32});

  auto tb0 =
      builder.create<pto::AllocTileOp>(loc, tileBuf32, Value{}, Value{}, Value{});
  auto tb1 =
      builder.create<pto::AllocTileOp>(loc, tileBuf32, Value{}, Value{}, Value{});
  auto tb2 =
      builder.create<pto::AllocTileOp>(loc, tileBuf32, Value{}, Value{}, Value{});

  builder.create<pto::TLoadOp>(loc, TypeRange{}, sv0.getResult(), tb0.getResult());
  builder.create<pto::TLoadOp>(loc, TypeRange{}, sv1.getResult(), tb1.getResult());
  builder.create<pto::TAddOp>(loc, tb0.getResult(), tb1.getResult(), tb2.getResult());
  builder.create<pto::TStoreOp>(loc, TypeRange{}, tb2.getResult(),
                                sv2.getResult(), Value{});

  builder.create<func::ReturnOp>(loc);
  return module;
}

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, pto::PTODialect>();

  MLIRContext ctx(registry);
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<pto::PTODialect>();

  ModuleOp module = buildVecAddKernel2D(ctx);
  if (failed(verify(module))) {
    llvm::errs() << "module verification failed\n";
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
