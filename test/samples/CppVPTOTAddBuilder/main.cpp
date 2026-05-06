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
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

static void buildTileBackedTAddBody(OpBuilder &builder, Location loc, Value lhs,
                                    Value rhs, Value out) {
  MLIRContext *ctx = builder.getContext();

  Type f32 = builder.getF32Type();
  auto tv5F32 = pto::TensorViewType::get(ctx, /*rank=*/5, f32);
  auto tileView1x1x1x16x64 = pto::PartitionTensorViewType::get(
      ctx, llvm::ArrayRef<int64_t>{1, 1, 1, 16, 64}, f32);

  auto vecMem = pto::AddressSpaceAttr::get(ctx, pto::AddressSpace::VEC);
  auto ptrF32Ub = pto::PtrType::get(ctx, f32, vecMem);
  auto cfg = pto::TileBufConfigAttr::getDefault(ctx);
  auto tileBuf16x64 = pto::TileBufType::get(
      ctx, llvm::ArrayRef<int64_t>{16, 64}, f32, vecMem,
      llvm::ArrayRef<int64_t>{16, 64}, cfg);

  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value c16 = builder.create<arith::ConstantIndexOp>(loc, 16);
  Value c64 = builder.create<arith::ConstantIndexOp>(loc, 64);
  Value c1024 = builder.create<arith::ConstantIndexOp>(loc, 1024);
  Value c0I64 = builder.create<arith::ConstantIntOp>(loc, 0, 64);
  Value c4096I64 = builder.create<arith::ConstantIntOp>(loc, 4096, 64);
  Value c8192I64 = builder.create<arith::ConstantIntOp>(loc, 8192, 64);

  auto lhsTv = builder.create<pto::MakeTensorViewOp>(
      loc, tv5F32, lhs, ValueRange{c1, c1, c1, c16, c64},
      ValueRange{c1024, c1024, c1024, c64, c1},
      pto::LayoutAttr{});
  auto rhsTv = builder.create<pto::MakeTensorViewOp>(
      loc, tv5F32, rhs, ValueRange{c1, c1, c1, c16, c64},
      ValueRange{c1024, c1024, c1024, c64, c1},
      pto::LayoutAttr{});
  auto outTv = builder.create<pto::MakeTensorViewOp>(
      loc, tv5F32, out, ValueRange{c1, c1, c1, c16, c64},
      ValueRange{c1024, c1024, c1024, c64, c1},
      pto::LayoutAttr{});

  auto lhsPart = builder.create<pto::PartitionViewOp>(
      loc, tileView1x1x1x16x64, lhsTv.getResult(),
      ValueRange{c0, c0, c0, c0, c0}, ValueRange{c1, c1, c1, c16, c64});
  auto rhsPart = builder.create<pto::PartitionViewOp>(
      loc, tileView1x1x1x16x64, rhsTv.getResult(),
      ValueRange{c0, c0, c0, c0, c0}, ValueRange{c1, c1, c1, c16, c64});
  auto outPart = builder.create<pto::PartitionViewOp>(
      loc, tileView1x1x1x16x64, outTv.getResult(),
      ValueRange{c0, c0, c0, c0, c0}, ValueRange{c1, c1, c1, c16, c64});

  auto lhsTile =
      builder.create<pto::AllocTileOp>(loc, tileBuf16x64, c0I64, Value{}, Value{});
  auto rhsTile = builder.create<pto::AllocTileOp>(loc, tileBuf16x64, c4096I64,
                                                  Value{}, Value{});
  auto outTile = builder.create<pto::AllocTileOp>(loc, tileBuf16x64, c8192I64,
                                                  Value{}, Value{});

  builder.create<pto::TLoadOp>(loc, TypeRange{}, lhsPart.getResult(),
                               lhsTile.getResult());
  builder.create<pto::TLoadOp>(loc, TypeRange{}, rhsPart.getResult(),
                               rhsTile.getResult());

  auto event0 = pto::EventAttr::get(ctx, pto::EVENT::EVENT_ID0);
  auto pipeMte2 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE2);
  auto pipeV = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_V);
  auto pipeMte3 = pto::PipeAttr::get(ctx, pto::PIPE::PIPE_MTE3);

  builder.create<pto::SetFlagOp>(loc, pipeMte2, pipeV, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeMte2, pipeV, event0);

  auto vecScope = builder.create<pto::VecScopeOp>(loc);
  vecScope.getBody().push_back(new Block());

  {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&vecScope.getBody().front());

    auto vregTy = pto::VRegType::get(ctx, /*elementCount=*/64, f32);
    auto maskTy = pto::MaskType::get(ctx, "b32");

    auto lhsUb =
        builder.create<pto::TileBufAddrOp>(loc, ptrF32Ub, lhsTile.getResult());
    auto rhsUb =
        builder.create<pto::TileBufAddrOp>(loc, ptrF32Ub, rhsTile.getResult());
    auto outUb =
        builder.create<pto::TileBufAddrOp>(loc, ptrF32Ub, outTile.getResult());

    auto activeMask = builder.create<pto::PsetB32Op>(loc, maskTy, "PAT_ALL");

    auto rowFor = builder.create<scf::ForOp>(loc, c0, c16, c1);
    builder.setInsertionPointToStart(rowFor.getBody());

    Value rowBase =
        builder.create<arith::MulIOp>(loc, rowFor.getInductionVar(), c64);
    auto colFor = builder.create<scf::ForOp>(loc, c0, c64, c64);
    builder.setInsertionPointToStart(colFor.getBody());

    Value offset = builder.create<arith::AddIOp>(loc, rowBase,
                                                 colFor.getInductionVar());
    auto lhsVec = builder.create<pto::VldsOp>(loc, vregTy, lhsUb.getResult(),
                                              offset, StringAttr{});
    auto rhsVec = builder.create<pto::VldsOp>(loc, vregTy, rhsUb.getResult(),
                                              offset, StringAttr{});
    auto sum = builder.create<pto::VaddOp>(loc, vregTy, lhsVec.getResult(),
                                           rhsVec.getResult(),
                                           activeMask.getResult());
    builder.create<pto::VstsOp>(loc, sum.getResult(), outUb.getResult(), offset,
                                StringAttr{}, activeMask.getResult());
  }

  builder.create<pto::SetFlagOp>(loc, pipeV, pipeMte3, event0);
  builder.create<pto::WaitFlagOp>(loc, pipeV, pipeMte3, event0);

  builder.create<pto::TStoreOp>(loc, TypeRange{}, outTile.getResult(),
                                outPart.getResult(), Value{});
}

static ModuleOp buildModule(MLIRContext &ctx) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  ModuleOp module = ModuleOp::create(loc);
  module->setAttr("pto.target_arch", builder.getStringAttr("a5"));

  Type f32 = builder.getF32Type();
  auto ptrF32 = pto::PtrType::get(&ctx, f32);
  auto fnType = builder.getFunctionType({ptrF32, ptrF32, ptrF32}, {});

  builder.setInsertionPointToStart(module.getBody());
  auto fn = builder.create<func::FuncOp>(loc, "tadd_tile_lowered_to_vecscope",
                                         fnType);

  Block *entry = fn.addEntryBlock();
  builder.setInsertionPointToStart(entry);
  buildTileBackedTAddBody(builder, loc, entry->getArgument(0),
                          entry->getArgument(1), entry->getArgument(2));
  builder.create<func::ReturnOp>(loc);

  return module;
}

int main() {
  DialectRegistry registry;
  registry.insert<arith::ArithDialect, func::FuncDialect, scf::SCFDialect,
                  pto::PTODialect>();

  MLIRContext ctx(registry);
  ctx.getOrLoadDialect<arith::ArithDialect>();
  ctx.getOrLoadDialect<func::FuncDialect>();
  ctx.getOrLoadDialect<scf::SCFDialect>();
  ctx.getOrLoadDialect<pto::PTODialect>();

  ModuleOp module = buildModule(ctx);
  if (failed(verify(module))) {
    llvm::errs() << "module verification failed\n";
    return 1;
  }

  module.print(llvm::outs());
  llvm::outs() << "\n";
  return 0;
}
