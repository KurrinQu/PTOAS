//===- A5VMLowering.h - PTO to A5VM lowering contracts ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_
#define MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_

#include "PTO/IR/PTO.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace pto {

enum class A5VMTileDomain {
  Vec,
  Acc,
  Mat,
};

struct A5VMPartitionTrace {
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  bool hasDynamicOffsets = false;
  bool hasDynamicSizes = false;
};

struct A5VMLoadContract {
  StringRef layout;
  SmallVector<int64_t> srcShape;
  SmallVector<int64_t> srcStrides;
  StringRef tileLayout;
  A5VMTileDomain tileDomain = A5VMTileDomain::Vec;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  StringRef padMode;
  bool hasPadValue = false;
  bool leftPaddingPresent = false;
  bool rightPaddingPresent = false;
  bool initOutBuffer = false;
  bool hasInitCondition = false;
  A5VMPartitionTrace trace;
};

struct A5VMUnaryContract {
  StringRef family;
  A5VMTileDomain tileDomain = A5VMTileDomain::Vec;
  StringRef tileLayout;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  Type elementType;
};

struct A5VMStoreContract {
  A5VMTileDomain srcDomain = A5VMTileDomain::Vec;
  StringRef dstLayout;
  SmallVector<int64_t> dstShape;
  SmallVector<int64_t> dstStrides;
  int64_t validRows = ShapedType::kDynamic;
  int64_t validCols = ShapedType::kDynamic;
  A5VMPartitionTrace trace;
};

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter);
LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter);
LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter);

LogicalResult lowerUnaryTileOp(StringRef family,
                               const A5VMUnaryContract &contract, Value src,
                               Value dst, PatternRewriter &rewriter,
                               Location loc);

} // namespace pto
} // namespace mlir

#endif // MLIR_DIALECT_PTO_TRANSFORMS_A5VMLOWERING_H_
