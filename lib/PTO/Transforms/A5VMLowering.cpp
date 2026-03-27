//===- A5VMLowering.cpp - Shared A5VM lowering helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace pto {

LogicalResult validateA5VMLoweringChoiceAttr(Operation *op, Attribute attr) {
  if (!attr) {
    op->emitOpError() << "requires '" << kA5VMLoweringChoiceAttrName
                      << "' on A5VM candidate ops";
    return failure();
  }

  auto choice = dyn_cast<A5VMLoweringChoiceAttr>(attr);
  if (!choice) {
    op->emitOpError() << "expects '" << kA5VMLoweringChoiceAttrName
                      << "' to be #pto.a5vm_lowering_choice<...>, but got "
                      << attr;
    return failure();
  }

  return validateA5VMLoweringChoiceAttr(op, choice);
}

LogicalResult validateA5VMLoweringChoiceAttr(Operation *op,
                                             A5VMLoweringChoiceAttr attr) {
  switch (attr.getUpdateMode()) {
  case A5VMUpdateMode::PostUpdate:
  case A5VMUpdateMode::NoPostUpdate:
    break;
  }

  switch (attr.getLoopShape()) {
  case A5VMLoopShape::OneD:
  case A5VMLoopShape::TwoD:
    break;
  }

  (void)op;
  return success();
}

FailureOr<A5VMLoweringChoiceAttr> getA5VMLoweringChoiceAttr(Operation *op) {
  Attribute attr = op->getAttr(kA5VMLoweringChoiceAttrName);
  if (failed(validateA5VMLoweringChoiceAttr(op, attr)))
    return failure();
  return cast<A5VMLoweringChoiceAttr>(attr);
}

A5VMLoweringStrategy
convertA5VMUpdateModeToLoweringStrategy(A5VMUpdateMode updateMode) {
  switch (updateMode) {
  case A5VMUpdateMode::PostUpdate:
    return A5VMLoweringStrategy::PostUpdate;
  case A5VMUpdateMode::NoPostUpdate:
    return A5VMLoweringStrategy::NoPostUpdate;
  }
  llvm_unreachable("unsupported A5VM update mode");
}

A5VMUpdateMode
convertA5VMLoweringStrategyToUpdateMode(A5VMLoweringStrategy strategy) {
  switch (strategy) {
  case A5VMLoweringStrategy::PostUpdate:
    return A5VMUpdateMode::PostUpdate;
  case A5VMLoweringStrategy::NoPostUpdate:
    return A5VMUpdateMode::NoPostUpdate;
  }
  llvm_unreachable("unsupported A5VM lowering strategy");
}

bool hasA5VMSameShapeLinearPath(ArrayRef<int64_t> rowStrides,
                                ArrayRef<int64_t> tileCols) {
  if (rowStrides.empty() || rowStrides.size() != tileCols.size())
    return false;

  int64_t referenceStride = rowStrides.back();
  int64_t referenceCols = tileCols.back();
  if (referenceStride == ShapedType::kDynamic ||
      referenceCols == ShapedType::kDynamic)
    return false;

  for (auto [rowStride, tileCol] : llvm::zip_equal(rowStrides, tileCols)) {
    if (rowStride == ShapedType::kDynamic || tileCol == ShapedType::kDynamic)
      return false;
    if (rowStride != referenceStride || tileCol != referenceCols)
      return false;
  }
  return true;
}

FailureOr<A5VMLoopShape>
selectA5VMLoopShapeForFullWidthCols(ArrayRef<int64_t> tileCols,
                                    int64_t validCols) {
  if (validCols == ShapedType::kDynamic)
    return failure();

  for (int64_t tileCol : tileCols) {
    if (tileCol == ShapedType::kDynamic)
      return failure();
    if (tileCol != validCols)
      return A5VMLoopShape::TwoD;
  }
  return A5VMLoopShape::OneD;
}

FailureOr<A5VMLoopShape>
selectA5VMLoopShapeForSameShapeLinearPath(ArrayRef<int64_t> rowStrides,
                                          ArrayRef<int64_t> tileCols,
                                          int64_t validCols) {
  if (!hasA5VMSameShapeLinearPath(rowStrides, tileCols))
    return A5VMLoopShape::TwoD;
  return selectA5VMLoopShapeForFullWidthCols(tileCols, validCols);
}

Value buildA5VMFullWidthColsCondition(ArrayRef<int64_t> tileCols,
                                      Value validColsValue,
                                      PatternRewriter &rewriter, Location loc) {
  Value condition;
  for (int64_t tileCol : tileCols) {
    if (tileCol == ShapedType::kDynamic)
      return {};
    Value tileColValue = rewriter.create<arith::ConstantIndexOp>(loc, tileCol);
    Value isFullWidth = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, validColsValue, tileColValue);
    condition = condition ? rewriter.create<arith::AndIOp>(loc, condition, isFullWidth)
                          : isFullWidth;
  }
  return condition;
}

} // namespace pto
} // namespace mlir
