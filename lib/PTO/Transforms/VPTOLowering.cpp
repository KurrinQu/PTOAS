//===- VPTOLowering.cpp - Shared VPTO lowering helpers -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/VPTOLowering.h"

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir {
namespace pto {

LogicalResult validateVPTOLoweringChoiceAttr(Operation *op, Attribute attr) {
  if (!attr) {
    op->emitOpError() << "requires '" << kPTOLoweringChoiceAttrName
                      << "' on VPTO candidate ops";
    return failure();
  }

  auto choice = dyn_cast<PTOLoweringChoiceAttr>(attr);
  if (!choice) {
    op->emitOpError() << "expects '" << kPTOLoweringChoiceAttrName
                      << "' to be #pto.lowering_choice<...>, but got "
                      << attr;
    return failure();
  }

  return validateVPTOLoweringChoiceAttr(op, choice);
}

LogicalResult validateVPTOLoweringChoiceAttr(Operation *op,
                                             PTOLoweringChoiceAttr attr) {
  switch (attr.getUpdateMode()) {
  case PTOUpdateMode::PostUpdate:
  case PTOUpdateMode::NoPostUpdate:
    break;
  }

  switch (attr.getLoopShape()) {
  case PTOLoopShape::OneD:
  case PTOLoopShape::TwoD:
    break;
  }

  (void)op;
  return success();
}

FailureOr<PTOLoweringChoiceAttr> getVPTOLoweringChoiceAttr(Operation *op) {
  Attribute attr = op->getAttr(kPTOLoweringChoiceAttrName);
  if (failed(validateVPTOLoweringChoiceAttr(op, attr)))
    return failure();
  return cast<PTOLoweringChoiceAttr>(attr);
}

VPTOLoweringStrategy
convertPTOUpdateModeToVPTOLoweringStrategy(PTOUpdateMode updateMode) {
  switch (updateMode) {
  case PTOUpdateMode::PostUpdate:
    return VPTOLoweringStrategy::PostUpdate;
  case PTOUpdateMode::NoPostUpdate:
    return VPTOLoweringStrategy::NoPostUpdate;
  }
  llvm_unreachable("unsupported VPTO update mode");
}

PTOUpdateMode
convertVPTOLoweringStrategyToPTOUpdateMode(VPTOLoweringStrategy strategy) {
  switch (strategy) {
  case VPTOLoweringStrategy::PostUpdate:
    return PTOUpdateMode::PostUpdate;
  case VPTOLoweringStrategy::NoPostUpdate:
    return PTOUpdateMode::NoPostUpdate;
  }
  llvm_unreachable("unsupported VPTO lowering strategy");
}

bool hasVPTOSameShapeLinearPath(ArrayRef<int64_t> rowStrides,
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

FailureOr<PTOLoopShape>
selectVPTOLoopShapeForFullWidthCols(ArrayRef<int64_t> tileCols,
                                    int64_t validCols) {
  if (validCols == ShapedType::kDynamic)
    return failure();

  for (int64_t tileCol : tileCols) {
    if (tileCol == ShapedType::kDynamic)
      return failure();
    if (tileCol != validCols)
      return PTOLoopShape::TwoD;
  }
  return PTOLoopShape::OneD;
}

FailureOr<PTOLoopShape>
selectVPTOLoopShapeForSameShapeLinearPath(ArrayRef<int64_t> rowStrides,
                                          ArrayRef<int64_t> tileCols,
                                          int64_t validCols) {
  if (!hasVPTOSameShapeLinearPath(rowStrides, tileCols))
    return PTOLoopShape::TwoD;
  return selectVPTOLoopShapeForFullWidthCols(tileCols, validCols);
}

Value buildVPTOFullWidthColsCondition(ArrayRef<int64_t> tileCols,
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
