//===- PTOToA5VMLowering.cpp - PTO to A5VM lowering helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/Support/ErrorHandling.h"

#include <optional>

namespace mlir {
namespace pto {

namespace {

constexpr StringLiteral kVecScopeName = "__VEC_SCOPE__";
constexpr StringLiteral kSourceLoopScopeAttrName = "cce_aiv_loop_hint";
constexpr StringLiteral kLoweredLoopScopeAttrName = "llvm.loop.aivector_scope";

std::optional<int64_t> getConstInt(Value value) {
  if (!value)
    return std::nullopt;

  if (auto constIndex = value.getDefiningOp<arith::ConstantIndexOp>())
    return constIndex.value();
  if (auto constInt = value.getDefiningOp<arith::ConstantIntOp>())
    return constInt.value();
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      return intAttr.getInt();
  }
  return std::nullopt;
}

StringRef stringifyTileDomain(A5VMTileDomain domain) {
  switch (domain) {
  case A5VMTileDomain::Vec:
    return "vec";
  case A5VMTileDomain::Acc:
    return "acc";
  case A5VMTileDomain::Mat:
    return "mat";
  }
  llvm_unreachable("unknown A5VM tile domain");
}

StringRef stringifyTileLayout(TileBufType type) {
  if (auto layoutAttr = dyn_cast_or_null<BLayoutAttr>(type.getBLayoutAttr())) {
    switch (layoutAttr.getValue()) {
    case BLayout::RowMajor:
      return "row_major";
    case BLayout::ColMajor:
      return "col_major";
    }
  }
  return "row_major";
}

StringRef stringifyTileLayoutConfig(TileBufConfigAttr config) {
  if (!config)
    return "row_major";
  if (auto layoutAttr = dyn_cast_or_null<BLayoutAttr>(config.getBLayout())) {
    switch (layoutAttr.getValue()) {
    case BLayout::RowMajor:
      return "row_major";
    case BLayout::ColMajor:
      return "col_major";
    }
  }
  return "row_major";
}

StringRef stringifyPadModeAttr(PadModeAttr padMode) {
  if (!padMode)
    return "none";

  switch (padMode.getPadmode()) {
  case PadMode::PadNull:
    return "none";
  case PadMode::PadFirstElem:
    return "first_elem";
  case PadMode::PadValue:
    return "value";
  }
  return "none";
}

StringRef stringifyLayoutAttr(Attribute layoutAttr) {
  if (auto attr = dyn_cast_or_null<LayoutAttr>(layoutAttr))
    return stringifyLayout(attr.getLayout());
  return "nd";
}

A5VMTileDomain deriveTileDomain(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case AddressSpace::ACC:
      return A5VMTileDomain::Acc;
    case AddressSpace::MAT:
      return A5VMTileDomain::Mat;
    case AddressSpace::VEC:
    default:
      return A5VMTileDomain::Vec;
    }
  }
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(AddressSpace::ACC):
      return A5VMTileDomain::Acc;
    case static_cast<int64_t>(AddressSpace::MAT):
      return A5VMTileDomain::Mat;
    default:
      return A5VMTileDomain::Vec;
    }
  }
  return A5VMTileDomain::Vec;
}

void getValidShape(TileBufType type, int64_t &rows, int64_t &cols) {
  ArrayRef<int64_t> validShape = type.getValidShape();
  rows = validShape.size() > 0 ? validShape[0] : ShapedType::kDynamic;
  cols = validShape.size() > 1 ? validShape[1] : ShapedType::kDynamic;
}

TileBufConfigAttr lookupTileConfig(Value value) {
  if (!value)
    return {};
  if (auto bind = value.getDefiningOp<BindTileOp>())
    return bind.getConfig();
  if (auto cast = value.getDefiningOp<PointerCastOp>())
    return cast.getConfig().value_or(TileBufConfigAttr{});
  if (auto subview = value.getDefiningOp<memref::SubViewOp>())
    return lookupTileConfig(subview.getSource());
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return lookupTileConfig(reinterpret.getSource());
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return lookupTileConfig(cast.getSource());
  return {};
}

void lookupValidDims(Value value, Value &validRow, Value &validCol) {
  if (!value) {
    validRow = {};
    validCol = {};
    return;
  }
  if (auto bind = value.getDefiningOp<BindTileOp>()) {
    validRow = bind.getValidRow();
    validCol = bind.getValidCol();
    return;
  }
  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    validRow = cast.getValidRow();
    validCol = cast.getValidCol();
    return;
  }
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    lookupValidDims(subview.getSource(), validRow, validCol);
    return;
  }
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    lookupValidDims(reinterpret.getSource(), validRow, validCol);
    return;
  }
  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    lookupValidDims(cast.getSource(), validRow, validCol);
    return;
  }
  validRow = {};
  validCol = {};
}

Type getElementType(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<TileBufType>(type))
    return tileType.getElementType();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getElementType();
  if (auto ptrType = dyn_cast<PtrType>(type))
    return ptrType.getElementType();
  return {};
}

Attribute getMemorySpace(Value value) {
  Type type = value.getType();
  if (auto tileType = dyn_cast<TileBufType>(type))
    return tileType.getMemorySpace();
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return memrefType.getMemorySpace();
  return {};
}

StringRef deriveTileLayout(Value value) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType()))
    return stringifyTileLayout(tileType);
  return stringifyTileLayoutConfig(lookupTileConfig(value));
}

void deriveValidShape(Value value, int64_t &rows, int64_t &cols) {
  if (auto tileType = dyn_cast<TileBufType>(value.getType())) {
    getValidShape(tileType, rows, cols);
    return;
  }

  Value validRow;
  Value validCol;
  lookupValidDims(value, validRow, validCol);
  rows = getConstInt(validRow).value_or(ShapedType::kDynamic);
  cols = getConstInt(validCol).value_or(ShapedType::kDynamic);
}

void appendStaticSizes(ValueRange values, SmallVectorImpl<int64_t> &out,
                       bool &hasDynamic) {
  out.clear();
  hasDynamic = false;
  out.reserve(values.size());
  for (Value value : values) {
    if (std::optional<int64_t> constant = getConstInt(value)) {
      out.push_back(*constant);
      continue;
    }
    out.push_back(ShapedType::kDynamic);
    hasDynamic = true;
  }
}

int64_t defaultTransferExtent(int64_t value) {
  return value == ShapedType::kDynamic ? 1 : value;
}

int64_t deriveTransferStride(ArrayRef<int64_t> strides, int64_t fallback,
                             size_t index = 0) {
  if (index < strides.size() && strides[index] != ShapedType::kDynamic)
    return strides[index];
  for (int64_t stride : strides) {
    if (stride != ShapedType::kDynamic)
      return stride;
  }
  return fallback;
}

Value resolveTensorViewBase(Value value, Attribute &layoutAttr,
                            SmallVectorImpl<int64_t> &shape,
                            SmallVectorImpl<int64_t> &strides) {
  if (!value)
    return {};

  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    if (auto source = part.getSource().getDefiningOp<MakeTensorViewOp>()) {
      layoutAttr = source.getLayoutAttr();
      auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
      shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
      strides.clear();
      for (Value stride : source.getStrides()) {
        if (std::optional<int64_t> constant = getConstInt(stride))
          strides.push_back(*constant);
        else
          strides.push_back(ShapedType::kDynamic);
      }
      return source.getPtr();
    }
    return resolveTensorViewBase(part.getSource(), layoutAttr, shape, strides);
  }

  if (auto source = value.getDefiningOp<MakeTensorViewOp>()) {
    layoutAttr = source.getLayoutAttr();
    auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
    shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
    strides.clear();
    for (Value stride : source.getStrides()) {
      if (std::optional<int64_t> constant = getConstInt(stride))
        strides.push_back(*constant);
      else
        strides.push_back(ShapedType::kDynamic);
    }
    return source.getPtr();
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    int64_t offset = 0;
    if (failed(mlir::getStridesAndOffset(memrefType, strides, offset)))
      strides.assign(shape.size(), ShapedType::kDynamic);
    return value;
  }

  return {};
}

a5vm::VecType getA5VMVecType(MLIRContext *context, Type elementType) {
  unsigned bitWidth = 0;
  if (auto floatType = dyn_cast<FloatType>(elementType))
    bitWidth = floatType.getWidth();
  else if (auto intType = dyn_cast<IntegerType>(elementType))
    bitWidth = intType.getWidth();

  if (bitWidth == 0 || 2048 % bitWidth != 0)
    return {};
  return a5vm::VecType::get(context, 2048 / bitWidth, elementType);
}

ArrayAttr asI64ArrayAttr(Builder &builder, ArrayRef<int64_t> values) {
  SmallVector<Attribute> attrs;
  attrs.reserve(values.size());
  for (int64_t value : values)
    attrs.push_back(builder.getI64IntegerAttr(value));
  return builder.getArrayAttr(attrs);
}

DictionaryAttr makeStrideAttr(Builder &builder, int64_t first, int64_t second,
                              StringRef firstName, StringRef secondName) {
  return builder.getDictionaryAttr(
      {builder.getNamedAttr(firstName, builder.getI64IntegerAttr(first)),
       builder.getNamedAttr(secondName, builder.getI64IntegerAttr(second))});
}

void attachTraceAttrs(Operation *op, const A5VMPartitionTrace &trace,
                      Builder &builder) {
  op->setAttr("trace_offsets", asI64ArrayAttr(builder, trace.offsets));
  op->setAttr("trace_sizes", asI64ArrayAttr(builder, trace.sizes));
}

int64_t deriveTileLoop2Stride(StringRef tileLayout, int64_t validRows,
                              int64_t validCols) {
  if (tileLayout == "col_major")
    return defaultTransferExtent(validRows);
  return defaultTransferExtent(validCols);
}

int64_t deriveTileLoop1Stride(StringRef tileLayout, int64_t validRows,
                              int64_t validCols) {
  if (tileLayout == "col_major")
    return 1;
  return 1;
}

Attribute getGmMemorySpace(MLIRContext *context) {
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

Value materializeMemRefView(Value value, ArrayRef<int64_t> shape, Type elementType,
                            Attribute memorySpace,
                            PatternRewriter &rewriter, Location loc) {
  auto memrefType =
      MemRefType::get(shape, elementType, AffineMap(), memorySpace);
  if (value.getType() == memrefType)
    return value;
  return rewriter
      .create<UnrealizedConversionCastOp>(
          loc, TypeRange(ArrayRef<Type>{memrefType}), value)
      .getResult(0);
}

Value materializeTileBufferView(Value value, PatternRewriter &rewriter,
                                Location loc) {
  if (auto memrefType = dyn_cast<BaseMemRefType>(value.getType()))
    return value;

  auto tileType = dyn_cast<TileBufType>(value.getType());
  if (!tileType)
    return {};

  return materializeMemRefView(value, tileType.getShape(), tileType.getElementType(),
                               tileType.getMemorySpace(), rewriter, loc);
}

Value materializeGmView(Value value, ArrayRef<int64_t> shape, Type elementType,
                        PatternRewriter &rewriter, Location loc) {
  if (isa<BaseMemRefType>(value.getType()))
    return value;
  return materializeMemRefView(value, shape, elementType,
                               getGmMemorySpace(rewriter.getContext()),
                               rewriter, loc);
}

A5VMPartitionTrace extractPartitionTrace(Value value) {
  A5VMPartitionTrace trace;
  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    appendStaticSizes(part.getOffsets(), trace.offsets, trace.hasDynamicOffsets);
    appendStaticSizes(part.getSizes(), trace.sizes, trace.hasDynamicSizes);
  }
  return trace;
}

A5VMLoadContract extractTLoadContract(TLoadOp op) {
  A5VMLoadContract contract;
  contract.trace = extractPartitionTrace(op.getSrc());

  Attribute layoutAttr;
  Value base = resolveTensorViewBase(op.getSrc(), layoutAttr, contract.sourceShape,
                                     contract.sourceStrides);
  (void)base;
  contract.sourceLayout = stringifyLayoutAttr(layoutAttr);

  contract.tileLayout = deriveTileLayout(op.getDst());
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getDst()));
  deriveValidShape(op.getDst(), contract.validRows, contract.validCols);
  contract.padMode = stringifyPadModeAttr(op.getPadModeAttr());
  contract.padValue = op.getPadValue();
  contract.leftPaddingNum = op.getLeftPaddingNum();
  contract.rightPaddingNum = op.getRightPaddingNum();
  contract.initOutBuffer = op.getInitOutBuffer();
  contract.initCondition = op.getInitCondition();
  return contract;
}

A5VMUnaryContract extractTAbsContract(TAbsOp op) {
  A5VMUnaryContract contract;
  contract.family = "abs";
  contract.tileDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  contract.tileLayout = deriveTileLayout(op.getSrc());
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getSrc());
  contract.loopScope.kind = A5VMLoopScopeKind::AIVVectorScope;
  contract.loopScope.sourceAttr = kSourceLoopScopeAttrName;
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

A5VMStoreContract extractTStoreContract(TStoreOp op) {
  A5VMStoreContract contract;
  contract.trace = extractPartitionTrace(op.getDst());

  contract.srcDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);

  Attribute layoutAttr;
  Value base = resolveTensorViewBase(op.getDst(), layoutAttr,
                                     contract.destinationShape,
                                     contract.destinationStrides);
  (void)base;
  contract.destinationLayout = stringifyLayoutAttr(layoutAttr);
  return contract;
}

void attachLoadContractAttrs(Operation *op, const A5VMLoadContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("layout", builder.getStringAttr(contract.sourceLayout));
  op->setAttr("src_shape", asI64ArrayAttr(builder, contract.sourceShape));
  op->setAttr("src_strides", asI64ArrayAttr(builder, contract.sourceStrides));
  op->setAttr("tile_layout", builder.getStringAttr(contract.tileLayout));
  op->setAttr("tile_domain",
              builder.getStringAttr(stringifyTileDomain(contract.tileDomain)));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  op->setAttr("pad_mode", builder.getStringAttr(contract.padMode));
  op->setAttr("has_pad_value", builder.getBoolAttr(static_cast<bool>(contract.padValue)));
  op->setAttr("left_padding_num", builder.getI64IntegerAttr(
                                      getConstInt(contract.leftPaddingNum).value_or(0)));
  op->setAttr("right_padding_num", builder.getI64IntegerAttr(
                                       getConstInt(contract.rightPaddingNum).value_or(0)));
  op->setAttr("init_out_buffer", builder.getBoolAttr(contract.initOutBuffer));
  op->setAttr("has_init_condition",
              builder.getBoolAttr(static_cast<bool>(contract.initCondition)));
  attachTraceAttrs(op, contract.trace, builder);
}

void attachUnaryContractAttrs(Operation *op, const A5VMUnaryContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("unary_family", builder.getStringAttr(contract.family));
  op->setAttr("tile_domain",
              builder.getStringAttr(stringifyTileDomain(contract.tileDomain)));
  op->setAttr("tile_layout", builder.getStringAttr(contract.tileLayout));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  if (contract.loopScope.kind == A5VMLoopScopeKind::AIVVectorScope) {
    op->setAttr(contract.loopScope.sourceAttr, builder.getUnitAttr());
    op->setAttr(contract.loopScope.loweredAttr, builder.getUnitAttr());
    op->setAttr("a5vm.scope", builder.getStringAttr(kVecScopeName));
  }
}

void attachStoreContractAttrs(Operation *op, const A5VMStoreContract &contract) {
  Builder builder(op->getContext());
  op->setAttr("src_domain",
              builder.getStringAttr(stringifyTileDomain(contract.srcDomain)));
  op->setAttr("dst_layout", builder.getStringAttr(contract.destinationLayout));
  op->setAttr("dst_shape", asI64ArrayAttr(builder, contract.destinationShape));
  op->setAttr("dst_strides", asI64ArrayAttr(builder, contract.destinationStrides));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  attachTraceAttrs(op, contract.trace, builder);
}

LogicalResult lowerUnsupportedAccStore(Location loc) {
  emitError(loc) << "TSTORE ACC lowering TODO for a5vm backend";
  return failure();
}

LogicalResult lowerUnsupportedMatStore(Location loc) {
  emitError(loc) << "TSTORE MAT lowering TODO for a5vm backend";
  return failure();
}

} // namespace

LogicalResult attachLoopScopeMetadata(LoopLikeOpInterface loop,
                                      const A5VMLoopScopeContract &contract,
                                      PatternRewriter &rewriter) {
  if (!loop)
    return failure();
  if (contract.kind == A5VMLoopScopeKind::None)
    return success();
  if (contract.kind != A5VMLoopScopeKind::AIVVectorScope)
    return failure();

  Operation *loopOp = loop.getOperation();
  loopOp->setAttr(contract.sourceAttr, rewriter.getUnitAttr());
  loopOp->setAttr(contract.loweredAttr, rewriter.getUnitAttr());
  loopOp->setAttr("a5vm.scope", rewriter.getStringAttr(kVecScopeName));
  loopOp->setAttr("a5vm.loop_scope_depth",
                  rewriter.getI64IntegerAttr(contract.loopDepth));
  return success();
}

void set_loop2_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_outtoub",
                  makeStrideAttr(builder, dstStride, srcStride, "dst_stride",
                                 "src_stride"));
}

void set_loop1_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_outtoub",
                  makeStrideAttr(builder, dstStride, srcStride, "dst_stride",
                                 "src_stride"));
}

void set_loop_size_outtoub(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_outtoub",
                  builder.getDictionaryAttr(
                      {builder.getNamedAttr("loop2",
                                            builder.getI64IntegerAttr(loop2)),
                       builder.getNamedAttr("loop1",
                                            builder.getI64IntegerAttr(loop1))}));
}

void set_loop2_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_ubtoout",
                  makeStrideAttr(builder, srcStride, dstStride, "src_stride",
                                 "dst_stride"));
}

void set_loop1_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_ubtoout",
                  makeStrideAttr(builder, srcStride, dstStride, "src_stride",
                                 "dst_stride"));
}

void set_loop_size_ubtoout(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_ubtoout",
                  builder.getDictionaryAttr(
                      {builder.getNamedAttr("loop2",
                                            builder.getI64IntegerAttr(loop2)),
                       builder.getNamedAttr("loop1",
                                            builder.getI64IntegerAttr(loop1))}));
}

LogicalResult programCopyGmToUbLoops(Operation *copyOp,
                                     const A5VMLoadContract &contract,
                                     Builder &builder) {
  int64_t loop2 = defaultTransferExtent(contract.validRows);
  int64_t loop1 = defaultTransferExtent(contract.validCols);
  int64_t dstLoop2Stride =
      deriveTileLoop2Stride(contract.tileLayout, contract.validRows,
                            contract.validCols);
  int64_t dstLoop1Stride =
      deriveTileLoop1Stride(contract.tileLayout, contract.validRows,
                            contract.validCols);
  int64_t srcLoop2Stride =
      deriveTransferStride(contract.sourceStrides, loop1, 0);
  int64_t srcLoop1Stride =
      deriveTransferStride(contract.sourceStrides, 1, 1);

  set_loop2_stride_outtoub(copyOp, dstLoop2Stride, srcLoop2Stride, builder);
  set_loop1_stride_outtoub(copyOp, dstLoop1Stride, srcLoop1Stride, builder);
  set_loop_size_outtoub(copyOp, loop2, loop1, builder);
  return success();
}

LogicalResult programCopyUbToGmLoops(Operation *copyOp,
                                     const A5VMStoreContract &contract,
                                     Builder &builder) {
  int64_t loop2 = defaultTransferExtent(contract.validRows);
  int64_t loop1 = defaultTransferExtent(contract.validCols);
  int64_t srcLoop2Stride = deriveTileLoop2Stride("row_major", contract.validRows,
                                                 contract.validCols);
  int64_t srcLoop1Stride = deriveTileLoop1Stride("row_major", contract.validRows,
                                                 contract.validCols);
  int64_t dstLoop2Stride =
      deriveTransferStride(contract.destinationStrides, loop1, 0);
  int64_t dstLoop1Stride =
      deriveTransferStride(contract.destinationStrides, 1, 1);

  set_loop_size_ubtoout(copyOp, loop2, loop1, builder);
  set_loop1_stride_ubtoout(copyOp, srcLoop1Stride, dstLoop1Stride, builder);
  set_loop2_stride_ubtoout(copyOp, srcLoop2Stride, dstLoop2Stride, builder);
  return success();
}

LogicalResult buildUnaryVecScope(StringRef family,
                                 const A5VMUnaryContract &contract, Value src,
                                 Value dst, PatternRewriter &rewriter,
                                 Location loc) {
  auto vecType = getA5VMVecType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported A5VM unary element type";

  Value srcBuffer = materializeTileBufferView(src, rewriter, loc);
  Value dstBuffer = materializeTileBufferView(dst, rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires memref-backed tile buffers for unary lowering";

  int64_t validRows = contract.validRows;
  int64_t validCols = contract.validCols;
  int64_t vectorWidth = vecType.getElementCount();
  if (validRows == ShapedType::kDynamic || validCols == ShapedType::kDynamic)
    return emitError(loc) << "TABS lowering requires static valid rows and cols";
  int64_t totalElements = validRows * validCols;
  if (totalElements % vectorWidth != 0)
    return emitError(loc)
           << "TABS lowering requires total valid elements divisible by vector width";

  int64_t outerStep = 1;
  int64_t innerStep = vectorWidth;
  int64_t innerUpperBound = validCols;
  if (validCols < vectorWidth) {
    if (vectorWidth % validCols != 0)
      return emitError(loc)
             << "TABS lowering requires valid cols to evenly divide the vector width";
    outerStep = vectorWidth / validCols;
    innerStep = validCols;
  }

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value rows = rewriter.create<arith::ConstantIndexOp>(loc, validRows);
  Value cols = rewriter.create<arith::ConstantIndexOp>(loc, validCols);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value outerStepValue = rewriter.create<arith::ConstantIndexOp>(loc, outerStep);
  Value innerUpperBoundValue =
      rewriter.create<arith::ConstantIndexOp>(loc, innerUpperBound);
  Value innerStepValue = rewriter.create<arith::ConstantIndexOp>(loc, innerStep);

  auto aivScopeLoop = rewriter.create<scf::ForOp>(loc, c0, c1, c1);
  if (failed(attachLoopScopeMetadata(aivScopeLoop, contract.loopScope, rewriter)))
    return emitError(loc) << "failed to attach AIV loop scope metadata";
  attachUnaryContractAttrs(aivScopeLoop, contract);

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(aivScopeLoop.getBody());
  auto chunkLoop = rewriter.create<scf::ForOp>(loc, c0, rows, outerStepValue);
  attachUnaryContractAttrs(chunkLoop, contract);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  auto vectorLoop =
      rewriter.create<scf::ForOp>(loc, c0, innerUpperBoundValue, innerStepValue);

  OpBuilder::InsertionGuard vectorGuard(rewriter);
  rewriter.setInsertionPointToStart(vectorLoop.getBody());
  Value rowOffset = rewriter.create<arith::MulIOp>(loc, chunkLoop.getInductionVar(),
                                                   cols);
  Value offset =
      rewriter.create<arith::AddIOp>(loc, rowOffset, vectorLoop.getInductionVar());
  auto vlds = rewriter.create<a5vm::VldsOp>(loc, vecType, srcBuffer, offset);
  auto vabs = rewriter.create<a5vm::VabsOp>(loc, vecType, vlds.getResult());
  attachUnaryContractAttrs(vabs, contract);
  (void)family;
  rewriter.create<a5vm::VstsOp>(loc, vabs.getResult(), dstBuffer, offset,
                                rewriter.getStringAttr(kVecScopeName));

  return success();
}

LogicalResult lowerTLOAD(TLoadOp op, PatternRewriter &rewriter) {
  A5VMLoadContract contract = extractTLoadContract(op);

  Attribute layoutAttr;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  Value base = resolveTensorViewBase(op.getSrc(), layoutAttr, shape, strides);
  if (!base)
    return op.emitOpError("requires a memref- or ptr-backed source for A5VM lowering");

  Value sourceBuffer = materializeGmView(base, contract.sourceShape,
                                         getElementType(base), rewriter,
                                         op.getLoc());
  Value destinationBuffer = materializeTileBufferView(op.getDst(), rewriter,
                                                      op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  int64_t burstCount = defaultTransferExtent(contract.validRows);
  int64_t burstLen = defaultTransferExtent(contract.validCols);
  int64_t gmStride =
      deriveTransferStride(contract.sourceStrides, burstLen, 0);
  int64_t ubStride =
      deriveTileLoop2Stride(contract.tileLayout, contract.validRows,
                            contract.validCols);
  bool ubPad = contract.padMode != "none" || contract.padValue ||
               contract.leftPaddingNum || contract.rightPaddingNum;

  auto copyOp = rewriter.create<a5vm::CopyGmToUbufOp>(
      op.getLoc(), sourceBuffer, destinationBuffer,
      rewriter.getStringAttr(contract.sourceLayout),
      rewriter.getI64IntegerAttr(contract.validRows),
      rewriter.getI64IntegerAttr(contract.validCols),
      rewriter.getI64IntegerAttr(burstCount),
      rewriter.getI64IntegerAttr(burstLen),
      rewriter.getI64IntegerAttr(gmStride),
      rewriter.getI64IntegerAttr(ubStride),
      rewriter.getBoolAttr(ubPad));
  attachLoadContractAttrs(copyOp, contract);
  if (failed(programCopyGmToUbLoops(copyOp, contract, rewriter)))
    return failure();
  return success();
}

LogicalResult lowerTABS(TAbsOp op, PatternRewriter &rewriter) {
  A5VMUnaryContract contract = extractTAbsContract(op);
  int64_t dstRows = ShapedType::kDynamic;
  int64_t dstCols = ShapedType::kDynamic;
  deriveValidShape(op.getDst(), dstRows, dstCols);
  StringRef dstLayout = deriveTileLayout(op.getDst());
  A5VMTileDomain dstDomain = deriveTileDomain(getMemorySpace(op.getDst()));

  bool hasPrecheckFailure = false;
  if (contract.tileDomain != A5VMTileDomain::Vec || dstDomain != A5VMTileDomain::Vec) {
    op.emitOpError("TABS lowering requires tile domain vec");
    hasPrecheckFailure = true;
  }
  if (contract.tileLayout != "row_major" || dstLayout != "row_major") {
    op.emitOpError("TABS lowering requires row-major tile layout");
    hasPrecheckFailure = true;
  }
  if (contract.validRows != dstRows || contract.validCols != dstCols) {
    op.emitOpError(
        "TABS lowering requires matching source and destination valid region");
    hasPrecheckFailure = true;
  }
  if (!contract.elementType ||
      (!contract.elementType.isF16() && !contract.elementType.isF32())) {
    op.emitOpError("TABS lowering supports only f16 and f32 element types");
    hasPrecheckFailure = true;
  }
  if (hasPrecheckFailure)
    return failure();

  return buildUnaryVecScope("abs", contract, op.getSrc(), op.getDst(), rewriter,
                            op.getLoc());
}

LogicalResult lowerTSTORE(TStoreOp op, PatternRewriter &rewriter) {
  A5VMStoreContract contract = extractTStoreContract(op);

  switch (contract.srcDomain) {
  case A5VMTileDomain::Acc:
    return lowerUnsupportedAccStore(op.getLoc());
  case A5VMTileDomain::Mat:
    return lowerUnsupportedMatStore(op.getLoc());
  case A5VMTileDomain::Vec:
    break;
  }

  Attribute layoutAttr;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> strides;
  Value base = resolveTensorViewBase(op.getDst(), layoutAttr, shape, strides);
  if (!base)
    return op.emitOpError("requires a memref- or ptr-backed destination for A5VM lowering");

  Value sourceBuffer = materializeTileBufferView(op.getSrc(), rewriter, op.getLoc());
  Value destinationBuffer = materializeGmView(base, contract.destinationShape,
                                              getElementType(base), rewriter,
                                              op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  int64_t burstCount = defaultTransferExtent(contract.validRows);
  int64_t burstLen = defaultTransferExtent(contract.validCols);
  int64_t gmStride =
      deriveTransferStride(contract.destinationStrides, burstLen, 0);
  int64_t ubStride = deriveTileLoop2Stride("row_major", contract.validRows,
                                           contract.validCols);

  auto copyOp = rewriter.create<a5vm::CopyUbufToGmOp>(
      op.getLoc(), sourceBuffer, destinationBuffer,
      rewriter.getStringAttr(contract.destinationLayout),
      rewriter.getI64IntegerAttr(contract.validRows),
      rewriter.getI64IntegerAttr(contract.validCols),
      rewriter.getI64IntegerAttr(burstCount),
      rewriter.getI64IntegerAttr(burstLen),
      rewriter.getI64IntegerAttr(gmStride),
      rewriter.getI64IntegerAttr(ubStride), rewriter.getBoolAttr(false));
  attachStoreContractAttrs(copyOp, contract);
  if (failed(programCopyUbToGmLoops(copyOp, contract, rewriter)))
    return failure();
  return success();
}

} // namespace pto
} // namespace mlir
