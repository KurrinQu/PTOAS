//===- PTOToA5VMLowering.cpp - PTO to A5VM lowering helpers --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/Transforms/A5VMLowering.h"

#include "PTO/IR/A5VM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
#include <utility>

namespace mlir {
namespace pto {

namespace {

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

StringAttr stringifyPipeAttr(PipeAttr pipe, PatternRewriter &rewriter) {
  return rewriter.getStringAttr(stringifyPIPE(pipe.getPipe()));
}

StringAttr stringifyEventAttr(EventAttr event, PatternRewriter &rewriter) {
  return rewriter.getStringAttr(stringifyEVENT(event.getEvent()));
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

int64_t getElementByteSize(Type type) {
  if (auto floatType = dyn_cast<FloatType>(type))
    return (floatType.getWidth() + 7) / 8;
  if (auto intType = dyn_cast<IntegerType>(type))
    return (intType.getWidth() + 7) / 8;
  return 0;
}

void recordStaticValues(ValueRange values, SmallVectorImpl<int64_t> &out) {
  out.clear();
  out.reserve(values.size());
  for (Value value : values)
    out.push_back(getConstInt(value).value_or(ShapedType::kDynamic));
}

void recordStaticSizes(ArrayRef<OpFoldResult> values,
                       SmallVectorImpl<int64_t> &out, bool &hasDynamic) {
  out.clear();
  hasDynamic = false;
  out.reserve(values.size());
  for (OpFoldResult value : values) {
    if (auto attr = dyn_cast<Attribute>(value)) {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
        out.push_back(intAttr.getInt());
        continue;
      }
    } else if (std::optional<int64_t> constant =
                   getConstInt(cast<Value>(value))) {
      out.push_back(*constant);
      continue;
    }
    out.push_back(ShapedType::kDynamic);
    hasDynamic = true;
  }
}

void mergeSubviewTrace(A5VMPartitionTrace &trace, ArrayRef<int64_t> offsets,
                       ArrayRef<int64_t> sizes, bool hasDynamicOffsets,
                       bool hasDynamicSizes) {
  if (trace.offsets.empty()) {
    trace.offsets.assign(offsets.begin(), offsets.end());
    trace.hasDynamicOffsets = hasDynamicOffsets;
  } else {
    size_t count = std::min(trace.offsets.size(), offsets.size());
    for (size_t i = 0; i < count; ++i) {
      if (trace.offsets[i] == ShapedType::kDynamic ||
          offsets[i] == ShapedType::kDynamic) {
        trace.offsets[i] = ShapedType::kDynamic;
        trace.hasDynamicOffsets = true;
        continue;
      }
      trace.offsets[i] += offsets[i];
    }
    trace.hasDynamicOffsets = trace.hasDynamicOffsets || hasDynamicOffsets;
  }

  trace.sizes.assign(sizes.begin(), sizes.end());
  trace.hasDynamicSizes = hasDynamicSizes;
}

Value resolveTensorViewBase(Value value, Attribute &layoutAttr,
                            SmallVectorImpl<int64_t> &shape,
                            SmallVectorImpl<int64_t> &strides) {
  if (!value)
    return {};

  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    return resolveTensorViewBase(part.getSource(), layoutAttr, shape, strides);
  }

  if (auto source = value.getDefiningOp<MakeTensorViewOp>()) {
    layoutAttr = source.getLayoutAttr();
    auto tensorType = dyn_cast<TensorViewType>(source.getResult().getType());
    shape.assign(tensorType.getShape().begin(), tensorType.getShape().end());
    recordStaticValues(source.getStrides(), strides);
    return source.getPtr();
  }

  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    Value base =
        resolveTensorViewBase(subview.getSource(), layoutAttr, shape, strides);
    if (shape.empty()) {
      bool hasDynamicSizes = false;
      recordStaticSizes(subview.getMixedSizes(), shape, hasDynamicSizes);
    }
    return base ? base : value;
  }

  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>()) {
    if (Attribute layout = reinterpret->getAttr("layout"))
      layoutAttr = layout;
    if (shape.empty()) {
      bool hasDynamicSizes = false;
      recordStaticSizes(reinterpret.getMixedSizes(), shape, hasDynamicSizes);
    }
    if (strides.empty()) {
      bool hasDynamicStrides = false;
      recordStaticSizes(reinterpret.getMixedStrides(), strides,
                        hasDynamicStrides);
    }
    Value base =
        resolveTensorViewBase(reinterpret.getSource(), layoutAttr, shape, strides);
    return base ? base : value;
  }

  if (auto cast = value.getDefiningOp<memref::CastOp>()) {
    Value base =
        resolveTensorViewBase(cast.getSource(), layoutAttr, shape, strides);
    return base ? base : value;
  }

  if (auto memrefType = dyn_cast<MemRefType>(value.getType())) {
    if (shape.empty())
      shape.assign(memrefType.getShape().begin(), memrefType.getShape().end());
    if (strides.empty()) {
      int64_t offset = 0;
      if (failed(mlir::getStridesAndOffset(memrefType, strides, offset)))
        strides.assign(shape.size(), ShapedType::kDynamic);
    }
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

void normalizeToPTOGlobalShapeAndStride(ArrayRef<int64_t> shape,
                                        ArrayRef<int64_t> strides,
                                        SmallVectorImpl<int64_t> &globalShape,
                                        SmallVectorImpl<int64_t> &globalStride) {
  constexpr int64_t kRank = 5;
  globalShape.assign(kRank, 1);
  globalStride.assign(kRank, 1);

  size_t shapeRank = std::min<size_t>(shape.size(), kRank);
  size_t strideRank = std::min<size_t>(strides.size(), kRank);
  size_t rank = std::min(shapeRank, strideRank);
  size_t base = kRank - rank;

  for (size_t i = 0; i < rank; ++i) {
    globalShape[base + i] = shape[shape.size() - rank + i];
    globalStride[base + i] = strides[strides.size() - rank + i];
  }

  for (int i = static_cast<int>(kRank) - 2; i >= 0; --i) {
    if (i >= static_cast<int>(base))
      continue;
    if (globalStride[i + 1] == ShapedType::kDynamic ||
        globalShape[i + 1] == ShapedType::kDynamic) {
      globalStride[i] = ShapedType::kDynamic;
      continue;
    }
    globalStride[i] = globalStride[i + 1] * globalShape[i + 1];
  }
}

int64_t packLoopStrideConfig(int64_t first, int64_t second) {
  return (static_cast<int64_t>(first) << 40) | static_cast<int64_t>(second);
}

int64_t packLoopSizeConfig(int64_t loop2, int64_t loop1) {
  return (static_cast<int64_t>(loop2) << 21) | static_cast<int64_t>(loop1);
}

LogicalResult deriveVecNDTransferConfig(ArrayRef<int64_t> shape,
                                        ArrayRef<int64_t> strides,
                                        StringRef tileLayout, Type elementType,
                                        int64_t validRows, int64_t validCols,
                                        SmallVectorImpl<int64_t> &globalShape,
                                        SmallVectorImpl<int64_t> &globalStride,
                                        int64_t &nBurst, int64_t &lenBurst,
                                        int64_t &gmStrideBytes,
                                        int64_t &ubStrideBytes,
                                        int64_t &loop1Size,
                                        int64_t &loop2Size,
                                        int64_t &loop1FirstStrideBytes,
                                        int64_t &loop1SecondStrideBytes,
                                        int64_t &loop2FirstStrideBytes,
                                        int64_t &loop2SecondStrideBytes) {
  if (tileLayout != "row_major")
    return failure();

  int64_t elemBytes = getElementByteSize(elementType);
  if (elemBytes <= 0)
    return failure();

  normalizeToPTOGlobalShapeAndStride(shape, strides, globalShape, globalStride);
  if (globalShape.size() != 5 || globalStride.size() != 5)
    return failure();
  if (llvm::any_of(globalShape, [](int64_t v) { return v == ShapedType::kDynamic; }) ||
      llvm::any_of(globalStride, [](int64_t v) { return v == ShapedType::kDynamic; }))
    return failure();
  if (validRows == ShapedType::kDynamic || validCols == ShapedType::kDynamic)
    return failure();

  nBurst = globalShape[3];
  lenBurst = validCols * elemBytes;
  gmStrideBytes = globalStride[3] * elemBytes;
  ubStrideBytes = validCols * elemBytes;

  int64_t dstStride2 = globalShape[3] * validCols;
  int64_t dstStride1 = globalShape[2] * dstStride2;

  loop2Size = globalShape[1];
  loop1Size = globalShape[2];
  loop2FirstStrideBytes = dstStride1 * elemBytes;
  loop2SecondStrideBytes = globalStride[1] * elemBytes;
  loop1FirstStrideBytes = dstStride2 * elemBytes;
  loop1SecondStrideBytes = globalStride[2] * elemBytes;
  return success();
}

Attribute getGmMemorySpace(MLIRContext *context) {
  return AddressSpaceAttr::get(context, AddressSpace::GM);
}

unsigned getLLVMAddressSpace(Attribute memorySpace) {
  if (auto addrSpace = dyn_cast_or_null<AddressSpaceAttr>(memorySpace))
    return static_cast<unsigned>(addrSpace.getAddressSpace());
  if (auto intAttr = dyn_cast_or_null<IntegerAttr>(memorySpace))
    return static_cast<unsigned>(intAttr.getInt());
  return static_cast<unsigned>(AddressSpace::GM);
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

Value materializeBufferPointer(Value value, Type elementType,
                               Attribute memorySpace,
                               PatternRewriter &rewriter, Location loc) {
  if (!value)
    return {};

  if (auto bind = value.getDefiningOp<BindTileOp>())
    return materializeBufferPointer(bind.getSource(), elementType, memorySpace,
                                    rewriter, loc);

  if (auto cast = value.getDefiningOp<PointerCastOp>()) {
    if (cast.getAddrs().empty())
      return {};
    auto ptrType =
        LLVM::LLVMPointerType::get(rewriter.getContext(),
                                   getLLVMAddressSpace(memorySpace));
    return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, cast.getAddrs().front());
  }

  Value memrefValue = materializeTileBufferView(value, rewriter, loc);
  if (!memrefValue || !isa<BaseMemRefType>(memrefValue.getType()))
    return {};

  Value ptrAsIndex =
      rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(loc, memrefValue);
  Value ptrAsI64 =
      rewriter.create<arith::IndexCastUIOp>(loc, rewriter.getI64Type(), ptrAsIndex);
  auto ptrType =
      LLVM::LLVMPointerType::get(rewriter.getContext(),
                                 getLLVMAddressSpace(memorySpace));
  return rewriter.create<LLVM::IntToPtrOp>(loc, ptrType, ptrAsI64);
}

A5VMPartitionTrace extractPartitionTrace(Value value) {
  A5VMPartitionTrace trace;
  if (auto part = value.getDefiningOp<PartitionViewOp>()) {
    appendStaticSizes(part.getOffsets(), trace.offsets, trace.hasDynamicOffsets);
    appendStaticSizes(part.getSizes(), trace.sizes, trace.hasDynamicSizes);
    return trace;
  }
  if (auto subview = value.getDefiningOp<memref::SubViewOp>()) {
    trace = extractPartitionTrace(subview.getSource());
    SmallVector<int64_t> offsets;
    SmallVector<int64_t> sizes;
    bool hasDynamicOffsets = false;
    bool hasDynamicSizes = false;
    recordStaticSizes(subview.getMixedOffsets(), offsets, hasDynamicOffsets);
    recordStaticSizes(subview.getMixedSizes(), sizes, hasDynamicSizes);
    mergeSubviewTrace(trace, offsets, sizes, hasDynamicOffsets, hasDynamicSizes);
    return trace;
  }
  if (auto reinterpret = value.getDefiningOp<memref::ReinterpretCastOp>())
    return extractPartitionTrace(reinterpret.getSource());
  if (auto cast = value.getDefiningOp<memref::CastOp>())
    return extractPartitionTrace(cast.getSource());
  if (auto unrealized = value.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (!unrealized.getInputs().empty())
      return extractPartitionTrace(unrealized.getInputs().front());
  }
  return trace;
}

A5VMLoadContract extractTLoadContract(TLoadOp op) {
  A5VMLoadContract contract;
  contract.trace = extractPartitionTrace(op.getSrc());
  contract.elementType = getElementType(op.getDst());

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
  contract.loopScope.loweredAttr = kLoweredLoopScopeAttrName;
  contract.loopScope.loopDepth = 0;
  return contract;
}

A5VMStoreContract extractTStoreContract(TStoreOp op) {
  A5VMStoreContract contract;
  contract.trace = extractPartitionTrace(op.getDst());

  contract.srcDomain = deriveTileDomain(getMemorySpace(op.getSrc()));
  deriveValidShape(op.getSrc(), contract.validRows, contract.validCols);
  contract.elementType = getElementType(op.getSrc());

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
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  normalizeToPTOGlobalShapeAndStride(contract.sourceShape, contract.sourceStrides,
                                     globalShape, globalStride);
  op->setAttr("layout", builder.getStringAttr(contract.sourceLayout));
  op->setAttr("g_shape", asI64ArrayAttr(builder, globalShape));
  op->setAttr("g_strides", asI64ArrayAttr(builder, globalStride));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  op->setAttr("sid", builder.getI64IntegerAttr(0));
  op->setAttr("left_padding_count", builder.getI64IntegerAttr(0));
  op->setAttr("right_padding_count", builder.getI64IntegerAttr(0));
  op->setAttr("l2_cache_ctl", builder.getI64IntegerAttr(0));
  op->setAttr("data_select_bit",
              builder.getBoolAttr(contract.padMode != "none" || contract.padValue ||
                                  contract.leftPaddingNum || contract.rightPaddingNum));
}

void attachStoreContractAttrs(Operation *op, const A5VMStoreContract &contract) {
  Builder builder(op->getContext());
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  normalizeToPTOGlobalShapeAndStride(contract.destinationShape,
                                     contract.destinationStrides, globalShape,
                                     globalStride);
  op->setAttr("g_shape", asI64ArrayAttr(builder, globalShape));
  op->setAttr("g_strides", asI64ArrayAttr(builder, globalStride));
  op->setAttr("valid_rows", builder.getI64IntegerAttr(contract.validRows));
  op->setAttr("valid_cols", builder.getI64IntegerAttr(contract.validCols));
  op->setAttr("sid", builder.getI64IntegerAttr(0));
  op->setAttr("reserved", builder.getI64IntegerAttr(0));
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
  loopOp->setAttr(contract.loweredAttr, rewriter.getUnitAttr());
  return success();
}

void set_loop2_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop1_stride_outtoub(Operation *copyOp, int64_t dstStride,
                              int64_t srcStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_outtoub",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(dstStride, srcStride)));
}

void set_loop_size_outtoub(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_outtoub",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

void set_loop2_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop2_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop1_stride_ubtoout(Operation *copyOp, int64_t srcStride,
                              int64_t dstStride, Builder &builder) {
  copyOp->setAttr("a5vm.set_loop1_stride_ubtoout",
                  builder.getI64IntegerAttr(
                      packLoopStrideConfig(srcStride, dstStride)));
}

void set_loop_size_ubtoout(Operation *copyOp, int64_t loop2, int64_t loop1,
                           Builder &builder) {
  copyOp->setAttr("a5vm.set_loop_size_ubtoout",
                  builder.getI64IntegerAttr(packLoopSizeConfig(loop2, loop1)));
}

LogicalResult programCopyGmToUbLoops(Operation *copyOp,
                                     const A5VMLoadContract &contract,
                                     Builder &builder) {
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t nBurst = 0, lenBurst = 0, gmStrideBytes = 0, ubStrideBytes = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1DstStrideBytes = 0, loop1SrcStrideBytes = 0;
  int64_t loop2DstStrideBytes = 0, loop2SrcStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.sourceShape, contract.sourceStrides,
                                       contract.tileLayout, contract.elementType,
                                       contract.validRows, contract.validCols,
                                       globalShape, globalStride, nBurst, lenBurst,
                                       gmStrideBytes, ubStrideBytes, loop1Size,
                                       loop2Size, loop1DstStrideBytes,
                                       loop1SrcStrideBytes, loop2DstStrideBytes,
                                       loop2SrcStrideBytes)))
    return failure();

  set_loop2_stride_outtoub(copyOp, loop2DstStrideBytes, loop2SrcStrideBytes, builder);
  set_loop1_stride_outtoub(copyOp, loop1DstStrideBytes, loop1SrcStrideBytes, builder);
  set_loop_size_outtoub(copyOp, loop2Size, loop1Size, builder);
  return success();
}

LogicalResult programCopyUbToGmLoops(Operation *copyOp,
                                     const A5VMStoreContract &contract,
                                     Builder &builder) {
  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t nBurst = 0, lenBurst = 0, burstDstStrideBytes = 0, burstSrcStrideBytes = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1SrcStrideBytes = 0, loop1DstStrideBytes = 0;
  int64_t loop2SrcStrideBytes = 0, loop2DstStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.destinationShape,
                                       contract.destinationStrides,
                                       "row_major", contract.elementType,
                                       contract.validRows, contract.validCols,
                                       globalShape, globalStride, nBurst, lenBurst,
                                       burstDstStrideBytes, burstSrcStrideBytes,
                                       loop1Size, loop2Size, loop1SrcStrideBytes,
                                       loop1DstStrideBytes, loop2SrcStrideBytes,
                                       loop2DstStrideBytes)))
    return failure();

  set_loop_size_ubtoout(copyOp, loop2Size, loop1Size, builder);
  set_loop1_stride_ubtoout(copyOp, loop1SrcStrideBytes, loop1DstStrideBytes, builder);
  set_loop2_stride_ubtoout(copyOp, loop2SrcStrideBytes, loop2DstStrideBytes, builder);
  return success();
}

LogicalResult buildUnaryVecScope(StringRef family,
                                 const A5VMUnaryContract &contract, Value src,
                                 Value dst, PatternRewriter &rewriter,
                                 Location loc) {
  auto vecType = getA5VMVecType(rewriter.getContext(), contract.elementType);
  if (!vecType)
    return emitError(loc) << "unsupported A5VM unary element type";

  Value srcBuffer = materializeBufferPointer(src, contract.elementType,
                                             getMemorySpace(src), rewriter, loc);
  Value dstBuffer = materializeBufferPointer(dst, contract.elementType,
                                             getMemorySpace(dst), rewriter, loc);
  if (!srcBuffer || !dstBuffer)
    return emitError(loc) << "requires pointer-backed tile buffers for unary lowering";

  int64_t validRows = contract.validRows;
  int64_t validCols = contract.validCols;
  int64_t vectorWidth = vecType.getElementCount();
  if (validRows == ShapedType::kDynamic || validCols == ShapedType::kDynamic)
    return emitError(loc) << "TABS lowering requires static valid rows and cols";
  int64_t totalElements = validRows * validCols;
  if (totalElements % vectorWidth != 0)
    return emitError(loc)
           << "TABS lowering requires total valid elements divisible by vector width";

  Value c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
  Value c1 = rewriter.create<arith::ConstantIndexOp>(loc, 1);
  Value totalElementsValue =
      rewriter.create<arith::ConstantIndexOp>(loc, totalElements);
  Value vectorStepValue =
      rewriter.create<arith::ConstantIndexOp>(loc, vectorWidth);

  auto aivScopeLoop = rewriter.create<scf::ForOp>(loc, c0, c1, c1);
  if (failed(attachLoopScopeMetadata(aivScopeLoop, contract.loopScope, rewriter)))
    return emitError(loc) << "failed to attach AIV loop scope metadata";

  OpBuilder::InsertionGuard aivGuard(rewriter);
  rewriter.setInsertionPointToStart(aivScopeLoop.getBody());
  auto chunkLoop =
      rewriter.create<scf::ForOp>(loc, c0, totalElementsValue, vectorStepValue);

  OpBuilder::InsertionGuard chunkGuard(rewriter);
  rewriter.setInsertionPointToStart(chunkLoop.getBody());
  Value offset = chunkLoop.getInductionVar();
  auto vlds = rewriter.create<a5vm::VldsOp>(loc, vecType, srcBuffer, offset);
  auto vabs = rewriter.create<a5vm::VabsOp>(loc, vecType, vlds.getResult());
  (void)family;
  rewriter.create<a5vm::VstsOp>(loc, vabs.getResult(), dstBuffer, offset,
                                StringAttr());

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

  Value sourceBuffer =
      materializeBufferPointer(base, getElementType(base),
                               getGmMemorySpace(rewriter.getContext()), rewriter,
                               op.getLoc());
  Value destinationBuffer =
      materializeBufferPointer(op.getDst(), contract.elementType,
                               getMemorySpace(op.getDst()), rewriter, op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t burstCount = 0, burstLen = 0, gmStride = 0, ubStride = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1DstStrideBytes = 0, loop1SrcStrideBytes = 0;
  int64_t loop2DstStrideBytes = 0, loop2SrcStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.sourceShape, contract.sourceStrides,
                                       contract.tileLayout, contract.elementType,
                                       contract.validRows, contract.validCols,
                                       globalShape, globalStride, burstCount,
                                       burstLen, gmStride, ubStride, loop1Size,
                                       loop2Size, loop1DstStrideBytes,
                                       loop1SrcStrideBytes, loop2DstStrideBytes,
                                       loop2SrcStrideBytes)))
    return op.emitOpError("requires PTO-compatible vec ND2ND copy_gm_to_ubuf arguments");
  bool ubPad = contract.padMode != "none" || contract.padValue ||
               contract.leftPaddingNum || contract.rightPaddingNum;

  auto copyOp = rewriter.create<a5vm::CopyGmToUbufOp>(
      op.getLoc(), sourceBuffer, destinationBuffer,
      rewriter.getStringAttr(contract.sourceLayout),
      rewriter.getI64IntegerAttr(contract.validRows),
      rewriter.getI64IntegerAttr(contract.validCols),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(burstCount),
      rewriter.getI64IntegerAttr(burstLen),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(0),
      rewriter.getBoolAttr(ubPad),
      rewriter.getI64IntegerAttr(0),
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

  Value sourceBuffer =
      materializeBufferPointer(op.getSrc(), contract.elementType,
                               getMemorySpace(op.getSrc()), rewriter, op.getLoc());
  Value destinationBuffer =
      materializeBufferPointer(base, getElementType(base),
                               getGmMemorySpace(rewriter.getContext()), rewriter,
                               op.getLoc());
  if (!sourceBuffer || !destinationBuffer)
    return op.emitOpError("requires A5-compatible source and destination buffers");

  SmallVector<int64_t> globalShape;
  SmallVector<int64_t> globalStride;
  int64_t burstCount = 0, burstLen = 0, gmStride = 0, ubStride = 0;
  int64_t loop1Size = 0, loop2Size = 0;
  int64_t loop1SrcStrideBytes = 0, loop1DstStrideBytes = 0;
  int64_t loop2SrcStrideBytes = 0, loop2DstStrideBytes = 0;
  if (failed(deriveVecNDTransferConfig(contract.destinationShape,
                                       contract.destinationStrides, "row_major",
                                       contract.elementType, contract.validRows,
                                       contract.validCols, globalShape,
                                       globalStride, burstCount, burstLen,
                                       gmStride, ubStride, loop1Size, loop2Size,
                                       loop1SrcStrideBytes, loop1DstStrideBytes,
                                       loop2SrcStrideBytes, loop2DstStrideBytes)))
    return op.emitOpError("requires PTO-compatible vec ND2ND copy_ubuf_to_gm arguments");

  auto copyOp = rewriter.create<a5vm::CopyUbufToGmOp>(
      op.getLoc(), sourceBuffer, destinationBuffer,
      rewriter.getStringAttr(contract.destinationLayout),
      rewriter.getI64IntegerAttr(contract.validRows),
      rewriter.getI64IntegerAttr(contract.validCols),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(burstCount),
      rewriter.getI64IntegerAttr(burstLen),
      rewriter.getI64IntegerAttr(0),
      rewriter.getI64IntegerAttr(gmStride),
      rewriter.getI64IntegerAttr(ubStride));
  attachStoreContractAttrs(copyOp, contract);
  if (failed(programCopyUbToGmLoops(copyOp, contract, rewriter)))
    return failure();
  return success();
}

LogicalResult lowerSetFlag(SetFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::SetFlagOp>(op.getLoc(),
                                   stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                   stringifyPipeAttr(op.getDstPipe(), rewriter),
                                   stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerWaitFlag(WaitFlagOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::WaitFlagOp>(op.getLoc(),
                                    stringifyPipeAttr(op.getSrcPipe(), rewriter),
                                    stringifyPipeAttr(op.getDstPipe(), rewriter),
                                    stringifyEventAttr(op.getEventId(), rewriter));
  return success();
}

LogicalResult lowerBarrier(BarrierOp op, PatternRewriter &rewriter) {
  rewriter.create<a5vm::PipeBarrierOp>(op.getLoc(),
                                       stringifyPipeAttr(op.getPipe(), rewriter));
  return success();
}

} // namespace pto
} // namespace mlir
