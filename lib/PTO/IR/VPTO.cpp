//===- VPTO.cpp - VPTO dialect -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"

#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <optional>

using namespace mlir;
using namespace mlir::pto;

namespace {

static ParseResult parseBracketedIndices(OpAsmParser &parser,
                                        SmallVectorImpl<OpAsmParser::UnresolvedOperand> &indices) {
  if (parser.parseLSquare())
    return failure();
  do {
    OpAsmParser::UnresolvedOperand index;
    if (parser.parseOperand(index))
      return failure();
    indices.push_back(index);
  } while (succeeded(parser.parseOptionalComma()));
  if (indices.empty())
    return parser.emitError(parser.getCurrentLocation(),
                            "expected at least one index operand");
  return parser.parseRSquare();
}

static void printBracketedIndices(OpAsmPrinter &printer, OperandRange indices) {
  printer << "[";
  printer.printOperands(indices);
  printer << "]";
}

static LogicalResult verifyBufferLikeIndexArity(Operation *op, Type bufferType,
                                                ValueRange indices,
                                                StringRef bufferRole) {
  if (indices.empty())
    return op->emitOpError() << "requires at least one index for "
                             << bufferRole;

  for (Value index : indices) {
    if (!index.getType().isIndex())
      return op->emitOpError() << "requires index-typed operands for "
                               << bufferRole;
  }

  if (isa<pto::PtrType>(bufferType)) {
    if (indices.size() != 1)
      return op->emitOpError()
             << "requires exactly one linearized index when " << bufferRole
             << " is !pto.ptr";
    return success();
  }

  auto memrefType = dyn_cast<BaseMemRefType>(bufferType);
  if (!memrefType)
    return op->emitOpError() << "requires " << bufferRole
                             << " to be memref or !pto.ptr";

  if (indices.size() == 1)
    return success();

  if (indices.size() != static_cast<size_t>(memrefType.getRank()))
    return op->emitOpError()
           << "requires either one linearized index or one index per memref "
              "dimension for "
           << bufferRole << "; got " << indices.size() << " indices for rank "
           << memrefType.getRank();

  return success();
}

} // namespace

static std::string formatVRegType(int64_t elementCount, Type elementType) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.vreg<" << elementCount << "x" << elementType << ">";
  return storage;
}

static std::string formatMaskType(StringRef granularity) {
  std::string storage;
  llvm::raw_string_ostream os(storage);
  os << "!pto.mask<" << granularity << ">";
  return storage;
}

static LogicalResult verifyVRegTypeLike(Operation *op, Type type,
                                       StringRef roleDescription) {
  auto vecType = dyn_cast<VRegType>(type);
  if (!vecType)
    return op->emitOpError() << roleDescription << " must be !pto.vreg<...>";

  return VRegType::verify(
      [&]() { return op->emitOpError() << roleDescription << " "; },
      vecType.getElementCount(), vecType.getElementType());
}

static LogicalResult verifyMaskTypeLike(Operation *op, Type type,
                                        StringRef roleDescription) {
  if (!isa<MaskType>(type))
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  return success();
}

static LogicalResult verifyMaskTypeWithGranularityLike(Operation *op, Type type,
                                                       StringRef roleDescription,
                                                       StringRef granularity) {
  auto maskType = dyn_cast<MaskType>(type);
  if (!maskType)
    return op->emitOpError() << roleDescription << " must be !pto.mask<...>";
  if (maskType.getGranularity() != granularity) {
    return op->emitOpError()
           << roleDescription << " must be " << formatMaskType(granularity);
  }
  return success();
}

static LogicalResult verifyAlignTypeLike(Operation *op, Type type,
                                         StringRef roleDescription) {
  if (!isa<AlignType>(type))
    return op->emitOpError() << roleDescription << " must be !pto.align";
  return success();
}

static bool isSupportedPredicatePattern(StringRef pattern) {
  return pattern == "PAT_ALL" || pattern == "PAT_VL1" || pattern == "PAT_VL2" ||
         pattern == "PAT_VL3" || pattern == "PAT_VL4" || pattern == "PAT_VL8" ||
         pattern == "PAT_VL16" || pattern == "PAT_VL32" ||
         pattern == "PAT_VL64" || pattern == "PAT_VL128" ||
         pattern == "PAT_M3" || pattern == "PAT_M4" || pattern == "PAT_H" ||
         pattern == "PAT_Q" || pattern == "PAT_ALLF";
}

static bool isSupportedPredicateLoadDist(StringRef dist) {
  return dist == "NORM" || dist == "US" || dist == "DS";
}

static bool isSupportedPredicateStoreDist(StringRef dist) {
  return dist == "NORM" || dist == "PK";
}

static bool isSupportedStrideToken(StringRef stride) {
  return stride == "STRIDE_S3_B16" || stride == "STRIDE_S4_B64" ||
         stride == "STRIDE_S8_B32" || stride == "STRIDE_S2_B64" ||
         stride == "STRIDE_VSST_S8_B16";
}

static bool isSupportedPartToken(StringRef part) {
  return part == "LOWER" || part == "HIGHER";
}

static bool isSupportedVldx2DistToken(StringRef dist) {
  return dist == "DINTLV_B8" || dist == "DINTLV_B16" ||
         dist == "DINTLV_B32" || dist == "BDINTLV";
}

static bool isSupportedVstx2DistToken(StringRef dist) {
  return dist == "INTLV_B8" || dist == "INTLV_B16" || dist == "INTLV_B32";
}

static bool isSupportedPostMode(StringRef mode) {
  return mode == "NO_POST_UPDATE" || mode == "POST_UPDATE";
}

static bool isSupportedVstuMode(StringRef mode) {
  return mode == "POST_UPDATE" || mode == "NO_POST_UPDATE";
}

static std::optional<StringRef> getOptionalPostModeAttr(Operation *op) {
  if (auto mode = op->getAttrOfType<StringAttr>("mode"))
    return mode.getValue();
  return std::nullopt;
}

static unsigned getIntOrFloatBitWidth(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth();
  if (auto floatType = dyn_cast<FloatType>(type))
    return floatType.getWidth();
  return 0;
}

static LogicalResult verifyIntegerVRegTypeLike(Operation *op, Type type,
                                              StringRef roleDescription) {
  if (failed(verifyVRegTypeLike(op, type, roleDescription)))
    return failure();
  auto vecType = cast<VRegType>(type);
  if (!isa<IntegerType>(vecType.getElementType()))
    return op->emitOpError()
           << roleDescription << " must use integer vector element type";
  return success();
}

enum class MemoryRole {
  Unknown,
  GM,
  UB,
  Other,
};

static MemoryRole classifyMemoryRole(Type type) {
  auto memrefType = dyn_cast<BaseMemRefType>(type);
  if (!memrefType) {
    if (auto ptrType = dyn_cast<pto::PtrType>(type)) {
      switch (ptrType.getMemorySpace().getAddressSpace()) {
      case pto::AddressSpace::GM:
      case pto::AddressSpace::Zero:
        return MemoryRole::GM;
      case pto::AddressSpace::VEC:
        return MemoryRole::UB;
      default:
        return MemoryRole::Other;
      }
    }
    return MemoryRole::Other;
  }

  Attribute memorySpace = memrefType.getMemorySpace();
  if (!memorySpace)
    return MemoryRole::Unknown;

  if (auto addrSpace = dyn_cast<pto::AddressSpaceAttr>(memorySpace)) {
    switch (addrSpace.getAddressSpace()) {
    case pto::AddressSpace::GM:
    case pto::AddressSpace::Zero:
      return MemoryRole::GM;
    case pto::AddressSpace::VEC:
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  if (auto intAttr = dyn_cast<IntegerAttr>(memorySpace)) {
    switch (intAttr.getInt()) {
    case static_cast<int64_t>(pto::AddressSpace::GM):
    case static_cast<int64_t>(pto::AddressSpace::Zero):
      return MemoryRole::GM;
    case static_cast<int64_t>(pto::AddressSpace::VEC):
      return MemoryRole::UB;
    default:
      return MemoryRole::Other;
    }
  }

  return MemoryRole::Other;
}

static bool isBufferLike(Type type) {
  return isa<BaseMemRefType, pto::PtrType>(type);
}

static int64_t getPtrElementByteSize(Type type) {
  auto ptrType = dyn_cast<pto::PtrType>(type);
  if (!ptrType)
    return 0;

  Type elementType = ptrType.getElementType();
  if (auto floatType = dyn_cast<FloatType>(elementType))
    return (floatType.getWidth() + 7) / 8;
  if (auto intType = dyn_cast<IntegerType>(elementType))
    return (intType.getWidth() + 7) / 8;
  return 0;
}

static bool isPointerBuffer(Type type) {
  return isa<LLVM::LLVMPointerType, pto::PtrType>(type);
}

template <typename CopyOp>
static LogicalResult verifyCopyGmToUbufOp(CopyOp op, bool expectSourceGM) {
  auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
  auto destinationType = dyn_cast<pto::PtrType>(op.getDestination().getType());
  if (!sourceType || !destinationType)
    return op.emitOpError("requires typed !pto.ptr source and destination");

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getPtrElementByteSize(sourceType);
  int64_t destinationElemBytes = getPtrElementByteSize(destinationType);
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0)
    return op.emitOpError("requires copy source and destination element types with known byte width");
  if (sourceElemBytes != destinationElemBytes)
    return op.emitOpError("requires source and destination element byte widths to match");

  return success();
}

template <typename CopyOp>
static LogicalResult verifyCopyUbufToGmOp(CopyOp op, bool expectSourceGM) {
  auto sourceType = dyn_cast<pto::PtrType>(op.getSource().getType());
  auto destinationType = dyn_cast<pto::PtrType>(op.getDestination().getType());
  if (!sourceType || !destinationType)
    return op.emitOpError("requires typed !pto.ptr source and destination");

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  bool directionMatches = true;
  if (expectSourceGM) {
    directionMatches &= sourceRole != MemoryRole::UB;
    directionMatches &= destinationRole != MemoryRole::GM;
  } else {
    directionMatches &= sourceRole != MemoryRole::GM;
    directionMatches &= destinationRole != MemoryRole::UB;
  }

  if (!directionMatches) {
    return op.emitOpError()
           << "requires "
           << (expectSourceGM ? "GM source and UB destination"
                              : "UB source and GM destination");
  }

  int64_t sourceElemBytes = getPtrElementByteSize(sourceType);
  int64_t destinationElemBytes = getPtrElementByteSize(destinationType);
  if (sourceElemBytes <= 0 || destinationElemBytes <= 0)
    return op.emitOpError("requires copy source and destination element types with known byte width");
  if (sourceElemBytes != destinationElemBytes)
    return op.emitOpError("requires source and destination element byte widths to match");

  return success();
}

Type VRegType::parse(AsmParser &parser) {
  SmallVector<int64_t, 1> shape;
  Type elementType;
  SMLoc loc = parser.getCurrentLocation();

  if (failed(parser.parseLess()) ||
      failed(parser.parseDimensionList(shape, /*allowDynamic=*/false,
                                       /*withTrailingX=*/true)) ||
      shape.size() != 1 || failed(parser.parseType(elementType)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<VRegType>(loc, parser.getContext(), shape.front(),
                                    elementType);
}

void VRegType::print(AsmPrinter &printer) const {
  printer << "<" << getElementCount() << "x";
  printer.printType(getElementType());
  printer << ">";
}

LogicalResult VRegType::verify(function_ref<InFlightDiagnostic()> emitError,
                              int64_t elementCount, Type elementType) {
  if (elementCount <= 0)
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected a positive element count";

  auto intOrFloat = mlir::dyn_cast<IntegerType>(elementType);
  unsigned elementBitWidth = 0;
  if (intOrFloat) {
    elementBitWidth = intOrFloat.getWidth();
  } else if (auto floatType = mlir::dyn_cast<FloatType>(elementType)) {
    elementBitWidth = floatType.getWidth();
  } else {
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected an integer or floating-point element type";
  }

  if (elementCount * static_cast<int64_t>(elementBitWidth) != 2048)
    return emitError() << "'" << formatVRegType(elementCount, elementType)
                       << "' expected exactly 256 bytes";

  return success();
}

bool MaskType::isSupportedGranularity(StringRef granularity) {
  return granularity == "b8" || granularity == "b16" ||
         granularity == "b32";
}

Type MaskType::parse(AsmParser &parser) {
  auto loc = parser.getCurrentLocation();
  StringRef granularity;
  if (failed(parser.parseLess()) || failed(parser.parseKeyword(&granularity)) ||
      failed(parser.parseGreater()))
    return {};

  return parser.getChecked<MaskType>(loc, parser.getContext(), granularity);
}

void MaskType::print(AsmPrinter &printer) const {
  printer << "<" << getGranularity() << ">";
}

LogicalResult
MaskType::verify(function_ref<InFlightDiagnostic()> emitError,
                 StringRef granularity) {
  if (!isSupportedGranularity(granularity))
    return emitError() << "'" << formatMaskType(granularity)
                       << "' expected granularity to be one of b8, b16, b32";
  return success();
}

void CopyGmToUbufOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyGmToUbufOp::verify() {
  return verifyCopyGmToUbufOp(*this, true);
}

LogicalResult VbrOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();

  auto resultVecType = cast<VRegType>(getResult().getType());
  Type elementType = getValue().getType();
  if (isa<ShapedType, VectorType>(elementType))
    return emitOpError("value must be a scalar matching the result element type");
  if (elementType != resultVecType.getElementType())
    return emitOpError("value type must match result element type");
  return success();
}

LogicalResult VcaddOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VcminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "input")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("input and result must have the same vector type");
  return success();
}

LogicalResult VciOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !pto.vreg<...>");
  if (!isa<IntegerType>(resultType.getElementType()))
    return emitOpError("result element type must be integer");
  auto indexType = dyn_cast<IntegerType>(getIndex().getType());
  if (!indexType)
    return emitOpError("index must be an integer scalar");
  if (indexType != resultType.getElementType())
    return emitOpError("index type must match result element type");
  return success();
}

void Vgather2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2Op::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  if (!isa<IntegerType>(offsetsType.getElementType()))
    return emitOpError("offset vector must use integer element type");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

LogicalResult CopyUbufToUbufOp::verify() {
  if (!isBufferLike(getSource().getType()) || !isBufferLike(getDestination().getType()))
    return emitOpError("requires pointer-like source and destination");
  if (classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getDestination().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed source and destination");
  return success();
}

void VgatherbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VgatherbOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

void Vgather2BcOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vgather2BcOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();

  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!offsetsType || !resultType)
    return emitOpError("offsets and result must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != resultType.getElementCount())
    return emitOpError("offset and result vectors must have the same element count");
  return success();
}

LogicalResult VbitsortOp::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource().getType()) ||
      !isBufferLike(getIndices().getType()))
    return emitOpError("requires pointer-like destination/source/indices");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getIndices().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination/source/indices");
  if (!getRepeatTimes().getType().isIndex())
    return emitOpError("repeat_times must be index");
  return success();
}

LogicalResult Vmrgsort4Op::verify() {
  if (!isBufferLike(getDestination().getType()) || !isBufferLike(getSource0().getType()) ||
      !isBufferLike(getSource1().getType()) || !isBufferLike(getSource2().getType()) ||
      !isBufferLike(getSource3().getType()))
    return emitOpError("requires pointer-like destination and sources");
  if (classifyMemoryRole(getDestination().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource0().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource1().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource2().getType()) != MemoryRole::UB ||
      classifyMemoryRole(getSource3().getType()) != MemoryRole::UB)
    return emitOpError("requires UB-backed destination and sources");
  return success();
}

LogicalResult VmaxOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
}

LogicalResult VminOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result")))
    return failure();
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError("lhs, rhs, and result must have the same vector type");
  return success();
}

void VldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

ParseResult VldsOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand source;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
  Type sourceType;
  SmallVector<Type, 1> resultTypes;

  if (parser.parseOperand(source) || parseBracketedIndices(parser, indices) ||
      parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(sourceType) ||
      parser.parseArrowTypeList(resultTypes))
    return failure();

  if (resultTypes.size() != 1)
    return parser.emitError(parser.getCurrentLocation(),
                            "expected exactly one result type");
  result.addTypes(resultTypes);

  if (parser.resolveOperand(source, sourceType, result.operands) ||
      parser.resolveOperands(indices, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();

  return success();
}

void VldsOp::print(OpAsmPrinter &printer) {
  printer << " " << getSource();
  printBracketedIndices(printer, getIndices());
  printer.printOptionalAttrDict((*this)->getAttrs());
  printer << " : " << getSource().getType() << " -> " << getResult().getType();
}

template <typename LoadOp>
static LogicalResult verifyVldsCommon(LoadOp op, ValueRange indices) {
  if (!isBufferLike(op.getSource().getType()))
    return op.emitOpError("requires a buffer-like source (memref or !pto.ptr)");

  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();

  MemoryRole sourceRole = classifyMemoryRole(op.getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return op.emitOpError("requires a UB-backed source");

  if (failed(verifyBufferLikeIndexArity(op.getOperation(),
                                        op.getSource().getType(), indices,
                                        "source")))
    return failure();

  if (op.getDistAttr()) {
    StringRef dist = *op.getDist();
    if (dist != "NORM" && dist != "BLK" && dist != "DINTLV_B32" &&
        dist != "UNPK_B16")
      return op.emitOpError(
          "supports only NORM, BLK, DINTLV_B32, and UNPK_B16 distributions");
  }

  return success();
}

LogicalResult VldsOp::verify() {
  if (failed(verifyVldsCommon(*this, getIndices())))
    return failure();
  if (std::optional<StringRef> mode = getOptionalPostModeAttr(getOperation());
      mode && !isSupportedPostMode(*mode))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}
void VldsPostOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldsPostOp::verify() {
  if (failed(verifyVldsCommon(*this, ValueRange{getOffset()})))
    return failure();
  if (!isPointerBuffer(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (getUpdatedSource().getType() != getSource().getType())
    return emitOpError("requires updated source result to match source type");
  return success();
}

void VldasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldasOp::verify() {
  if (!isPointerBuffer(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (failed(verifyAlignTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

void VldusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VldusOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlign().getType(), "align type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")) ||
      failed(verifyAlignTypeLike(*this, getUpdatedAlign().getType(),
                                 "updated align type")))
    return failure();
  if (!isPointerBuffer(getSource().getType()))
    return emitOpError("requires a pointer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (getUpdatedSource().getType() != getSource().getType())
    return emitOpError(
        "requires updated source result to match source type");
  return success();
}

void UvldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult UvldOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");

  auto sourceMemRef = dyn_cast<BaseMemRefType>(getSource().getType());
  if (!sourceMemRef)
    return success();

  Type sourceElementType = sourceMemRef.getElementType();
  Type vectorElementType = cast<VRegType>(getResult().getType()).getElementType();
  if (sourceElementType != vectorElementType)
    return emitOpError(
        "requires source element type to match vector element type");
  return success();
}

LogicalResult VdupOp::verify() {
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!resultType)
    return emitOpError("result must be !pto.vreg<...>");

  Type inputType = getInput().getType();
  if (auto inputVecType = dyn_cast<VRegType>(inputType)) {
    if (inputVecType != resultType)
      return emitOpError("vector input must match result vector type");
    return success();
  }

  if (inputType != resultType.getElementType())
    return emitOpError("scalar input must match result element type");

  return success();
}

LogicalResult PsetB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8")))
    return failure();

  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PsetB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16")))
    return failure();

  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PsetB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b8")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b16")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

LogicalResult PgeB32Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getResult().getType(),
                                               "result type", "b32")))
    return failure();
  if (!isSupportedPredicatePattern(getPattern()))
    return emitOpError("requires a supported PAT_* predicate pattern");
  return success();
}

template <typename PltOp>
static LogicalResult verifyPredicateLaneCountOp(PltOp op,
                                                StringRef granularity) {
  if (failed(verifyMaskTypeWithGranularityLike(op, op.getMask().getType(),
                                               "mask type", granularity)))
    return failure();
  Type scalarType = op.getScalar().getType();
  auto scalarIntType = dyn_cast<IntegerType>(scalarType);
  if (!scalarIntType || scalarIntType.getWidth() != 32)
    return op.emitOpError("requires scalar to be i32");
  if (op.getScalarOut().getType() != scalarType)
    return op.emitOpError("requires scalar_out to match scalar type");
  return success();
}

LogicalResult PltB8Op::verify() { return verifyPredicateLaneCountOp(*this, "b8"); }
LogicalResult PltB16Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b16");
}
LogicalResult PltB32Op::verify() {
  return verifyPredicateLaneCountOp(*this, "b32");
}

LogicalResult PpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

LogicalResult PunpackOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getPart() != "LOWER")
    return emitOpError("currently supports only LOWER part");
  return success();
}

LogicalResult PnotOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getInput().getType(), "input type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  return success();
}

LogicalResult PselOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyMaskTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  return success();
}

void PldsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldsOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  MemoryRole sourceRole = classifyMemoryRole(getSource().getType());
  if (sourceRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  return success();
}

void PldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedPredicateLoadDist(getDist()))
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  return success();
}

void PldiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult PldiOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!isSupportedPredicateLoadDist(getDist()))
    return emitOpError("requires predicate load dist to be NORM, US, or DS");
  return success();
}

template <typename OpTy>
static LogicalResult verifyVecScalarOpLike(OpTy op) {
  auto inputType = dyn_cast<VRegType>(op.getInput().getType());
  auto resultType = dyn_cast<VRegType>(op.getResult().getType());
  if (!inputType || !resultType)
    return op.emitOpError("input and result must be !pto.vreg<...>");
  if (inputType != resultType)
    return op.emitOpError("input and result vector types must match");
  if (op.getScalar().getType() != inputType.getElementType())
    return op.emitOpError("scalar type must match vector element type");
  return success();
}

template <typename CarryOp>
static LogicalResult verifyCarryVecOp(CarryOp op) {
  if (failed(verifyIntegerVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")) ||
      failed(verifyIntegerVRegTypeLike(op, op.getResult().getType(),
                                      "result type")) ||
      failed(verifyMaskTypeLike(op, op.getCarry().getType(), "carry type")))
    return failure();

  auto lhsType = cast<VRegType>(op.getLhs().getType());
  auto rhsType = cast<VRegType>(op.getRhs().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  auto lhsElemType = cast<IntegerType>(lhsType.getElementType());
  if (lhsType != rhsType || lhsType != resultType)
    return op.emitOpError("requires lhs, rhs, and result to have matching vector types");
  if (lhsElemType.getWidth() != 32)
    return op.emitOpError("currently requires 32-bit integer vector elements");
  return success();
}

template <typename CarryWithInputOp>
static LogicalResult verifyCarryVecOpWithInput(CarryWithInputOp op) {
  if (failed(verifyCarryVecOp(op)) ||
      failed(verifyMaskTypeLike(op, op.getCarryIn().getType(),
                                "carry_in type")))
    return failure();
  return success();
}

LogicalResult VmulsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VaddsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VmaxsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VminsOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VlreluOp::verify() { return verifyVecScalarOpLike(*this); }
LogicalResult VshlsOp::verify() {
  if (failed(verifyVecScalarOpLike(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()) ||
      !isa<IntegerType>(getScalar().getType()))
    return emitOpError("requires integer vector and integer scalar");
  return success();
}
LogicalResult VshrsOp::verify() {
  if (failed(verifyVecScalarOpLike(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()) ||
      !isa<IntegerType>(getScalar().getType()))
    return emitOpError("requires integer vector and integer scalar");
  return success();
}

LogicalResult VabsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getInput().getType() != getResult().getType())
    return emitOpError("requires matching register vector shape");
  return success();
}

template <typename UnaryOp>
static LogicalResult verifyUnaryVecOp(UnaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getInput().getType(), "operand type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getInput().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shape");
  return success();
}

LogicalResult VexpOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VlnOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VsqrtOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VrecOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VreluOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VnotOp::verify() { return verifyUnaryVecOp(*this); }
LogicalResult VbcntOp::verify() {
  if (failed(verifyUnaryVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VclsOp::verify() {
  if (failed(verifyUnaryVecOp(*this)))
    return failure();
  auto inputType = cast<VRegType>(getInput().getType());
  if (!isa<IntegerType>(inputType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}

template <typename BinaryOp>
static LogicalResult verifyBinaryVecOp(BinaryOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();
  if (failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires matching register vector shapes");
  return success();
}

LogicalResult VaddOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VsubOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VmulOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VdivOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VandOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VorOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VxorOp::verify() { return verifyBinaryVecOp(*this); }
LogicalResult VshlOp::verify() {
  if (failed(verifyBinaryVecOp(*this)))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  if (!isa<IntegerType>(lhsType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VshrOp::verify() {
  if (failed(verifyBinaryVecOp(*this)))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  if (!isa<IntegerType>(lhsType.getElementType()))
    return emitOpError("requires integer vector element type");
  return success();
}
LogicalResult VaddcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VsubcOp::verify() { return verifyCarryVecOp(*this); }
LogicalResult VaddcsOp::verify() { return verifyCarryVecOpWithInput(*this); }
LogicalResult VsubcsOp::verify() { return verifyCarryVecOpWithInput(*this); }

template <typename SelectOp>
static LogicalResult verifyLaneSelectOp(SelectOp op) {
  if (failed(verifyVRegTypeLike(op, op.getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(op, op.getSrc1().getType(), "src1 type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();

  auto src0Type = cast<VRegType>(op.getSrc0().getType());
  auto src1Type = cast<VRegType>(op.getSrc1().getType());
  auto resultType = cast<VRegType>(op.getResult().getType());
  if (src0Type != resultType)
    return op.emitOpError("requires src0 and result to have identical vector types");
  if (src1Type.getElementCount() != src0Type.getElementCount())
    return op.emitOpError("requires src0/src1 to have identical element counts");
  auto src1ElemType = dyn_cast<IntegerType>(src1Type.getElementType());
  if (!src1ElemType)
    return op.emitOpError("requires src1 to use integer vector elements");
  if (src1ElemType.getWidth() != getIntOrFloatBitWidth(src0Type.getElementType()))
    return op.emitOpError("requires src1 integer element width to match src0 element width");
  return success();
}

template <typename PairOp>
static LogicalResult verifyPairVecResults(PairOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(op, op.getHigh().getType(), "high result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getLow().getType() ||
      op.getLhs().getType() != op.getHigh().getType())
    return op.emitOpError("requires operands and results to share one vector type");
  return success();
}

template <typename PartOp>
static LogicalResult verifyPartVecOp(PartOp op) {
  if (failed(verifyVRegTypeLike(op, op.getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(op, op.getRhs().getType(), "rhs type")) ||
      failed(verifyVRegTypeLike(op, op.getResult().getType(), "result type")))
    return failure();
  if (op.getLhs().getType() != op.getRhs().getType() ||
      op.getLhs().getType() != op.getResult().getType())
    return op.emitOpError("requires operands and result to share one vector type");
  if (!isSupportedPartToken(op.getPart()))
    return op.emitOpError("requires part to be LOWER or HIGHER");
  return success();
}

LogicalResult VselOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType() ||
      getSrc0().getType() != getResult().getType())
    return emitOpError("requires src0, src1, and result to have identical vector types");
  return success();
}

LogicalResult VselrOp::verify() { return verifyLaneSelectOp(*this); }
LogicalResult Vselrv2Op::verify() { return verifyLaneSelectOp(*this); }

static bool isSupportedCmpMode(StringRef mode) {
  return mode == "eq" || mode == "ne" || mode == "lt" || mode == "le" ||
         mode == "gt" || mode == "ge";
}

LogicalResult VcmpOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc0().getType(), "src0 type")) ||
      failed(verifyVRegTypeLike(*this, getSrc1().getType(), "src1 type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getSrc0().getType() != getSrc1().getType())
    return emitOpError("requires src0 and src1 to have identical vector types");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

LogicalResult VcmpsOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getSrc().getType(), "src type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyMaskTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  auto srcType = cast<VRegType>(getSrc().getType());
  if (getScalar().getType() != srcType.getElementType())
    return emitOpError("requires scalar type to match source element type");
  if (!isSupportedCmpMode(getCmpMode()))
    return emitOpError("requires cmp_mode to be one of eq/ne/lt/le/gt/ge");
  return success();
}

LogicalResult VcvtOp::verify() {
  auto inputType = dyn_cast<VRegType>(getInput().getType());
  auto resultType = dyn_cast<VRegType>(getResult().getType());
  if (!inputType || !resultType)
    return emitOpError("input and result must be !pto.vreg<...>");

  auto inputElemType = inputType.getElementType();
  auto resultElemType = resultType.getElementType();
  auto isSupportedElemType = [](Type type) {
    return type.isF16() || type.isBF16() || type.isF32();
  };
  if (!isSupportedElemType(inputElemType) || !isSupportedElemType(resultElemType))
    return emitOpError("currently supports only f16/bf16/f32 vector element types");

  if (getRoundModeAttr()) {
    StringRef roundMode = *getRoundMode();
    if (roundMode != "ROUND_R" && roundMode != "ROUND_A" &&
        roundMode != "ROUND_F" && roundMode != "ROUND_C" &&
        roundMode != "ROUND_Z" && roundMode != "ROUND_O")
      return emitOpError("round_mode must be one of ROUND_R/ROUND_A/ROUND_F/ROUND_C/ROUND_Z/ROUND_O");
  }

  if (getSatAttr()) {
    StringRef sat = *getSat();
    if (sat != "RS_ENABLE" && sat != "RS_DISABLE")
      return emitOpError("sat must be RS_ENABLE or RS_DISABLE");
  }

  if (getPartAttr()) {
    StringRef part = *getPart();
    if (part != "PART_EVEN" && part != "PART_ODD")
      return emitOpError("part must be PART_EVEN or PART_ODD");
  }

  return success();
}

LogicalResult PdintlvB8Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b8")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b8")))
    return failure();
  return success();
}

LogicalResult PintlvB16Op::verify() {
  if (failed(verifyMaskTypeWithGranularityLike(*this, getLhs().getType(),
                                               "lhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getRhs().getType(),
                                               "rhs type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getLow().getType(),
                                               "low type", "b16")) ||
      failed(verifyMaskTypeWithGranularityLike(*this, getHigh().getType(),
                                               "high type", "b16")))
    return failure();
  return success();
}

LogicalResult VintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult VdintlvOp::verify() { return verifyPairVecResults(*this); }
LogicalResult Vintlvv2Op::verify() { return verifyPartVecOp(*this); }
LogicalResult Vdintlvv2Op::verify() { return verifyPartVecOp(*this); }

LogicalResult VmullOp::verify() {
  if (failed(verifyPairVecResults(*this)) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  auto lhsType = cast<VRegType>(getLhs().getType());
  auto lhsElemType = dyn_cast<IntegerType>(lhsType.getElementType());
  if (!lhsElemType)
    return emitOpError("requires integer vector element type");
  if (lhsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit integer vector elements");
  return success();
}

LogicalResult VmulaOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getAcc().getType(), "acc type")) ||
      failed(verifyVRegTypeLike(*this, getLhs().getType(), "lhs type")) ||
      failed(verifyVRegTypeLike(*this, getRhs().getType(), "rhs type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (getAcc().getType() != getLhs().getType() ||
      getAcc().getType() != getRhs().getType() ||
      getAcc().getType() != getResult().getType())
    return emitOpError("requires acc, lhs, rhs, and result to share one vector type");
  return success();
}

void VsldOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!isSupportedStrideToken(getStride()))
    return emitOpError("requires a supported STRIDE_* token");
  return success();
}

void Vldx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult Vldx2Op::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low result type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high result type")))
    return failure();
  if (getLow().getType() != getHigh().getType())
    return emitOpError("requires low/high results to share one vector type");
  if (!isSupportedVldx2DistToken(getDist()))
    return emitOpError("requires a supported x2 load distribution token");
  return success();
}

void VstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

ParseResult VstsOp::parse(OpAsmParser &parser, OperationState &result) {
  OpAsmParser::UnresolvedOperand value, destination, mask;
  SmallVector<OpAsmParser::UnresolvedOperand, 4> indices;
  Type valueType, destinationType, maskType;

  if (parser.parseOperand(value) || parser.parseComma() ||
      parser.parseOperand(destination) ||
      parseBracketedIndices(parser, indices) || parser.parseComma() ||
      parser.parseOperand(mask) || parser.parseOptionalAttrDict(result.attributes) ||
      parser.parseColonType(valueType) || parser.parseComma() ||
      parser.parseType(destinationType) || parser.parseComma() ||
      parser.parseType(maskType))
    return failure();

  if (parser.resolveOperand(value, valueType, result.operands) ||
      parser.resolveOperand(destination, destinationType, result.operands) ||
      parser.resolveOperand(mask, maskType, result.operands) ||
      parser.resolveOperands(indices, parser.getBuilder().getIndexType(),
                             result.operands))
    return failure();
  return success();
}

void VstsOp::print(OpAsmPrinter &printer) {
  printer << " " << getValue() << ", " << getDestination();
  printBracketedIndices(printer, getIndices());
  printer << ", " << getMask();
  printer.printOptionalAttrDict((*this)->getAttrs(),
                                /*elidedAttrs=*/{"operandSegmentSizes"});
  printer << " : " << getValue().getType() << ", " << getDestination().getType()
          << ", " << getMask().getType();
}

template <typename StoreOp>
static LogicalResult verifyVstsCommon(StoreOp op, ValueRange indices) {
  if (failed(verifyVRegTypeLike(op, op.getValue().getType(), "value type")))
    return failure();
  if (failed(verifyMaskTypeLike(op, op.getMask().getType(), "mask type")))
    return failure();

  if (!isBufferLike(op.getDestination().getType()))
    return op.emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");

  MemoryRole destinationRole = classifyMemoryRole(op.getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return op.emitOpError("requires a UB-backed destination");

  if (failed(verifyBufferLikeIndexArity(op.getOperation(),
                                        op.getDestination().getType(), indices,
                                        "destination")))
    return failure();

  return success();
}

LogicalResult VstsOp::verify() {
  if (failed(verifyVstsCommon(*this, getIndices())))
    return failure();
  if (std::optional<StringRef> mode = getOptionalPostModeAttr(getOperation());
      mode && !isSupportedPostMode(*mode))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}
void VstsPostOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstsPostOp::verify() {
  if (failed(verifyVstsCommon(*this, ValueRange{getOffset()})))
    return failure();
  if (!isPointerBuffer(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  if (getUpdatedDestination().getType() != getDestination().getType())
    return emitOpError(
        "requires updated destination result to match destination type");
  return success();
}

void Vstx2Op::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getLowMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getHighMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult Vstx2Op::verify() {
  if (failed(verifyVRegTypeLike(*this, getLow().getType(), "low value type")) ||
      failed(verifyVRegTypeLike(*this, getHigh().getType(), "high value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (getLow().getType() != getHigh().getType())
    return emitOpError("requires low/high values to share one vector type");
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedVstx2DistToken(getDist()))
    return emitOpError("requires a supported x2 store distribution token");
  return success();
}

void VscatterOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VscatterOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError("requires a pointer-like destination");
  auto offsetsType = dyn_cast<VRegType>(getOffsets().getType());
  auto valueType = dyn_cast<VRegType>(getValue().getType());
  if (!offsetsType || !valueType)
    return emitOpError("value and offsets must be !pto.vreg<...>");
  auto offsetsElemType = dyn_cast<IntegerType>(offsetsType.getElementType());
  if (!offsetsElemType)
    return emitOpError("offset vector must use integer element type");
  if (offsetsElemType.getWidth() != 32)
    return emitOpError("currently requires 32-bit offset vector elements");
  if (offsetsType.getElementCount() != valueType.getElementCount())
    return emitOpError("offset and value vectors must have the same element count");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getActiveLanes().getType().isIndex())
    return emitOpError("active_lanes must be index");
  return success();
}

void VsldbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
}

LogicalResult VsldbOp::verify() {
  if (!isBufferLike(getSource().getType()))
    return emitOpError("requires a buffer-like source (memref or !llvm.ptr)");
  if (classifyMemoryRole(getSource().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed source");
  if (failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")) ||
      failed(verifyVRegTypeLike(*this, getResult().getType(), "result type")))
    return failure();
  return success();
}

void PstsOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

void PstOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  if (!isSupportedPredicateStoreDist(getDist()))
    return emitOpError("requires predicate store dist to be NORM or PK");
  return success();
}

void PstiOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult PstiOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!isSupportedPredicateStoreDist(getDist()))
    return emitOpError("requires predicate store dist to be NORM or PK");
  return success();
}

LogicalResult PstsOp::verify() {
  if (failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  MemoryRole destinationRole = classifyMemoryRole(getDestination().getType());
  if (destinationRole == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void VsstOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VsstOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!isSupportedStrideToken(getStride()))
    return emitOpError("requires a supported STRIDE_* token");
  return success();
}

void VsstbOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VsstbOp::verify() {
  if (failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyMaskTypeLike(*this, getMask().getType(), "mask type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void VstaOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstaOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  if (!getOffset().getType().isIndex())
    return emitOpError("requires index offset");
  return success();
}

void VstasOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstasOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void VstarOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult VstarOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getValue().getType(), "value type")))
    return failure();
  if (!isBufferLike(getDestination().getType()))
    return emitOpError(
        "requires a buffer-like destination (memref or !llvm.ptr)");
  if (classifyMemoryRole(getDestination().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed destination");
  return success();
}

void PstuOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult PstuOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlignIn().getType(), "align_in type")) ||
      failed(verifyMaskTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isPointerBuffer(getBase().getType()) ||
      !isPointerBuffer(getBaseOut().getType()))
    return emitOpError("requires pointer-only base and base_out");
  if (getBase().getType() != getBaseOut().getType())
    return emitOpError("requires base and base_out to have identical types");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  return success();
}

void VstuOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getOffsetInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VstuOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlignIn().getType(), "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isPointerBuffer(getBase().getType()))
    return emitOpError("requires a pointer-only base");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  if (!getOffsetIn().getType().isIndex() || !getOffsetOut().getType().isIndex())
    return emitOpError("requires index offset_in and offset_out");
  if (!isSupportedVstuMode(getMode()))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}

void VstusOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VstusOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlignIn().getType(), "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isPointerBuffer(getBase().getType()) ||
      !isPointerBuffer(getBaseOut().getType()))
    return emitOpError("requires pointer-only base and base_out");
  if (getBase().getType() != getBaseOut().getType())
    return emitOpError("requires base and base_out to have identical types");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  if (!isSupportedPostMode(getMode()))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}

void VsturOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getAlignInMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getValueMutable());
  effects.emplace_back(MemoryEffects::Read::get(), &getBaseMutable());
}

LogicalResult VsturOp::verify() {
  if (failed(verifyAlignTypeLike(*this, getAlignIn().getType(), "align_in type")) ||
      failed(verifyVRegTypeLike(*this, getValue().getType(), "value type")) ||
      failed(verifyAlignTypeLike(*this, getAlignOut().getType(), "align_out type")))
    return failure();
  if (!isPointerBuffer(getBase().getType()))
    return emitOpError("requires a pointer-only base");
  if (classifyMemoryRole(getBase().getType()) == MemoryRole::GM)
    return emitOpError("requires a UB-backed base");
  if (!isSupportedPostMode(getMode()))
    return emitOpError("requires mode to be POST_UPDATE or NO_POST_UPDATE");
  return success();
}

void CopyUbufToGmOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Read::get(), &getSourceMutable());
  effects.emplace_back(MemoryEffects::Write::get(), &getDestinationMutable());
}

LogicalResult CopyUbufToGmOp::verify() {
  return verifyCopyUbufToGmOp(*this, false);
}
