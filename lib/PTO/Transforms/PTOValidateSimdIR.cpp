#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>
#include <optional>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOVALIDATESIMDIR
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {

static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibAttrEntryRole =
    "pto.oplib.entry_role";
static constexpr llvm::StringLiteral kOpLibAttrOp = "pto.oplib.op";
static constexpr llvm::StringLiteral kOpLibAttrMatchRows =
    "pto.oplib.match.rows";
static constexpr llvm::StringLiteral kOpLibAttrMatchCols =
    "pto.oplib.match.cols";
static constexpr llvm::StringLiteral kOpLibAttrMatchBLayout =
    "pto.oplib.match.blayout";
static constexpr llvm::StringLiteral kOpLibAttrMatchSLayout =
    "pto.oplib.match.slayout";
static constexpr llvm::StringLiteral kOpLibAttrMatchFractal =
    "pto.oplib.match.fractal";
static constexpr llvm::StringLiteral kOpLibAttrMatchScalarPos =
    "pto.oplib.match.scalar_pos";
static constexpr llvm::StringLiteral kOpLibAttrMatchCmpMode =
    "pto.oplib.match.cmp_mode";
static constexpr llvm::StringLiteral kOpLibAttrMatchIsBinary =
    "pto.oplib.match.is_binary";
static constexpr llvm::StringLiteral kOpLibKindL3BinaryTemplate =
    "l3_binary_elementwise_template";
static constexpr llvm::StringLiteral kEntryRoleVariant = "variant";
static constexpr llvm::StringLiteral kEntryRoleSeed = "seed";
static constexpr llvm::StringLiteral kSimdLevelAttr = "pto.simd.level";
static constexpr llvm::StringLiteral kSimdLanesAttr = "pto.simd.lanes";
static constexpr llvm::StringLiteral kSimdCoreSlotAttr = "pto.simd.core_slot";
static constexpr llvm::StringLiteral kSimdVldDistAttr = "pto.simd.vld_dist";
static constexpr llvm::StringLiteral kSimdVstDistAttr = "pto.simd.vst_dist";
static constexpr llvm::StringLiteral kSimdExecModeAttr = "pto.simd.exec_mode";
static constexpr llvm::StringLiteral kSimdLevelBinaryEwiseV1 =
    "binary_ewise_v1";
static constexpr llvm::StringLiteral kSimdCoreSlotBinaryEwise =
    "binary_ewise_core";

static constexpr llvm::StringLiteral kErrEmptyBody =
    "E_OPLIB_EMPTY_BODY_FOR_SIMD";
static constexpr llvm::StringLiteral kErrLanesMismatch =
    "E_OPLIB_SIMD_LANES_MISMATCH";
static constexpr llvm::StringLiteral kErrCoreSlot =
    "E_OPLIB_SIMD_INVALID_CORE_SLOT";
static constexpr llvm::StringLiteral kErrDType =
    "E_OPLIB_SIMD_UNSUPPORTED_DTYPE";
static constexpr llvm::StringLiteral kErrLayout =
    "E_OPLIB_SIMD_UNSUPPORTED_LAYOUT";
static constexpr llvm::StringLiteral kErrBodyDisallowedIR =
    "E_OPLIB_BODY_DISALLOWED_IR";
static constexpr llvm::StringLiteral kErrSimdAttrRequired =
    "E_OPLIB_SIMD_ATTR_REQUIRED";
static constexpr int64_t kA5VectorLengthBits = 256 * 8;

static std::string typeToString(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

enum class TemplateArgRole {
  Tile,
  Scalar,
};

enum class Level3TemplateFamily {
  NonScalar,
  ScalarAbi,
};

static bool isBinaryFloatCore(Operation *op) {
  return isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
             arith::MaximumFOp, arith::MinimumFOp>(op);
}

static bool isBinaryIntCore(Operation *op) {
  return isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::DivSIOp,
             arith::DivUIOp>(op);
}

static bool isBinaryCore(Operation *op) {
  return isBinaryFloatCore(op) || isBinaryIntCore(op);
}

static bool isVectorFloatBinaryArith(Operation *op) {
  if (!isBinaryFloatCore(op))
    return false;
  if (op->getNumResults() != 1)
    return false;
  return isa<VectorType>(op->getResult(0).getType());
}

static bool opRequiresExplicitExecMode(StringRef kind, StringRef op) {
  if (kind == kOpLibKindL3BinaryTemplate ||
      kind == "l3_float_binary_elementwise_template") {
    return op == "tadd" || op == "tsub" || op == "tmul" || op == "tdiv" ||
           op == "tmax" || op == "tmin";
  }
  if (kind == "l3_float_unary_template")
    return op == "trecip" || op == "trelu";
  if (kind == "l3_float_tile_scalar_template")
    return op == "tdivs";
  return false;
}

static FailureOr<int64_t> getFixedVectorLanes(Type ty) {
  auto vecTy = dyn_cast<VectorType>(ty);
  if (!vecTy || vecTy.isScalable())
    return failure();
  return vecTy.getNumElements();
}

static std::optional<int64_t> getExpectedA5SimdLanesForElemType(Type elemTy) {
  unsigned bitWidth = 0;
  if (elemTy.isF16() || elemTy.isBF16())
    bitWidth = 16;
  else if (elemTy.isF32())
    bitWidth = 32;
  else if (auto intTy = dyn_cast<IntegerType>(elemTy)) {
    if (intTy.getWidth() == 1)
      return std::nullopt;
    bitWidth = intTy.getWidth();
  } else {
    return std::nullopt;
  }

  if (bitWidth == 0 || (kA5VectorLengthBits % bitWidth) != 0)
    return std::nullopt;
  return kA5VectorLengthBits / bitWidth;
}

static bool isSupportedA5MaskLanes(int64_t lanes) {
  return lanes == 32 || lanes == 64 || lanes == 128 || lanes == 256;
}

static bool isA5ByteLaneCarrierVector(Type elemTy, int64_t lanes,
                                      int64_t simdLanes) {
  return elemTy.isInteger(8) && simdLanes > 0 && lanes == simdLanes;
}

static std::optional<int64_t>
inferExpectedLevel3SimdLanes(FunctionType fnTy,
                             ArrayRef<TemplateArgRole> argRoles) {
  auto infer = [&](bool skipI8Tiles) -> std::optional<int64_t> {
    for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
      if (role != TemplateArgRole::Tile)
        continue;
      auto tileTy = dyn_cast<pto::TileBufType>(inTy);
      if (!tileTy)
        continue;
      Type elemTy = tileTy.getElementType();
      if (skipI8Tiles && elemTy.isInteger(8))
        continue;
      if (auto expected = getExpectedA5SimdLanesForElemType(elemTy))
        return expected;
    }
    return std::nullopt;
  };

  if (auto expected = infer(/*skipI8Tiles=*/true))
    return expected;

  for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
    if (role != TemplateArgRole::Tile)
      continue;
    auto tileTy = dyn_cast<pto::TileBufType>(inTy);
    if (!tileTy)
      continue;
    if (auto expected =
            getExpectedA5SimdLanesForElemType(tileTy.getElementType()))
      return expected;
  }
  return std::nullopt;
}

static bool isSimdBridgeOp(Operation *op) {
  return isa<pto::SimdPredicateOp, pto::SimdLoadOp, pto::SimdStoreOp,
             pto::SimdLoadPUOp, pto::SimdStorePUOp, pto::SimdStorePredicateOp>(
      op);
}

static bool isPackedCmpSimdBridgeOp(Operation *op) {
  return isa<pto::SimdStorePredicateOp>(op);
}

static bool hasSimdBridgeOps(func::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isSimdBridgeOp(op)) {
      found = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return found;
}

static bool hasOnlyPackedCmpSimdBridgeOps(func::FuncOp func) {
  bool sawPackedCmp = false;
  bool sawOtherBridge = false;
  func.walk([&](Operation *op) {
    if (!isSimdBridgeOp(op))
      return WalkResult::advance();
    if (isPackedCmpSimdBridgeOp(op)) {
      sawPackedCmp = true;
      return WalkResult::advance();
    }
    sawOtherBridge = true;
    return WalkResult::interrupt();
  });
  return sawPackedCmp && !sawOtherBridge;
}

static bool isAllowedTemplateBodyOp(Operation *op) {
  if (isa<func::ReturnOp>(op))
    return true;
  if (isa<pto::SimdVecScopeOp>(op))
    return true;
  if (isa<pto::SimdTileToMemrefOp>(op))
    return true;
  if (isa<pto::SimdReductionOp>(op))
    return true;
  if (isSimdBridgeOp(op))
    return true;
  if (isa<UnrealizedConversionCastOp>(op))
    return false;
  if (auto load = dyn_cast<memref::LoadOp>(op))
    return !isa<VectorType>(load.getType());
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return !isa<VectorType>(store.getValueToStore().getType());

  StringRef ns = op->getName().getDialectNamespace();
  if (ns == "arith" || ns == "vector" || ns == "memref" || ns == "scf")
    return true;
  if (isa<math::ExpOp, math::LogOp, math::SqrtOp, math::RsqrtOp>(op))
    return true;
  return false;
}

static LogicalResult emitCodeError(Operation *op, StringRef code,
                                   const Twine &msg) {
  op->emitError() << code << ": " << msg;
  return failure();
}

static LogicalResult emitCodeError(Location loc, StringRef code,
                                   const Twine &msg) {
  emitError(loc) << code << ": " << msg;
  return failure();
}

static bool isOplibScalarType(Type ty) {
  return isa<FloatType, IntegerType, IndexType>(ty);
}

static bool isOplibTileElementType(Type ty) {
  return isa<FloatType, IntegerType>(ty);
}

static bool isOplibTileType(Type ty) {
  auto tileTy = dyn_cast<pto::TileBufType>(ty);
  return tileTy && tileTy.getRank() == 2 &&
         isOplibTileElementType(tileTy.getElementType());
}

static bool isAllowedLayoutName(StringRef layoutName) {
  return layoutName == "row_major" || layoutName == "col_major" ||
         layoutName == "none_box" || layoutName == "any";
}

static bool isAllowedCmpModeName(StringRef cmpMode) {
  return cmpMode == "EQ" || cmpMode == "NE" || cmpMode == "LT" ||
         cmpMode == "LE" || cmpMode == "GT" || cmpMode == "GE";
}

static std::optional<Level3TemplateFamily>
classifyLevel3TemplateFamily(StringRef kind) {
  using Family = Level3TemplateFamily;
  return llvm::StringSwitch<std::optional<Family>>(kind)
      .Case(kOpLibKindL3BinaryTemplate, Family::NonScalar)
      .Case("l3_float_binary_elementwise_template", Family::NonScalar)
      .Case("l3_float_partial_binary_template", Family::NonScalar)
      .Case("l3_float_binary_special_template", Family::NonScalar)
      .Case("l3_float_ternary_tile_template", Family::NonScalar)
      .Case("l3_float_unary_template", Family::NonScalar)
      .Case("l3_float_unary_math_template", Family::NonScalar)
      .Case("l3_reduce_row_template", Family::NonScalar)
      .Case("l3_reduce_col_template", Family::NonScalar)
      .Case("l3_reduce_colsum_template", Family::NonScalar)
      .Case("l3_broadcast_row_template", Family::NonScalar)
      .Case("l3_broadcast_col_template", Family::NonScalar)
      .Case("l3_broadcast_row_binary_template", Family::NonScalar)
      .Case("l3_cmp_tile_tile_template", Family::NonScalar)
      .Case("l3_select_mask_template", Family::NonScalar)
      .Case("l3_int_binary_elementwise_template", Family::NonScalar)
      .Case("l3_int_unary_template", Family::NonScalar)
      .Case("l3_float_tile_scalar_template", Family::ScalarAbi)
      .Case("l3_float_ternary_tile_scalar_template", Family::ScalarAbi)
      .Case("l3_float_unary_scalar_template", Family::ScalarAbi)
      .Case("l3_scalar_expand_template", Family::ScalarAbi)
      .Case("l3_cmp_tile_scalar_template", Family::ScalarAbi)
      .Case("l3_select_scalar_template", Family::ScalarAbi)
      .Case("l3_int_tile_scalar_elementwise_template", Family::ScalarAbi)
      .Default(std::nullopt);
}

static bool isLevel3TemplateKind(StringRef kind) {
  return classifyLevel3TemplateFamily(kind).has_value();
}

static FailureOr<SmallVector<TemplateArgRole, 4>>
getTemplateArgRolesForKind(StringRef kind) {
  auto build = [&](std::initializer_list<TemplateArgRole> roles)
      -> FailureOr<SmallVector<TemplateArgRole, 4>> {
    return SmallVector<TemplateArgRole, 4>(roles);
  };

  if (kind == kOpLibKindL3BinaryTemplate)
    return build(
        {TemplateArgRole::Tile, TemplateArgRole::Tile, TemplateArgRole::Tile});

  return llvm::StringSwitch<FailureOr<SmallVector<TemplateArgRole, 4>>>(kind)
      .Case("l3_float_binary_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_partial_binary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_binary_special_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_float_tile_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_float_ternary_tile_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_ternary_tile_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_math_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_float_unary_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_reduce_row_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_reduce_col_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_reduce_colsum_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_broadcast_row_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_broadcast_col_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_broadcast_row_binary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_scalar_expand_template",
            build({TemplateArgRole::Scalar, TemplateArgRole::Tile}))
      .Case("l3_cmp_tile_tile_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_cmp_tile_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_select_mask_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Case("l3_select_scalar_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Scalar, TemplateArgRole::Tile}))
      .Case("l3_int_binary_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile,
                   TemplateArgRole::Tile}))
      .Case("l3_int_tile_scalar_elementwise_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Scalar,
                   TemplateArgRole::Tile}))
      .Case("l3_int_unary_template",
            build({TemplateArgRole::Tile, TemplateArgRole::Tile}))
      .Default(failure());
}

static bool kindRequiresCmpMode(StringRef kind) {
  return kind == "l3_cmp_tile_tile_template" ||
         kind == "l3_cmp_tile_scalar_template";
}

static bool kindRequiresIsBinary(StringRef kind) {
  return kind == "l3_reduce_colsum_template";
}

static FailureOr<int64_t> parseI64Attr(Operation *op, StringRef attrName,
                                       bool allowWildcard) {
  auto attr = op->getAttrOfType<IntegerAttr>(attrName);
  if (!attr)
    return failure();
  int64_t value = attr.getInt();
  if (value == -1) {
    if (!allowWildcard)
      return failure();
    return value;
  }
  if (value <= 0)
    return failure();
  return value;
}

static FailureOr<std::string> parseStringAttr(Operation *op,
                                              StringRef attrName) {
  auto attr = op->getAttrOfType<StringAttr>(attrName);
  if (!attr)
    return failure();
  return attr.getValue().str();
}

static std::string getArgMatchAttrName(unsigned argIndex, StringRef suffix) {
  return (Twine("pto.oplib.match.arg") + Twine(argIndex) + "." + suffix).str();
}

static LogicalResult validateLegacySimdBody(func::FuncOp func, int64_t lanes) {
  if (func.isExternal() || func.empty() ||
      func.front().without_terminator().empty()) {
    return emitCodeError(func.getLoc(), kErrEmptyBody,
                         "SIMD template body must not be empty");
  }

  llvm::DenseMap<Operation *, int64_t> preorder;
  int64_t seq = 0;
  func.walk([&](Operation *op) { preorder[op] = seq++; });

  int64_t firstLoad = std::numeric_limits<int64_t>::max();
  int64_t firstStore = std::numeric_limits<int64_t>::max();
  int64_t coreSeq = -1;
  int coreCount = 0;

  func.walk([&](Operation *op) {
    if (isa<pto::SimdLoadOp, pto::SimdLoadPUOp>(op))
      firstLoad = std::min(firstLoad, preorder[op]);
    if (isa<pto::SimdStoreOp, pto::SimdStorePUOp>(op))
      firstStore = std::min(firstStore, preorder[op]);

    if (auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr)) {
      if (slotAttr.getValue() != kSimdCoreSlotBinaryEwise) {
        (void)emitCodeError(
            op, kErrCoreSlot,
            "unsupported core_slot, expected 'binary_ewise_core'");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      if (!isBinaryCore(op)) {
        (void)emitCodeError(op, kErrCoreSlot,
                            "core_slot op must be supported binary arith op");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      if (op->getNumResults() != 1) {
        (void)emitCodeError(op, kErrCoreSlot,
                            "core_slot op must have exactly one result");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      FailureOr<int64_t> vLanes =
          getFixedVectorLanes(op->getResult(0).getType());
      if (failed(vLanes) || *vLanes != lanes) {
        (void)emitCodeError(
            op, kErrLanesMismatch,
            "core_slot vector lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
        return;
      }
      ++coreCount;
      coreSeq = preorder[op];
    }

    if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
      FailureOr<int64_t> predLanes =
          getFixedVectorLanes(pred.getMask().getType());
      if (failed(predLanes) || *predLanes != lanes) {
        (void)emitCodeError(
            pred, kErrLanesMismatch,
            "simd.predicate lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
      FailureOr<int64_t> loadLanes =
          getFixedVectorLanes(load.getValue().getType());
      if (failed(loadLanes) || *loadLanes != lanes) {
        (void)emitCodeError(load, kErrLanesMismatch,
                            "simd.load lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
      FailureOr<int64_t> loadLanes =
          getFixedVectorLanes(loadPU.getValue().getType());
      if (failed(loadLanes) || *loadLanes != lanes) {
        (void)emitCodeError(loadPU, kErrLanesMismatch,
                            "simd.load_pu lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
      FailureOr<int64_t> storeLanes =
          getFixedVectorLanes(store.getValue().getType());
      if (failed(storeLanes) || *storeLanes != lanes) {
        (void)emitCodeError(store, kErrLanesMismatch,
                            "simd.store lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }

    if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
      FailureOr<int64_t> storeLanes =
          getFixedVectorLanes(storePU.getValue().getType());
      if (failed(storeLanes) || *storeLanes != lanes) {
        (void)emitCodeError(storePU, kErrLanesMismatch,
                            "simd.store_pu lanes mismatch with pto.simd.lanes");
        coreSeq = std::numeric_limits<int64_t>::min();
      }
      return;
    }
  });

  if (coreSeq == std::numeric_limits<int64_t>::min())
    return failure();

  if (coreCount != 1) {
    return emitCodeError(
        func.getLoc(), kErrCoreSlot,
        "SIMD template must contain exactly one 'binary_ewise_core' op");
  }
  if (firstLoad == std::numeric_limits<int64_t>::max()) {
    return emitCodeError(
        func.getLoc(), kErrCoreSlot,
        "SIMD template must contain at least one simd.load/simd.load_pu");
  }
  if (firstStore == std::numeric_limits<int64_t>::max()) {
    return emitCodeError(
        func.getLoc(), kErrCoreSlot,
        "SIMD template must contain at least one simd.store/simd.store_pu");
  }
  if (!(firstLoad < coreSeq && coreSeq < firstStore)) {
    return emitCodeError(
        func.getLoc(), kErrCoreSlot,
        "SIMD template ordering must satisfy load -> core -> store");
  }
  return success();
}

static LogicalResult
validateOplibSignature(func::FuncOp func, StringRef kind,
                       SmallVectorImpl<TemplateArgRole> &argRoles) {
  auto rolesOr = getTemplateArgRolesForKind(kind);
  if (failed(rolesOr)) {
    func.emitError() << "unsupported pto.oplib.kind: " << kind;
    return failure();
  }

  argRoles.assign(rolesOr->begin(), rolesOr->end());
  FunctionType fnTy = func.getFunctionType();
  if (fnTy.getNumInputs() != argRoles.size() || fnTy.getNumResults() != 0) {
    func.emitError() << "invalid OP-Lib signature for kind=" << kind;
    return failure();
  }

  for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
    if (role == TemplateArgRole::Tile) {
      if (!isOplibTileType(inTy)) {
        func.emitError() << "invalid OP-Lib signature for kind=" << kind;
        return failure();
      }
      continue;
    }
    if (!isOplibScalarType(inTy)) {
      func.emitError() << "invalid OP-Lib signature for kind=" << kind;
      return failure();
    }
  }
  return success();
}

static LogicalResult
validateOplibMatcherAttrs(func::FuncOp func, StringRef kind,
                          ArrayRef<TemplateArgRole> argRoles) {
  Operation *op = func.getOperation();

  auto entryRoleAttr = op->getAttrOfType<StringAttr>(kOpLibAttrEntryRole);
  if (!entryRoleAttr) {
    func.emitError("missing required attr: pto.oplib.entry_role");
    return failure();
  }
  StringRef entryRole = entryRoleAttr.getValue();
  if (entryRole != kEntryRoleVariant && entryRole != kEntryRoleSeed) {
    func.emitError("invalid pto.oplib.entry_role, expected variant|seed");
    return failure();
  }

  if (entryRole == kEntryRoleVariant) {
    if (failed(parseStringAttr(op, "pto.oplib.op")) ||
        failed(parseStringAttr(op, "pto.oplib.variant_id")) ||
        failed(parseStringAttr(op, "pto.oplib.match.dtype"))) {
      func.emitError("variant entry missing attrs: pto.oplib.op / "
                     "pto.oplib.variant_id / pto.oplib.match.dtype");
      return failure();
    }
  }

  if (kind == kOpLibKindL3BinaryTemplate) {
    auto rowsOr = parseI64Attr(op, kOpLibAttrMatchRows, /*allowWildcard=*/true);
    auto colsOr = parseI64Attr(op, kOpLibAttrMatchCols, /*allowWildcard=*/true);
    auto fractalOr =
        parseI64Attr(op, kOpLibAttrMatchFractal, /*allowWildcard=*/true);
    auto blayoutOr = parseStringAttr(op, kOpLibAttrMatchBLayout);
    auto slayoutOr = parseStringAttr(op, kOpLibAttrMatchSLayout);
    if (failed(rowsOr) || failed(colsOr) || failed(fractalOr) ||
        failed(blayoutOr) || failed(slayoutOr)) {
      func.emitError("missing or invalid match rows/cols/fractal attrs");
      return failure();
    }
    if (!isAllowedLayoutName(*blayoutOr) || !isAllowedLayoutName(*slayoutOr)) {
      func.emitError("invalid layout value in match.blayout/match.slayout");
      return failure();
    }
  } else {
    for (unsigned i = 0; i < argRoles.size(); ++i) {
      bool hasAnyTileAttr = op->hasAttr(getArgMatchAttrName(i, "rows")) ||
                            op->hasAttr(getArgMatchAttrName(i, "cols")) ||
                            op->hasAttr(getArgMatchAttrName(i, "blayout")) ||
                            op->hasAttr(getArgMatchAttrName(i, "slayout")) ||
                            op->hasAttr(getArgMatchAttrName(i, "fractal"));

      if (argRoles[i] == TemplateArgRole::Scalar) {
        if (hasAnyTileAttr) {
          func.emitError() << "arg" << i << " is scalar in kind=" << kind
                           << ", but carries tile match attrs";
          return failure();
        }
        continue;
      }

      auto rowsOr = parseI64Attr(op, getArgMatchAttrName(i, "rows"),
                                 /*allowWildcard=*/true);
      auto colsOr = parseI64Attr(op, getArgMatchAttrName(i, "cols"),
                                 /*allowWildcard=*/true);
      auto fractalOr = parseI64Attr(op, getArgMatchAttrName(i, "fractal"),
                                    /*allowWildcard=*/true);
      auto blayoutOr = parseStringAttr(op, getArgMatchAttrName(i, "blayout"));
      auto slayoutOr = parseStringAttr(op, getArgMatchAttrName(i, "slayout"));
      if (failed(rowsOr) || failed(colsOr) || failed(fractalOr) ||
          failed(blayoutOr) || failed(slayoutOr)) {
        func.emitError() << "missing or invalid arg" << i
                         << " match attrs: rows/cols/blayout/slayout/fractal";
        return failure();
      }
      if (!isAllowedLayoutName(*blayoutOr) ||
          !isAllowedLayoutName(*slayoutOr)) {
        func.emitError() << "invalid layout value in arg" << i
                         << ".blayout/.slayout";
        return failure();
      }
    }
  }

  bool hasScalarArg = llvm::any_of(argRoles, [](TemplateArgRole role) {
    return role == TemplateArgRole::Scalar;
  });
  if (hasScalarArg) {
    auto scalarPosAttr =
        op->getAttrOfType<IntegerAttr>(kOpLibAttrMatchScalarPos);
    if (!scalarPosAttr) {
      func.emitError("missing required attr: pto.oplib.match.scalar_pos");
      return failure();
    }
    int64_t scalarPos = scalarPosAttr.getInt();
    if (scalarPos < 0 || scalarPos >= static_cast<int64_t>(argRoles.size()) ||
        argRoles[scalarPos] != TemplateArgRole::Scalar) {
      func.emitError("invalid pto.oplib.match.scalar_pos");
      return failure();
    }
  } else if (op->hasAttr(kOpLibAttrMatchScalarPos)) {
    func.emitError("invalid pto.oplib.match.scalar_pos");
    return failure();
  }

  if (kindRequiresCmpMode(kind)) {
    auto cmpModeOr = parseStringAttr(op, kOpLibAttrMatchCmpMode);
    if (failed(cmpModeOr)) {
      func.emitError("missing required attr: pto.oplib.match.cmp_mode");
      return failure();
    }
    if (!isAllowedCmpModeName(*cmpModeOr)) {
      func.emitError("invalid pto.oplib.match.cmp_mode");
      return failure();
    }
  } else if (op->hasAttr(kOpLibAttrMatchCmpMode)) {
    auto cmpModeOr = parseStringAttr(op, kOpLibAttrMatchCmpMode);
    if (failed(cmpModeOr) || !isAllowedCmpModeName(*cmpModeOr)) {
      func.emitError("invalid pto.oplib.match.cmp_mode");
      return failure();
    }
  }

  if (kindRequiresIsBinary(kind)) {
    if (!op->getAttrOfType<BoolAttr>(kOpLibAttrMatchIsBinary)) {
      func.emitError("missing required attr: pto.oplib.match.is_binary");
      return failure();
    }
  }

  return success();
}

static LogicalResult validateOplibBody(func::FuncOp func, StringRef kind,
                                       ArrayRef<TemplateArgRole> argRoles,
                                       bool hasSimdBridge) {
  if (func.isExternal() || func.empty() ||
      func.front().without_terminator().empty()) {
    return emitCodeError(func.getLoc(), kErrEmptyBody,
                         "template body must not be empty");
  }

  StringRef oplibOp;
  if (auto opAttr = func->getAttrOfType<StringAttr>(kOpLibAttrOp))
    oplibOp = opAttr.getValue();
  bool requireExplicitExecMode = opRequiresExplicitExecMode(kind, oplibOp);
  bool hasPackedCmpBridgeOnly = hasOnlyPackedCmpSimdBridgeOps(func);

  FunctionType fnTy = func.getFunctionType();
  for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
    if (role == TemplateArgRole::Scalar) {
      if (!isOplibScalarType(inTy)) {
        return emitCodeError(
            func.getLoc(), kErrDType,
            "template scalar inputs must use builtin scalar types");
      }
      continue;
    }

    auto tileTy = dyn_cast<pto::TileBufType>(inTy);
    if (!tileTy) {
      return emitCodeError(func.getLoc(), kErrEmptyBody,
                           "template tile inputs must be !pto.tile_buf");
    }
    if (!isOplibTileElementType(tileTy.getElementType())) {
      return emitCodeError(
          func.getLoc(), kErrDType,
          "SIMD template supports float/integer tile inputs only");
    }
    if (tileTy.getBLayoutValueI32() != 0) {
      return emitCodeError(func.getLoc(), kErrLayout,
                           "SIMD template supports row_major only");
    }
  }

  auto levelAttr = func->getAttrOfType<StringAttr>(kSimdLevelAttr);
  auto lanesAttr = func->getAttrOfType<IntegerAttr>(kSimdLanesAttr);
  bool hasLevelAttr = static_cast<bool>(levelAttr);
  bool hasLanesAttr = static_cast<bool>(lanesAttr);
  int64_t simdLanes = -1;

  if (hasSimdBridge || hasLevelAttr || hasLanesAttr) {
    if (!hasLevelAttr || !hasLanesAttr) {
      return emitCodeError(
          func.getOperation(), kErrSimdAttrRequired,
          "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
    }
    auto lanes = lanesAttr.getInt();
    if (lanes <= 0) {
      return emitCodeError(func.getOperation(), kErrSimdAttrRequired,
                           "using pto.simd.* requires positive pto.simd.lanes");
    }
    if (levelAttr.getValue() != kSimdLevelBinaryEwiseV1) {
      return emitCodeError(func.getOperation(), kErrCoreSlot,
                           Twine("unsupported pto.simd.level: ") +
                               levelAttr.getValue());
    }
    simdLanes = lanes;
  }

  if (isLevel3TemplateKind(kind) && simdLanes > 0 &&
      inferExpectedLevel3SimdLanes(fnTy, argRoles) &&
      simdLanes != *inferExpectedLevel3SimdLanes(fnTy, argRoles)) {
    return emitCodeError(
        func.getOperation(), kErrLanesMismatch,
        Twine("Level-3 template kind=") + kind + " requires pto.simd.lanes = " +
            Twine(*inferExpectedLevel3SimdLanes(fnTy, argRoles)) + ", got " +
            Twine(simdLanes));
  }

  llvm::DenseMap<Operation *, int64_t> preorder;
  int64_t seq = 0;
  func.walk([&](Operation *op) { preorder[op] = seq++; });

  int64_t firstLoad = std::numeric_limits<int64_t>::max();
  int64_t firstStore = std::numeric_limits<int64_t>::max();
  int64_t coreSeq = -1;
  int coreCount = 0;
  bool sawLevel3VectorValue = false;
  LogicalResult status = success();
  int64_t effectiveSimdLanes =
      simdLanes > 0 ? simdLanes
                    : inferExpectedLevel3SimdLanes(fnTy, argRoles).value_or(-1);

  auto requireLevel3VectorLanes = [&](Operation *targetOp, Type ty,
                                      StringRef position) -> bool {
    if (!isLevel3TemplateKind(kind))
      return true;
    auto vecTy = dyn_cast<VectorType>(ty);
    if (!vecTy || vecTy.isScalable())
      return true;
    auto lanes = getFixedVectorLanes(ty);
    if (failed(lanes))
      return true;
    sawLevel3VectorValue = true;
    Type elemTy = vecTy.getElementType();
    if (elemTy.isInteger(1)) {
      if (!isSupportedA5MaskLanes(*lanes)) {
        status = emitCodeError(
            targetOp, kErrLanesMismatch,
            Twine("Level-3 template kind=") + kind +
                " requires vector<i1> masks with supported A5 lane counts "
                "(32/64/128/256), but " +
                position + " of op '" + targetOp->getName().getStringRef() +
                "' uses " + Twine(*lanes) + " lanes");
        return false;
      }
    } else {
      auto expectedLanes = getExpectedA5SimdLanesForElemType(elemTy);
      if (!expectedLanes) {
        status = emitCodeError(targetOp, kErrDType,
                               Twine("Level-3 template kind=") + kind +
                                   " uses unsupported A5 SIMD element type " +
                                   typeToString(elemTy));
        return false;
      }
      if (*lanes != *expectedLanes &&
          !isA5ByteLaneCarrierVector(elemTy, *lanes, effectiveSimdLanes)) {
        status = emitCodeError(targetOp, kErrLanesMismatch,
                               Twine("Level-3 template kind=") + kind +
                                   " requires " + typeToString(elemTy) +
                                   " vectors to use " + Twine(*expectedLanes) +
                                   " lanes, but " + position + " of op '" +
                                   targetOp->getName().getStringRef() +
                                   "' uses " + Twine(*lanes) + " lanes");
        return false;
      }
    }

    if (effectiveSimdLanes > 0 && *lanes != effectiveSimdLanes) {
      status = emitCodeError(
          targetOp, kErrLanesMismatch,
          Twine("Level-3 template kind=") + kind +
              " requires vector lanes to match pto.simd.lanes = " +
              Twine(effectiveSimdLanes) + ", but " + position + " of op '" +
              targetOp->getName().getStringRef() + "' uses " + Twine(*lanes) +
              " lanes");
      return false;
    }
    return true;
  };

  auto requireNonEmptyTokenAttr = [&](Operation *targetOp, StringRef attrName,
                                      StringRef usage,
                                      StringRef requiredPrefix) -> bool {
    auto tokenAttr = targetOp->getAttrOfType<StringAttr>(attrName);
    if (!tokenAttr) {
      status = emitCodeError(targetOp, kErrSimdAttrRequired,
                             Twine(usage) + " requires string attr '" +
                                 attrName + "'");
      return false;
    }
    StringRef token = tokenAttr.getValue();
    if (token.empty()) {
      status = emitCodeError(targetOp, kErrSimdAttrRequired,
                             Twine(usage) + " attr '" + attrName +
                                 "' must be non-empty");
      return false;
    }
    if (!requiredPrefix.empty() && !token.starts_with(requiredPrefix)) {
      status = emitCodeError(targetOp, kErrSimdAttrRequired,
                             Twine(usage) + " attr '" + attrName +
                                 "' must start with '" + requiredPrefix +
                                 "', got '" + token + "'");
      return false;
    }
    return true;
  };

  func.walk([&](Operation *op) {
    if (failed(status))
      return;
    if (op == func.getOperation())
      return;

    for (Type resultTy : op->getResultTypes()) {
      if (!requireLevel3VectorLanes(op, resultTy, "result"))
        return;
    }
    for (Value operand : op->getOperands()) {
      if (!requireLevel3VectorLanes(op, operand.getType(), "operand"))
        return;
    }

    if (!isAllowedTemplateBodyOp(op)) {
      status = emitCodeError(op, kErrBodyDisallowedIR,
                             Twine("unsupported template body op: ") +
                                 op->getName().getStringRef());
      return;
    }

    if (isa<vector::LoadOp, vector::MaskedLoadOp>(op)) {
      if (!requireNonEmptyTokenAttr(op, kSimdVldDistAttr, "vector load",
                                    /*requiredPrefix=*/""))
        return;
    }

    if (isa<vector::StoreOp, vector::MaskedStoreOp>(op)) {
      if (!requireNonEmptyTokenAttr(op, kSimdVstDistAttr, "vector store",
                                    /*requiredPrefix=*/"DIST_"))
        return;
    }

    if (isVectorFloatBinaryArith(op) &&
        (requireExplicitExecMode || op->hasAttr(kSimdExecModeAttr))) {
      if (!requireNonEmptyTokenAttr(op, kSimdExecModeAttr,
                                    "vector float binary arith op",
                                    /*requiredPrefix=*/"MODE_"))
        return;
    }

    if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
      if (simdLanes <= 0) {
        status = emitCodeError(
            pred, kErrSimdAttrRequired,
            "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
        return;
      }
      auto lanes = getFixedVectorLanes(pred.getMask().getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status =
            emitCodeError(pred, kErrLanesMismatch,
                          "simd.predicate lanes mismatch with pto.simd.lanes");
      }
      return;
    }

    if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
      firstLoad = std::min(firstLoad, preorder[op]);
      if (simdLanes <= 0) {
        status = emitCodeError(
            load, kErrSimdAttrRequired,
            "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
        return;
      }
      auto lanes = getFixedVectorLanes(load.getValue().getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status = emitCodeError(load, kErrLanesMismatch,
                               "simd.load lanes mismatch with pto.simd.lanes");
      }
      return;
    }

    if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
      firstLoad = std::min(firstLoad, preorder[op]);
      if (simdLanes <= 0) {
        status = emitCodeError(
            loadPU, kErrSimdAttrRequired,
            "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
        return;
      }
      auto lanes = getFixedVectorLanes(loadPU.getValue().getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status =
            emitCodeError(loadPU, kErrLanesMismatch,
                          "simd.load_pu lanes mismatch with pto.simd.lanes");
      }
      return;
    }

    if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
      firstStore = std::min(firstStore, preorder[op]);
      if (simdLanes <= 0) {
        status = emitCodeError(
            store, kErrSimdAttrRequired,
            "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
        return;
      }
      auto lanes = getFixedVectorLanes(store.getValue().getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status = emitCodeError(store, kErrLanesMismatch,
                               "simd.store lanes mismatch with pto.simd.lanes");
      }
      return;
    }

    if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
      firstStore = std::min(firstStore, preorder[op]);
      if (simdLanes <= 0) {
        status = emitCodeError(
            storePU, kErrSimdAttrRequired,
            "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
        return;
      }
      auto lanes = getFixedVectorLanes(storePU.getValue().getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status =
            emitCodeError(storePU, kErrLanesMismatch,
                          "simd.store_pu lanes mismatch with pto.simd.lanes");
      }
      return;
    }

    auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr);
    if (!slotAttr)
      return;

    if (slotAttr.getValue() != kSimdCoreSlotBinaryEwise) {
      status =
          emitCodeError(op, kErrCoreSlot,
                        "unsupported core_slot, expected 'binary_ewise_core'");
      return;
    }
    if (!isBinaryCore(op)) {
      status = emitCodeError(op, kErrCoreSlot,
                             "core_slot op must be one of "
                             "arith.addf/subf/mulf/divf/maximumf/minimumf/addi/"
                             "subi/muli/divsi/divui");
      return;
    }
    if (op->getNumResults() != 1) {
      status = emitCodeError(op, kErrCoreSlot,
                             "core_slot op must have exactly one result");
      return;
    }
    if (simdLanes > 0) {
      auto lanes = getFixedVectorLanes(op->getResult(0).getType());
      if (failed(lanes) || *lanes != simdLanes) {
        status = emitCodeError(
            op, kErrLanesMismatch,
            "core_slot vector lanes mismatch with pto.simd.lanes");
        return;
      }
    }
    coreSeq = preorder[op];
    ++coreCount;
  });

  if (failed(status))
    return failure();

  if (isLevel3TemplateKind(kind) && !sawLevel3VectorValue) {
    return emitCodeError(func.getOperation(), kErrLanesMismatch,
                         Twine("Level-3 template kind=") + kind +
                             " must materialize a supported A5 SIMD body");
  }

  if (coreCount > 1) {
    return emitCodeError(func.getLoc(), kErrCoreSlot,
                         "template must not contain multiple core slot ops");
  }

  if (hasSimdBridge && !hasPackedCmpBridgeOnly) {
    if (coreCount != 1) {
      return emitCodeError(
          func.getLoc(), kErrCoreSlot,
          "template using pto.simd.* must contain exactly one core slot op");
    }
    if (firstLoad == std::numeric_limits<int64_t>::max()) {
      return emitCodeError(func.getLoc(), kErrCoreSlot,
                           "template must contain simd.load/simd.load_pu");
    }
    if (firstStore == std::numeric_limits<int64_t>::max()) {
      return emitCodeError(func.getLoc(), kErrCoreSlot,
                           "template must contain simd.store/simd.store_pu");
    }
    if (!(firstLoad < coreSeq && coreSeq < firstStore)) {
      return emitCodeError(
          func.getLoc(), kErrCoreSlot,
          "template ordering must satisfy load -> core -> store");
    }
  }

  return success();
}

struct PTOValidateSimdIRPass
    : public pto::impl::PTOValidateSimdIRBase<PTOValidateSimdIRPass> {
  using pto::impl::PTOValidateSimdIRBase<
      PTOValidateSimdIRPass>::PTOValidateSimdIRBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      bool hasBridge = hasSimdBridgeOps(func);
      auto kindAttr = func->getAttrOfType<StringAttr>(kOpLibAttrKind);
      if (!hasBridge && !kindAttr)
        continue;

      if (!kindAttr) {
        auto levelAttr = func->getAttrOfType<StringAttr>(kSimdLevelAttr);
        if (!levelAttr) {
          func.emitError() << kErrSimdAttrRequired
                           << ": using pto.simd.* requires attr pto.simd.level";
          signalPassFailure();
          return;
        }

        if (levelAttr.getValue() != kSimdLevelBinaryEwiseV1) {
          func.emitError() << kErrCoreSlot
                           << ": unsupported pto.simd.level, expected '"
                           << kSimdLevelBinaryEwiseV1 << "'";
          signalPassFailure();
          return;
        }

        auto lanesAttr = func->getAttrOfType<IntegerAttr>(kSimdLanesAttr);
        if (!lanesAttr || lanesAttr.getInt() <= 0) {
          func.emitError()
              << kErrSimdAttrRequired
              << ": using pto.simd.* requires positive pto.simd.lanes";
          signalPassFailure();
          return;
        }
        if (failed(validateLegacySimdBody(func, lanesAttr.getInt()))) {
          signalPassFailure();
          return;
        }
        continue;
      }

      SmallVector<TemplateArgRole, 4> argRoles;
      StringRef kind = kindAttr.getValue();
      if (failed(validateOplibSignature(func, kind, argRoles)) ||
          failed(validateOplibMatcherAttrs(func, kind, argRoles)) ||
          failed(validateOplibBody(func, kind, argRoles, hasBridge))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOValidateSimdIRPass() {
  return std::make_unique<PTOValidateSimdIRPass>();
}
