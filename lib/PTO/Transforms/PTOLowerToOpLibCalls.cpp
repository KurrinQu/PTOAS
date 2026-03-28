#include "PTO/IR/PTO.h"
#include "PTO/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace mlir {
namespace pto {
#define GEN_PASS_DEF_PTOINSTANTIATEANDLOWERTOLIBCALL
#include "PTO/Transforms/Passes.h.inc"
} // namespace pto
} // namespace mlir

using namespace mlir;

namespace {
static constexpr llvm::StringLiteral kA5OpLibManifestRelPath =
    "oplib/level3/families/a5_oplib_v1_manifest.yaml";

static constexpr llvm::StringLiteral kFusionGroupIdAttr = "pto.fusion.group_id";
static constexpr llvm::StringLiteral kFusionOrderAttr = "pto.fusion.order";

static constexpr llvm::StringLiteral kOpLibAttrKind = "pto.oplib.kind";
static constexpr llvm::StringLiteral kOpLibAttrEntryRole =
    "pto.oplib.entry_role";
static constexpr llvm::StringLiteral kOpLibAttrOp = "pto.oplib.op";
static constexpr llvm::StringLiteral kOpLibAttrVariantId =
    "pto.oplib.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrMatchDType =
    "pto.oplib.match.dtype";
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
static constexpr llvm::StringLiteral kCanonicalByteMaskContract =
    "byte_mask_canonical_v1";
static constexpr llvm::StringLiteral kOpLibAttrCost = "pto.oplib.cost";
static constexpr llvm::StringLiteral kOpLibAttrPriority = "pto.oplib.priority";
static constexpr llvm::StringLiteral kOpLibAttrSync = "pto.oplib.sync";

static constexpr llvm::StringLiteral kOpLibAttrSeedId = "pto.oplib.seed_id";
static constexpr llvm::StringLiteral kOpLibAttrSeedDType =
    "pto.oplib.seed_dtype";
static constexpr llvm::StringLiteral kOpLibAttrSeedSupportDTypes =
    "pto.oplib.seed.support_dtypes";
static constexpr llvm::StringLiteral kOpLibAttrSeedSupportOps =
    "pto.oplib.seed.support_ops";
static constexpr llvm::StringLiteral kOpLibAttrSeedCoreSlot =
    "pto.oplib.seed.core_slot";

static constexpr llvm::StringLiteral kOpLibAttrInstVariantId =
    "pto.oplib.instance.variant_id";
static constexpr llvm::StringLiteral kOpLibAttrInstKind =
    "pto.oplib.instance.kind";
static constexpr llvm::StringLiteral kOpLibAttrInstOp = "pto.oplib.instance.op";
static constexpr llvm::StringLiteral kOpLibAttrInstDType =
    "pto.oplib.instance.dtype";
static constexpr llvm::StringLiteral kOpLibAttrInstSource =
    "pto.oplib.instance.source";
static constexpr llvm::StringLiteral kOpLibAttrInstFromSeed =
    "pto.oplib.instance.from_seed";
static constexpr llvm::StringLiteral kOpLibAttrInstSeedId =
    "pto.oplib.instance.seed_id";
static constexpr llvm::StringLiteral kOpLibAttrInstCoreSlot =
    "pto.oplib.instance.core_slot";

static constexpr llvm::StringLiteral kOpLibKindL3BinaryTemplate =
    "l3_binary_elementwise_template";

static constexpr llvm::StringLiteral kEntryRoleVariant = "variant";
static constexpr llvm::StringLiteral kEntryRoleSeed = "seed";

static constexpr llvm::StringLiteral kDefaultAny = "any";
static constexpr llvm::StringLiteral kDefaultCoreSlot = "binary_ewise_core";

static constexpr llvm::StringLiteral kSimdLevelAttr = "pto.simd.level";
static constexpr llvm::StringLiteral kSimdLanesAttr = "pto.simd.lanes";
static constexpr llvm::StringLiteral kSimdCoreSlotAttr = "pto.simd.core_slot";
static constexpr llvm::StringLiteral kSimdLevelBinaryEwiseV1 =
    "binary_ewise_v1";
static constexpr llvm::StringLiteral kSimdVldDistAttr = "pto.simd.vld_dist";
static constexpr llvm::StringLiteral kSimdVstDistAttr = "pto.simd.vst_dist";
static constexpr llvm::StringLiteral kSimdExecModeAttr = "pto.simd.exec_mode";

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

struct GroupInfo {
  int64_t groupId = -1;
  pto::FusionRegionOp region;
  SmallVector<Operation *, 8> ops;
};

enum class A5ManifestStatus {
  Implemented,
  Deferred,
};

enum class A5ManifestSemanticClass {
  NativeA5Impl,
  PublicApiRewrite,
  MissingAcceptedSemantics,
};

struct A5ManifestEntry {
  std::string family;
  A5ManifestStatus status = A5ManifestStatus::Deferred;
  A5ManifestSemanticClass semanticClass =
      A5ManifestSemanticClass::MissingAcceptedSemantics;
  std::string deferredReason;
};

struct A5OpLibManifest {
  llvm::StringMap<A5ManifestEntry> operators;

  const A5ManifestEntry *lookup(StringRef opName) const {
    auto it = operators.find(opName);
    if (it == operators.end())
      return nullptr;
    return &it->second;
  }
};

enum class EntryRole {
  Variant,
  Seed,
};

enum class TemplateArgRole {
  Tile,
  Scalar,
};

enum class Level3TemplateFamily {
  NonScalar,
  ScalarAbi,
};

struct MatchKey {
  int64_t rows = -1;
  int64_t cols = -1;
  std::string blayout = kDefaultAny.str();
  std::string slayout = kDefaultAny.str();
  int64_t fractal = -1;
};

struct TemplateEntry {
  func::FuncOp symbol;
  std::string kind;
  EntryRole role = EntryRole::Variant;
  std::string op;
  std::string variantId;
  std::string matchDType;

  std::string seedId;
  std::string seedDType;
  SmallVector<std::string, 8> supportDTypes;
  SmallVector<std::string, 8> supportOps;
  std::string coreSlot = kDefaultCoreSlot.str();
  std::string simdLevel;
  int64_t simdLanes = -1;
  bool hasSimdBridgeOps = false;

  SmallVector<TemplateArgRole, 4> argRoles;
  SmallVector<std::optional<MatchKey>, 4> argMatches;
  std::optional<int64_t> scalarPos;
  std::optional<std::string> cmpMode;
  std::optional<bool> isBinary;
  int64_t cost = 0;
  int64_t priority = 0;
  bool sync = false;
};

struct SelectedVariant {
  TemplateEntry *entry = nullptr;
  std::string kind;
  std::string variantId;
  std::string op;
  std::string dtype;
  bool fromSeed = false;
  std::string seedId;
  std::string coreSlot;
  SmallVector<std::string, 4> instanceKeyAttrs;
  int64_t cost = 0;
  int64_t priority = 0;
};

struct PlannedOpLowering {
  Operation *op = nullptr;
  SelectedVariant selected;
  SmallVector<Value, 4> operands;
  func::FuncOp instance;
};

struct MatchRequest {
  std::string kind;
  std::string op;
  std::string dtype;
  SmallVector<Type, 4> argTypes;
  SmallVector<Value, 4> operands;
  SmallVector<std::optional<MatchKey>, 4> argMatches;
  std::optional<std::string> maskContract;
  std::optional<std::string> operandOrder;
  std::optional<int64_t> fullTilePos;
  std::optional<int64_t> rowBroadcastPos;
  std::optional<int64_t> scalarPos;
  std::optional<std::string> cmpMode;
  std::optional<bool> isBinary;
  std::optional<std::string> requiredVariantId;
};

static StringRef getPTOMnemonic(Operation *op) {
  StringRef fullName = op->getName().getStringRef();
  size_t dot = fullName.rfind('.');
  if (dot == StringRef::npos)
    return fullName;
  return fullName.drop_front(dot + 1);
}

static std::optional<StringRef>
getApprovedPublicApiRewriteTarget(StringRef opName) {
  if (opName == "trecip")
    return StringRef("TDIVS");
  return std::nullopt;
}

static StringRef
stringifyManifestSemanticClass(A5ManifestSemanticClass semanticClass) {
  switch (semanticClass) {
  case A5ManifestSemanticClass::NativeA5Impl:
    return "native_a5_impl";
  case A5ManifestSemanticClass::PublicApiRewrite:
    return "public_api_rewrite";
  case A5ManifestSemanticClass::MissingAcceptedSemantics:
    return "missing_accepted_semantics";
  }
  llvm_unreachable("unsupported A5 manifest semantic class");
}

static std::string formatManifestEntrySummary(StringRef opName,
                                              const A5ManifestEntry &entry) {
  std::string summary =
      (Twine("op='") + opName + "' family=" + entry.family +
       " classification=" + stringifyManifestSemanticClass(entry.semanticClass))
          .str();
  if (entry.status == A5ManifestStatus::Deferred &&
      !entry.deferredReason.empty())
    summary = (summary + " deferred_reason=\"" + entry.deferredReason + "\"");
  return summary;
}

static llvm::Expected<A5OpLibManifest> loadA5OpLibManifest(StringRef path) {
  auto bufferOrErr = llvm::MemoryBuffer::getFile(path);
  if (!bufferOrErr) {
    return llvm::createStringError(
        std::errc::no_such_file_or_directory,
        "failed to read A5 OpLib V1 manifest '%s': %s", path.data(),
        bufferOrErr.getError().message().c_str());
  }

  llvm::Expected<llvm::json::Value> rootOrErr =
      llvm::json::parse((*bufferOrErr)->getBuffer());
  if (!rootOrErr) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "failed to parse A5 OpLib V1 manifest '%s': %s", path.data(),
        llvm::toString(rootOrErr.takeError()).c_str());
  }

  auto *root = rootOrErr->getAsObject();
  if (!root) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "A5 OpLib V1 manifest '%s' root must be an object", path.data());
  }

  auto schema = root->getString("schema_version");
  if (!schema || *schema != "a5_oplib_v1_manifest/v1") {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "A5 OpLib V1 manifest '%s' has unexpected schema_version '%s'",
        path.data(), schema ? schema->data() : "<empty>");
  }

  auto *operators = root->getArray("operators");
  if (!operators) {
    return llvm::createStringError(
        std::errc::invalid_argument,
        "A5 OpLib V1 manifest '%s' is missing operators array", path.data());
  }

  A5OpLibManifest manifest;
  for (llvm::json::Value &value : *operators) {
    auto *entryObj = value.getAsObject();
    if (!entryObj) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "A5 OpLib V1 manifest '%s' contains non-object operator entry",
          path.data());
    }

    auto opName = entryObj->getString("op");
    auto family = entryObj->getString("family");
    auto status = entryObj->getString("a5_status");
    if (!opName || opName->empty() || !family || family->empty() || !status ||
        status->empty()) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "A5 OpLib V1 manifest '%s' contains operator entry with missing "
          "op/family/a5_status",
          path.data());
    }

    auto [it, inserted] = manifest.operators.try_emplace(opName->str());
    if (!inserted) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "A5 OpLib V1 manifest '%s' contains duplicate operator '%s'",
          path.data(), opName->data());
    }

    it->second.family = family->str();
    const std::string opNameStr = opName->str();
    if (*status == "implemented") {
      it->second.status = A5ManifestStatus::Implemented;
    } else if (*status == "deferred") {
      it->second.status = A5ManifestStatus::Deferred;
    } else {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "A5 OpLib V1 manifest '%s' contains "
                                     "unsupported status '%s' for op '%s'",
                                     path.data(), status->data(),
                                     opName->data());
    }

    if (auto deferredReason = entryObj->getString("deferred_reason"))
      it->second.deferredReason = deferredReason->str();

    auto hasPathWithPrefix = [&](StringRef key, StringRef prefix) -> bool {
      auto *paths = entryObj->getArray(key);
      if (!paths)
        return false;
      for (const llvm::json::Value &pathValue : *paths) {
        auto pathStr = pathValue.getAsString();
        if (pathStr && pathStr->starts_with(prefix))
          return true;
      }
      return false;
    };

    const bool hasPublicApiEvidence =
        hasPathWithPrefix("header_paths", "include/pto/common/pto_instr.hpp") ||
        hasPathWithPrefix("semantic_source_paths",
                          "include/pto/common/pto_instr.hpp:");
    const std::optional<StringRef> approvedRewriteTarget =
        getApprovedPublicApiRewriteTarget(opNameStr);

    if (approvedRewriteTarget && hasPublicApiEvidence) {
      it->second.semanticClass = A5ManifestSemanticClass::PublicApiRewrite;
    } else if (it->second.status == A5ManifestStatus::Implemented) {
      it->second.semanticClass = A5ManifestSemanticClass::NativeA5Impl;
    } else {
      it->second.semanticClass =
          A5ManifestSemanticClass::MissingAcceptedSemantics;
    }

    if (it->second.status == A5ManifestStatus::Implemented &&
        !it->second.deferredReason.empty()) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "A5 OpLib V1 manifest '%s' marks implemented op '%s' with non-empty "
          "deferred_reason",
          path.data(), opName->data());
    }
    if (it->second.status == A5ManifestStatus::Deferred &&
        it->second.deferredReason.empty()) {
      return llvm::createStringError(std::errc::invalid_argument,
                                     "A5 OpLib V1 manifest '%s' marks deferred "
                                     "op '%s' with empty deferred_reason",
                                     path.data(), opName->data());
    }
    if (approvedRewriteTarget && hasPublicApiEvidence &&
        it->second.status != A5ManifestStatus::Implemented) {
      return llvm::createStringError(
          std::errc::invalid_argument,
          "A5 OpLib V1 manifest '%s' keeps approved public_api_rewrite op '%s' "
          "deferred; expected implemented",
          path.data(), opName->data());
    }
  }

  return manifest;
}

static std::string extractSourceFilePath(Location loc) {
  if (auto fileLoc = dyn_cast<FileLineColLoc>(loc))
    return fileLoc.getFilename().str();
  if (auto nameLoc = dyn_cast<NameLoc>(loc))
    return extractSourceFilePath(nameLoc.getChildLoc());
  if (auto callLoc = dyn_cast<CallSiteLoc>(loc))
    return extractSourceFilePath(callLoc.getCallee());
  if (auto fusedLoc = dyn_cast<FusedLoc>(loc)) {
    for (Location child : fusedLoc.getLocations()) {
      std::string path = extractSourceFilePath(child);
      if (!path.empty())
        return path;
    }
  }
  return {};
}

static std::string findA5OpLibManifestFrom(StringRef startPath) {
  namespace fs = std::filesystem;

  if (startPath.empty())
    return {};

  std::error_code ec;
  fs::path current(startPath.str());
  if (fs::is_regular_file(current, ec))
    current = current.parent_path();

  while (!current.empty()) {
    fs::path candidate = current / kA5OpLibManifestRelPath.str();
    if (fs::exists(candidate, ec))
      return candidate.string();

    fs::path parent = current.parent_path();
    if (parent.empty() || parent == current)
      break;
    current = parent;
  }

  return {};
}

static std::string resolveA5OpLibManifestPath(StringRef explicitPath,
                                              StringRef opLibDir,
                                              Location moduleLoc) {
  if (!explicitPath.empty())
    return explicitPath.str();

  if (!opLibDir.empty()) {
    std::string siblingPath =
        (Twine(opLibDir) + "/families/a5_oplib_v1_manifest.yaml").str();
    if (llvm::sys::fs::exists(siblingPath))
      return siblingPath;

    std::string repoRelative = findA5OpLibManifestFrom(opLibDir);
    if (!repoRelative.empty())
      return repoRelative;
  }

  std::string moduleRelative =
      findA5OpLibManifestFrom(extractSourceFilePath(moduleLoc));
  if (!moduleRelative.empty())
    return moduleRelative;

  return {};
}

static bool isA5OpLibV1TargetOp(Operation *op,
                                const A5OpLibManifest &manifest) {
  if (op->getName().getDialectNamespace() != "pto")
    return false;
  return manifest.lookup(getPTOMnemonic(op)) != nullptr;
}

static bool isTileFusionOpLibAllowlisted(StringRef opName) {
  return llvm::StringSwitch<bool>(opName)
      .Cases("tadd", "tsub", "tmul", "tdiv", "tmax", "tmin", true)
      .Cases("tadds", "tsubs", "tmuls", "tdivs", "tmaxs", "tmins", true)
      .Cases("trowexpandmul", "texp", true)
      .Default(false);
}

static bool shouldLowerViaOpLib(Operation *op,
                                const A5OpLibManifest &manifest) {
  StringRef opName = getPTOMnemonic(op);
  const A5ManifestEntry *entry = manifest.lookup(opName);
  if (!entry || entry->status != A5ManifestStatus::Implemented)
    return false;

  // Temporarily keep OP-Lib lowering scoped to the active tile-fusion grouped
  // surface so grouped/single-op behavior stays aligned while other ops fall
  // back to the pre-existing non-OP-Lib lowering path.
  if (!isTileFusionOpLibAllowlisted(opName))
    return false;

  // A5 manifest marks tcolsum as implemented, but OP-Lib currently only
  // covers the linear variant. Preserve the native lowering path for the
  // binary form until a dedicated OP-Lib variant exists.
  if (auto colsum = dyn_cast<pto::TColSumOp>(op);
      colsum && colsum.getIsBinary())
    return false;

  return true;
}

static bool hasFusionGroupAttrs(Operation *op) {
  return op->hasAttr(kFusionGroupIdAttr) && op->hasAttr(kFusionOrderAttr);
}

static bool isOplibScalarType(Type ty) {
  return isa<FloatType, IntegerType, IndexType>(ty);
}

static bool isOplibTileElementType(Type ty) {
  return isa<FloatType, IntegerType>(ty);
}

static bool isOplibTileLikeType(Type ty, bool allowLoweredMemRefAbi) {
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    return tileTy.getRank() == 2 &&
           isOplibTileElementType(tileTy.getElementType());
  }
  if (!allowLoweredMemRefAbi)
    return false;
  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    return memTy.getRank() == 2 &&
           isOplibTileElementType(memTy.getElementType());
  }
  return false;
}

static int64_t getElemBytes(Type elemTy) {
  if (auto ft = dyn_cast<FloatType>(elemTy)) {
    if (ft.isF16() || ft.isBF16())
      return 2;
    if (ft.isF32())
      return 4;
    if (ft.isF64())
      return 8;
  }
  if (auto it = dyn_cast<IntegerType>(elemTy)) {
    int64_t bytes = it.getWidth() / 8;
    return bytes > 0 ? bytes : 1;
  }
  return -1;
}

static bool readBLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<pto::BLayoutAttr>(attr)) {
    out = static_cast<int32_t>(a.getValue());
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(a.getInt());
    return true;
  }
  return false;
}

static bool readSLayoutI32(Attribute attr, int32_t &out) {
  if (auto a = dyn_cast<pto::SLayoutAttr>(attr)) {
    out = static_cast<int32_t>(a.getValue());
    return true;
  }
  if (auto a = dyn_cast<IntegerAttr>(attr)) {
    out = static_cast<int32_t>(a.getInt());
    return true;
  }
  return false;
}

struct ValueTileMetadata {
  pto::TileBufConfigAttr config;
};

static void collectValueTileMetadata(Value v, ValueTileMetadata &meta) {
  if (!v || meta.config)
    return;

  Operation *def = v.getDefiningOp();
  if (!def)
    return;

  if (auto bind = dyn_cast<pto::BindTileOp>(def)) {
    meta.config = bind.getConfig();
    collectValueTileMetadata(bind.getSource(), meta);
    return;
  }

  if (auto pc = dyn_cast<pto::PointerCastOp>(def)) {
    if (auto cfg = pc.getConfig())
      meta.config = *cfg;
    return;
  }

  if (auto subview = dyn_cast<memref::SubViewOp>(def)) {
    collectValueTileMetadata(subview.getSource(), meta);
    return;
  }

  if (auto recast = dyn_cast<memref::ReinterpretCastOp>(def)) {
    collectValueTileMetadata(recast.getSource(), meta);
    return;
  }

  if (auto cast = dyn_cast<memref::CastOp>(def)) {
    collectValueTileMetadata(cast.getSource(), meta);
    return;
  }

  if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
    if (cast->getNumOperands() == 1)
      collectValueTileMetadata(cast.getOperand(0), meta);
    return;
  }
}

static pto::TileBufConfigAttr lookupTileConfigForValue(Value v,
                                                       MLIRContext *ctx) {
  ValueTileMetadata meta;
  collectValueTileMetadata(v, meta);
  if (meta.config)
    return meta.config;
  return pto::TileBufConfigAttr::getDefault(ctx);
}

static std::string toBLayoutName(int32_t bLayout) {
  switch (bLayout) {
  case 0:
    return "row_major";
  case 1:
    return "col_major";
  default:
    return kDefaultAny.str();
  }
}

static std::string toSLayoutName(int32_t sLayout) {
  switch (sLayout) {
  case 0:
    return "none_box";
  case 1:
    return "row_major";
  case 2:
    return "col_major";
  default:
    return kDefaultAny.str();
  }
}

static int64_t getFractalOrWildcard(pto::TileBufConfigAttr cfg) {
  if (!cfg)
    return -1;
  if (auto fractalAttr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    return fractalAttr.getInt();
  return -1;
}

static FailureOr<MemRefType> inferSimdBridgeMemRefType(pto::TileBufType tileTy,
                                                       MLIRContext *ctx) {
  if (tileTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> physicalShape = tileTy.getShape();
  if (physicalShape.size() != 2)
    return failure();
  if (physicalShape[0] == ShapedType::kDynamic ||
      physicalShape[1] == ShapedType::kDynamic)
    return failure();

  SmallVector<int64_t, 2> memShape(physicalShape.begin(), physicalShape.end());
  ArrayRef<int64_t> validShape = tileTy.getValidShape();
  if (validShape.size() == memShape.size()) {
    for (unsigned i = 0; i < validShape.size(); ++i) {
      // Keep OP-Lib bridge memrefs as static whenever the physical tile shape
      // is static. This lets canonicalization fold template-local memref.dim
      // users inside imported SIMD bodies instead of leaving illegal dynamic
      // shape IR.
      memShape[i] = validShape[i] < 0 ? physicalShape[i] : validShape[i];
    }
  }

  auto cfg = tileTy.getConfigAttr();
  if (!cfg)
    cfg = pto::TileBufConfigAttr::getDefault(ctx);

  int32_t bl = 0; // row_major
  int32_t sl = 0; // none_box
  int32_t fr = 512;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  if (auto attr = dyn_cast<IntegerAttr>(cfg.getSFractalSize()))
    fr = static_cast<int32_t>(attr.getInt());

  int64_t innerRows = 1;
  int64_t innerCols = 1;
  if (sl != 0) {
    int64_t elemBytes = getElemBytes(tileTy.getElementType());
    if (elemBytes <= 0)
      return failure();
    if (fr == 1024) {
      innerRows = 16;
      innerCols = 16;
    } else if (fr == 32) {
      innerRows = 16;
      innerCols = 2;
    } else if (fr == 512) {
      if (sl == 1) {
        innerRows = 16;
        innerCols = 32 / elemBytes;
      } else if (sl == 2) {
        innerRows = 32 / elemBytes;
        innerCols = 16;
      } else {
        return failure();
      }
    } else {
      return failure();
    }
  }

  SmallVector<int64_t, 2> strides;
  if (sl == 0) {
    if (bl == 1) {
      strides.push_back(1);
      strides.push_back(physicalShape[0]);
    } else {
      strides.push_back(physicalShape[1]);
      strides.push_back(1);
    }
  } else if (bl == 1) {
    if (sl != 1)
      return failure();
    strides.push_back(innerCols);
    strides.push_back(physicalShape[0]);
  } else {
    strides.push_back(physicalShape[1]);
    strides.push_back(innerRows);
  }

  auto layout = StridedLayoutAttr::get(ctx, /*offset=*/0, strides);
  return MemRefType::get(memShape, tileTy.getElementType(), layout,
                         tileTy.getMemorySpace());
}

static bool areIntegerCarrierTypesCompatible(Type lhs, Type rhs) {
  auto lhsInt = dyn_cast<IntegerType>(lhs);
  auto rhsInt = dyn_cast<IntegerType>(rhs);
  if (!lhsInt || !rhsInt)
    return false;
  return lhsInt.getWidth() == rhsInt.getWidth();
}

static bool canRemapScalarViaCarrierCast(Type actualTy, Type templateTy) {
  return areIntegerCarrierTypesCompatible(actualTy, templateTy);
}

static bool canRemapSimdBridgeViaCarrierCast(MemRefType actualTy,
                                             MemRefType templateTy) {
  if (actualTy.getRank() != templateTy.getRank())
    return false;
  if (actualTy.getMemorySpace() != templateTy.getMemorySpace())
    return false;
  return areIntegerCarrierTypesCompatible(actualTy.getElementType(),
                                          templateTy.getElementType());
}

static MemRefType remapMemRefToTemplateCarrier(MemRefType actualTy,
                                               MemRefType templateTy) {
  return MemRefType::get(actualTy.getShape(), templateTy.getElementType(),
                         actualTy.getLayout(), actualTy.getMemorySpace());
}

static std::string dtypeToString(Type ty) { return pto::getOpLibDTypeName(ty); }

static std::string typeToString(Type ty) {
  std::string s;
  llvm::raw_string_ostream os(s);
  ty.print(os);
  return os.str();
}

static std::string sanitizeSymbolComponent(StringRef text) {
  std::string out;
  out.reserve(text.size());
  for (char c : text) {
    if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
        (c >= '0' && c <= '9') || c == '_') {
      out.push_back(c);
    } else {
      out.push_back('_');
    }
  }
  if (out.empty())
    out = "unnamed";
  return out;
}

static bool isAllowedDTypeName(StringRef dtypeName) {
  return !dtypeName.empty();
}

static bool isAllowedLayoutName(StringRef layoutName) {
  return layoutName == "row_major" || layoutName == "col_major" ||
         layoutName == "none_box" || layoutName == "any";
}

static bool isAllowedCmpModeName(StringRef cmpMode) {
  return cmpMode == "EQ" || cmpMode == "NE" || cmpMode == "LT" ||
         cmpMode == "LE" || cmpMode == "GT" || cmpMode == "GE";
}

static bool containsString(ArrayRef<std::string> values, StringRef key) {
  return llvm::any_of(values, [&](const std::string &s) { return s == key; });
}

static bool canUseSeedRewriteFor(StringRef kind, StringRef op) {
  if (kind == "l3_float_binary_elementwise_template")
    return op == "tadd" || op == "tsub" || op == "tmul" || op == "tdiv" ||
           op == "tmax" || op == "tmin";
  if (kind == "l3_float_partial_binary_template")
    return op == "tpartadd" || op == "tpartmax" || op == "tpartmin";
  if (kind == "l3_float_tile_scalar_template")
    return op == "tadds" || op == "tsubs" || op == "tmuls" || op == "tmaxs" ||
           op == "tmins";
  if (kind == "l3_reduce_row_template")
    return op == "trowsum" || op == "trowmax" || op == "trowmin";
  if (kind == "l3_reduce_col_template")
    return op == "tcolmax" || op == "tcolmin";
  return false;
}

static std::optional<Level3TemplateFamily>
classifyLevel3TemplateFamily(StringRef kind) {
  using Family = Level3TemplateFamily;
  return llvm::StringSwitch<std::optional<Family>>(kind)
      // Current non-scalar-related Level-3 families in oplib/level3.
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
      // Current ABI-explicit-scalar Level-3 families in oplib/level3.
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

static bool kindRequiresIsBinary(StringRef kind) {
  return kind == "l3_reduce_colsum_template";
}

static bool kindRequiresCmpMode(StringRef kind) {
  return kind == "l3_cmp_tile_tile_template" ||
         kind == "l3_cmp_tile_scalar_template";
}

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

  // V1.2 strict mode: reject legacy bridge in OP-Lib templates.
  if (isa<UnrealizedConversionCastOp>(op))
    return false;

  // Family-aware Level-3 contract checks run before the generic allowlist.
  // Keep memref namespace ops available here for reinterpret_cast/subview/etc.
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

struct TemplateRegistry {
  ModuleOp module;
  SymbolTable &symbolTable;
  bool debug;

  SmallVector<std::unique_ptr<TemplateEntry>, 32> entries;
  llvm::StringMap<func::FuncOp> instanceByKey;
  llvm::StringSet<> seedIds;
  llvm::StringSet<> variantIds;

  LogicalResult emitFailure(Operation *anchor, const Twine &msg) {
    anchor->emitError(msg);
    return failure();
  }

  LogicalResult emitFailure(Location loc, const Twine &msg) {
    emitError(loc) << msg;
    return failure();
  }

  LogicalResult emitFailureWithCode(Operation *anchor, StringRef code,
                                    const Twine &msg) {
    anchor->emitError() << code << ": " << msg;
    return failure();
  }

  LogicalResult emitFailureWithCode(Location loc, StringRef code,
                                    const Twine &msg) {
    emitError(loc) << code << ": " << msg;
    return failure();
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

  static FailureOr<SmallVector<std::string, 8>>
  parseStringArrayAttr(Operation *op, StringRef attrName) {
    auto attr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!attr)
      return failure();

    SmallVector<std::string, 8> out;
    out.reserve(attr.size());
    for (Attribute item : attr) {
      auto s = dyn_cast<StringAttr>(item);
      if (!s)
        return failure();
      out.push_back(s.getValue().str());
    }
    return out;
  }

  static std::string getArgMatchAttrName(unsigned argIndex, StringRef suffix) {
    return (Twine("pto.oplib.match.arg") + Twine(argIndex) + "." + suffix)
        .str();
  }

  bool validateTemplateSignature(func::FuncOp fn, StringRef kind,
                                 SmallVectorImpl<TemplateArgRole> &argRoles,
                                 bool allowLoweredMemRefAbi = false) {
    FailureOr<SmallVector<TemplateArgRole, 4>> rolesOr =
        getTemplateArgRolesForKind(kind);
    if (failed(rolesOr))
      return false;

    argRoles.assign(rolesOr->begin(), rolesOr->end());
    FunctionType fnTy = fn.getFunctionType();
    if (fnTy.getNumInputs() != argRoles.size() || fnTy.getNumResults() != 0)
      return false;

    for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), argRoles)) {
      if (role == TemplateArgRole::Tile) {
        if (!isOplibTileLikeType(inTy, allowLoweredMemRefAbi))
          return false;
        continue;
      }
      if (!isOplibScalarType(inTy))
        return false;
    }
    return true;
  }

  FailureOr<func::FuncOp> cloneIntoModule(func::FuncOp libFunc) {
    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());

    std::string sym =
        "__pto_oplib_entry_" + sanitizeSymbolComponent(libFunc.getSymName());
    int suffix = 0;
    while (SymbolTable::lookupSymbolIn(module, sym)) {
      sym = "__pto_oplib_entry_" +
            sanitizeSymbolComponent(libFunc.getSymName()) + "_" +
            std::to_string(++suffix);
    }

    auto imported =
        cast<func::FuncOp>(modBuilder.clone(*libFunc.getOperation()));
    imported.setSymName(sym);
    imported.setPrivate();
    return imported;
  }

  LogicalResult parseCommonAttrs(func::FuncOp imported,
                                 std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();
    if (entry->kind == kOpLibKindL3BinaryTemplate) {
      auto rowsOr =
          parseI64Attr(op, kOpLibAttrMatchRows, /*allowWildcard=*/true);
      auto colsOr =
          parseI64Attr(op, kOpLibAttrMatchCols, /*allowWildcard=*/true);
      auto fractalOr =
          parseI64Attr(op, kOpLibAttrMatchFractal, /*allowWildcard=*/true);
      if (failed(rowsOr) || failed(colsOr) || failed(fractalOr)) {
        imported.emitError("missing or invalid match rows/cols/fractal attrs");
        return failure();
      }

      auto blayoutOr = parseStringAttr(op, kOpLibAttrMatchBLayout);
      auto slayoutOr = parseStringAttr(op, kOpLibAttrMatchSLayout);
      if (failed(blayoutOr) || failed(slayoutOr)) {
        imported.emitError(
            "missing match layout attrs: pto.oplib.match.blayout / "
            "pto.oplib.match.slayout");
        return failure();
      }
      if (!isAllowedLayoutName(*blayoutOr) ||
          !isAllowedLayoutName(*slayoutOr)) {
        imported.emitError(
            "invalid layout value in match.blayout/match.slayout");
        return failure();
      }

      MatchKey legacyMatch;
      legacyMatch.rows = *rowsOr;
      legacyMatch.cols = *colsOr;
      legacyMatch.fractal = *fractalOr;
      legacyMatch.blayout = *blayoutOr;
      legacyMatch.slayout = *slayoutOr;
      entry->argMatches.resize(entry->argRoles.size());
      for (unsigned i = 0; i < entry->argRoles.size(); ++i) {
        if (entry->argRoles[i] == TemplateArgRole::Tile)
          entry->argMatches[i] = legacyMatch;
      }
    } else {
      entry->argMatches.resize(entry->argRoles.size());
      for (unsigned i = 0; i < entry->argRoles.size(); ++i) {
        if (entry->argRoles[i] != TemplateArgRole::Tile) {
          bool hasUnexpectedAttr =
              op->hasAttr(getArgMatchAttrName(i, "rows")) ||
              op->hasAttr(getArgMatchAttrName(i, "cols")) ||
              op->hasAttr(getArgMatchAttrName(i, "blayout")) ||
              op->hasAttr(getArgMatchAttrName(i, "slayout")) ||
              op->hasAttr(getArgMatchAttrName(i, "fractal"));
          if (hasUnexpectedAttr) {
            imported.emitError()
                << "arg" << i << " is scalar in kind=" << entry->kind
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
          imported.emitError()
              << "missing or invalid arg" << i
              << " match attrs: rows/cols/blayout/slayout/fractal";
          return failure();
        }
        if (!isAllowedLayoutName(*blayoutOr) ||
            !isAllowedLayoutName(*slayoutOr)) {
          imported.emitError()
              << "invalid layout value in arg" << i << ".blayout/.slayout";
          return failure();
        }

        MatchKey argMatch;
        argMatch.rows = *rowsOr;
        argMatch.cols = *colsOr;
        argMatch.fractal = *fractalOr;
        argMatch.blayout = *blayoutOr;
        argMatch.slayout = *slayoutOr;
        entry->argMatches[i] = argMatch;
      }
    }

    auto costAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrCost);
    auto priorityAttr = op->getAttrOfType<IntegerAttr>(kOpLibAttrPriority);
    if (!costAttr || !priorityAttr) {
      imported.emitError("missing required attrs: pto.oplib.cost / "
                         "pto.oplib.priority");
      return failure();
    }

    entry->cost = costAttr.getInt();
    entry->priority = priorityAttr.getInt();

    if (auto syncAttr = op->getAttrOfType<BoolAttr>(kOpLibAttrSync))
      entry->sync = syncAttr.getValue();

    if (auto scalarPosAttr =
            op->getAttrOfType<IntegerAttr>(kOpLibAttrMatchScalarPos)) {
      int64_t scalarPos = scalarPosAttr.getInt();
      if (scalarPos < 0 ||
          scalarPos >= static_cast<int64_t>(entry->argRoles.size()) ||
          entry->argRoles[scalarPos] != TemplateArgRole::Scalar) {
        imported.emitError("invalid pto.oplib.match.scalar_pos");
        return failure();
      }
      entry->scalarPos = scalarPos;
    }

    if (op->hasAttr(kOpLibAttrMatchCmpMode)) {
      auto cmpModeOr = parseStringAttr(op, kOpLibAttrMatchCmpMode);
      if (failed(cmpModeOr) || !isAllowedCmpModeName(*cmpModeOr)) {
        imported.emitError("invalid pto.oplib.match.cmp_mode");
        return failure();
      }
      entry->cmpMode = *cmpModeOr;
    } else if (kindRequiresCmpMode(entry->kind)) {
      imported.emitError("missing required attr: pto.oplib.match.cmp_mode");
      return failure();
    }

    if (auto isBinaryAttr =
            op->getAttrOfType<BoolAttr>(kOpLibAttrMatchIsBinary)) {
      entry->isBinary = isBinaryAttr.getValue();
    } else if (kindRequiresIsBinary(entry->kind)) {
      imported.emitError("missing required attr: pto.oplib.match.is_binary");
      return failure();
    }

    return success();
  }

  LogicalResult parseVariantAttrs(func::FuncOp imported,
                                  std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();

    auto opOr = parseStringAttr(op, kOpLibAttrOp);
    auto variantIdOr = parseStringAttr(op, kOpLibAttrVariantId);
    auto dtypeOr = parseStringAttr(op, kOpLibAttrMatchDType);
    if (failed(opOr) || failed(variantIdOr) || failed(dtypeOr)) {
      imported.emitError("variant entry missing attrs: pto.oplib.op / "
                         "pto.oplib.variant_id / pto.oplib.match.dtype");
      return failure();
    }

    if (!isAllowedDTypeName(*dtypeOr)) {
      imported.emitError() << "unsupported variant dtype: " << *dtypeOr;
      return failure();
    }

    std::string uniqueKey =
        entry->kind + "|" + *opOr + "|" + *variantIdOr + "|" + *dtypeOr;
    if (!variantIds.insert(uniqueKey).second) {
      imported.emitError() << "duplicate variant_id under op=" << *opOr << ": "
                           << *variantIdOr;
      return failure();
    }

    entry->role = EntryRole::Variant;
    entry->op = *opOr;
    entry->variantId = *variantIdOr;
    entry->matchDType = *dtypeOr;

    return success();
  }

  LogicalResult parseSeedAttrs(func::FuncOp imported,
                               std::unique_ptr<TemplateEntry> &entry) {
    Operation *op = imported.getOperation();

    auto seedIdOr = parseStringAttr(op, kOpLibAttrSeedId);
    auto seedDTypeOr = parseStringAttr(op, kOpLibAttrSeedDType);
    auto supportDTypesOr =
        parseStringArrayAttr(op, kOpLibAttrSeedSupportDTypes);
    auto supportOpsOr = parseStringArrayAttr(op, kOpLibAttrSeedSupportOps);
    if (failed(seedIdOr) || failed(seedDTypeOr) || failed(supportDTypesOr) ||
        failed(supportOpsOr)) {
      imported.emitError(
          "seed entry missing attrs: pto.oplib.seed_id / "
          "pto.oplib.seed_dtype / pto.oplib.seed.support_dtypes / "
          "pto.oplib.seed.support_ops");
      return failure();
    }

    std::string uniqueSeedKey = entry->kind + "|" + *seedIdOr;
    if (!seedIds.insert(uniqueSeedKey).second) {
      imported.emitError() << "duplicate seed_id under kind=" << entry->kind
                           << ": " << *seedIdOr;
      return failure();
    }

    if (!isAllowedDTypeName(*seedDTypeOr)) {
      imported.emitError() << "unsupported seed_dtype: " << *seedDTypeOr;
      return failure();
    }

    for (const std::string &dtype : *supportDTypesOr) {
      if (!isAllowedDTypeName(dtype)) {
        imported.emitError() << "unsupported support_dtypes item: " << dtype;
        return failure();
      }
    }

    entry->role = EntryRole::Seed;
    entry->seedId = *seedIdOr;
    entry->seedDType = *seedDTypeOr;
    entry->supportDTypes = *supportDTypesOr;
    entry->supportOps = *supportOpsOr;

    if (auto slotAttr = op->getAttrOfType<StringAttr>(kOpLibAttrSeedCoreSlot))
      entry->coreSlot = slotAttr.getValue().str();

    return success();
  }

  LogicalResult parseSimdAttrs(func::FuncOp imported,
                               std::unique_ptr<TemplateEntry> &entry,
                               bool hasSimdOps) {
    Operation *op = imported.getOperation();
    auto levelAttr = op->getAttrOfType<StringAttr>(kSimdLevelAttr);
    auto lanesAttr = op->getAttrOfType<IntegerAttr>(kSimdLanesAttr);
    bool hasLevelAttr = static_cast<bool>(levelAttr);
    bool hasLanesAttr = static_cast<bool>(lanesAttr);

    if (!hasSimdOps && !hasLevelAttr && !hasLanesAttr)
      return success();

    if (!hasLevelAttr || !hasLanesAttr) {
      return emitFailureWithCode(
          op, kErrSimdAttrRequired,
          "using pto.simd.* requires attrs: pto.simd.level / pto.simd.lanes");
    }

    auto levelOr = parseStringAttr(op, kSimdLevelAttr);
    auto lanesOr = parseI64Attr(op, kSimdLanesAttr, /*allowWildcard=*/false);
    if (failed(levelOr) || failed(lanesOr)) {
      return emitFailureWithCode(
          op, kErrLanesMismatch,
          "missing required attrs: pto.simd.level / pto.simd.lanes");
    }
    if (*levelOr != kSimdLevelBinaryEwiseV1) {
      return emitFailureWithCode(
          op, kErrCoreSlot, Twine("unsupported pto.simd.level: ") + *levelOr);
    }
    if (isLevel3TemplateKind(entry->kind) &&
        inferExpectedLevel3SimdLanes(imported.getFunctionType(),
                                     entry->argRoles) &&
        *lanesOr != *inferExpectedLevel3SimdLanes(imported.getFunctionType(),
                                                  entry->argRoles)) {
      return emitFailureWithCode(
          op, kErrLanesMismatch,
          Twine("Level-3 template kind=") + entry->kind +
              " requires pto.simd.lanes = " +
              Twine(*inferExpectedLevel3SimdLanes(imported.getFunctionType(),
                                                  entry->argRoles)) +
              ", got " + Twine(*lanesOr));
    }
    entry->simdLevel = *levelOr;
    entry->simdLanes = *lanesOr;
    entry->hasSimdBridgeOps = hasSimdOps;
    return success();
  }

  LogicalResult validateTemplateBody(func::FuncOp imported,
                                     const TemplateEntry &entry) {
    if (imported.isExternal() || imported.empty() ||
        imported.front().without_terminator().empty()) {
      return emitFailureWithCode(imported.getLoc(), kErrEmptyBody,
                                 "template body must not be empty");
    }

    bool requireExplicitExecMode =
        opRequiresExplicitExecMode(entry.kind, entry.op);

    FunctionType fnTy = imported.getFunctionType();
    for (auto [inTy, role] : llvm::zip(fnTy.getInputs(), entry.argRoles)) {
      if (role == TemplateArgRole::Scalar) {
        if (!isOplibScalarType(inTy)) {
          return emitFailureWithCode(
              imported.getLoc(), kErrDType,
              "template scalar inputs must use builtin scalar types");
        }
        continue;
      }

      auto tileTy = dyn_cast<pto::TileBufType>(inTy);
      if (!tileTy) {
        return emitFailureWithCode(
            imported.getLoc(), kErrEmptyBody,
            "template tile inputs must be !pto.tile_buf");
      }
      Type elemTy = tileTy.getElementType();
      if (!isOplibTileElementType(elemTy)) {
        return emitFailureWithCode(
            imported.getLoc(), kErrDType,
            "SIMD template supports float/integer tile inputs only");
      }
      if (tileTy.getBLayoutValueI32() != 0) {
        return emitFailureWithCode(imported.getLoc(), kErrLayout,
                                   "SIMD template supports row_major only");
      }
    }

    int64_t firstLoad = std::numeric_limits<int64_t>::max();
    int64_t firstStore = std::numeric_limits<int64_t>::max();
    int64_t coreSeq = -1;
    int coreCount = 0;
    bool sawLevel3VectorValue = false;
    std::string inferredVldDist;
    std::string inferredVstDist;
    std::string inferredExecMode;
    bool requiresUnifiedLevel3Simd = isLevel3TemplateKind(entry.kind);
    bool hasPackedCmpBridgeOnly = false;
    if (entry.hasSimdBridgeOps) {
      bool sawPackedCmp = false;
      bool sawOtherBridge = false;
      imported.walk([&](Operation *op) {
        if (!isSimdBridgeOp(op))
          return WalkResult::advance();
        if (isPackedCmpSimdBridgeOp(op)) {
          sawPackedCmp = true;
          return WalkResult::advance();
        }
        sawOtherBridge = true;
        return WalkResult::interrupt();
      });
      hasPackedCmpBridgeOnly = sawPackedCmp && !sawOtherBridge;
    }

    llvm::DenseMap<Operation *, int64_t> preorder;
    int64_t seq = 0;
    imported.walk([&](Operation *op) { preorder[op] = seq++; });

    LogicalResult status = success();
    int64_t effectiveSimdLanes =
        entry.simdLanes > 0 ? entry.simdLanes
                            : inferExpectedLevel3SimdLanes(
                                  imported.getFunctionType(), entry.argRoles)
                                  .value_or(-1);
    auto requireLevel3VectorLanes = [&](Operation *targetOp, Type ty,
                                        StringRef position) -> bool {
      if (!requiresUnifiedLevel3Simd)
        return true;
      auto vecTy = dyn_cast<VectorType>(ty);
      if (!vecTy || vecTy.isScalable())
        return true;
      FailureOr<int64_t> lanes = getFixedVectorLanes(ty);
      if (failed(lanes))
        return true;
      sawLevel3VectorValue = true;
      Type elemTy = vecTy.getElementType();
      if (elemTy.isInteger(1)) {
        if (!isSupportedA5MaskLanes(*lanes)) {
          status = emitFailureWithCode(
              targetOp, kErrLanesMismatch,
              Twine("Level-3 template kind=") + entry.kind +
                  " requires vector<i1> masks with supported A5 lane counts "
                  "(32/64/128/256), but " +
                  position + " of op '" + targetOp->getName().getStringRef() +
                  "' uses " + Twine(*lanes) + " lanes");
          return false;
        }
      } else {
        auto expectedLanes = getExpectedA5SimdLanesForElemType(elemTy);
        if (!expectedLanes) {
          status = emitFailureWithCode(
              targetOp, kErrDType,
              Twine("Level-3 template kind=") + entry.kind +
                  " uses unsupported A5 SIMD element type " +
                  typeToString(elemTy));
          return false;
        }
        if (*lanes != *expectedLanes &&
            !isA5ByteLaneCarrierVector(elemTy, *lanes, effectiveSimdLanes)) {
          status = emitFailureWithCode(
              targetOp, kErrLanesMismatch,
              Twine("Level-3 template kind=") + entry.kind + " requires " +
                  typeToString(elemTy) + " vectors to use " +
                  Twine(*expectedLanes) + " lanes, but " + position +
                  " of op '" + targetOp->getName().getStringRef() + "' uses " +
                  Twine(*lanes) + " lanes");
          return false;
        }
      }
      if (effectiveSimdLanes > 0 && *lanes != effectiveSimdLanes) {
        status = emitFailureWithCode(
            targetOp, kErrLanesMismatch,
            Twine("Level-3 template kind=") + entry.kind +
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
        status = emitFailureWithCode(targetOp, kErrSimdAttrRequired,
                                     Twine(usage) + " requires string attr '" +
                                         attrName + "'");
        return false;
      }

      StringRef token = tokenAttr.getValue();
      if (token.empty()) {
        status = emitFailureWithCode(targetOp, kErrSimdAttrRequired,
                                     Twine(usage) + " attr '" + attrName +
                                         "' must be non-empty");
        return false;
      }

      if (!requiredPrefix.empty() && !token.starts_with(requiredPrefix)) {
        status = emitFailureWithCode(
            targetOp, kErrSimdAttrRequired,
            Twine(usage) + " attr '" + attrName + "' must start with '" +
                requiredPrefix + "', got '" + token + "'");
        return false;
      }

      return true;
    };

    imported.walk([&](Operation *op) {
      if (failed(status))
        return;

      if (op == imported.getOperation())
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
        status = emitFailureWithCode(op, kErrBodyDisallowedIR,
                                     Twine("unsupported template body op: ") +
                                         op->getName().getStringRef());
        return;
      }

      if (isa<vector::LoadOp, vector::MaskedLoadOp>(op)) {
        if (!requireNonEmptyTokenAttr(op, kSimdVldDistAttr, "vector load",
                                      /*requiredPrefix=*/""))
          return;
        if (inferredVldDist.empty()) {
          auto vld = op->getAttrOfType<StringAttr>(kSimdVldDistAttr);
          if (vld)
            inferredVldDist = vld.getValue().str();
        }
      }

      if (isa<vector::StoreOp, vector::MaskedStoreOp>(op)) {
        if (!requireNonEmptyTokenAttr(op, kSimdVstDistAttr, "vector store",
                                      /*requiredPrefix=*/"DIST_"))
          return;
        if (inferredVstDist.empty()) {
          auto vst = op->getAttrOfType<StringAttr>(kSimdVstDistAttr);
          if (vst)
            inferredVstDist = vst.getValue().str();
        }
      }

      if (isVectorFloatBinaryArith(op) &&
          (requireExplicitExecMode || op->hasAttr(kSimdExecModeAttr))) {
        if (!requireNonEmptyTokenAttr(op, kSimdExecModeAttr,
                                      "vector float binary arith op",
                                      /*requiredPrefix=*/"MODE_"))
          return;
        if (inferredExecMode.empty()) {
          auto execMode = op->getAttrOfType<StringAttr>(kSimdExecModeAttr);
          if (execMode)
            inferredExecMode = execMode.getValue().str();
        }
      }

      if (isa<pto::SimdLoadOp, pto::SimdLoadPUOp>(op))
        firstLoad = std::min(firstLoad, preorder[op]);
      if (isa<pto::SimdStoreOp, pto::SimdStorePUOp>(op))
        firstStore = std::min(firstStore, preorder[op]);

      if (auto pred = dyn_cast<pto::SimdPredicateOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(pred, kErrSimdAttrRequired,
                                       "using pto.simd.* requires attrs: "
                                       "pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(pred.getMask().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              pred, kErrLanesMismatch,
              "simd.predicate lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto load = dyn_cast<pto::SimdLoadOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(load, kErrSimdAttrRequired,
                                       "using pto.simd.* requires attrs: "
                                       "pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(load.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              load, kErrLanesMismatch,
              "simd.load lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto loadPU = dyn_cast<pto::SimdLoadPUOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(loadPU, kErrSimdAttrRequired,
                                       "using pto.simd.* requires attrs: "
                                       "pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(loadPU.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              loadPU, kErrLanesMismatch,
              "simd.load_pu lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto store = dyn_cast<pto::SimdStoreOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(store, kErrSimdAttrRequired,
                                       "using pto.simd.* requires attrs: "
                                       "pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(store.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              store, kErrLanesMismatch,
              "simd.store lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      if (auto storePU = dyn_cast<pto::SimdStorePUOp>(op)) {
        if (entry.simdLanes <= 0) {
          status = emitFailureWithCode(storePU, kErrSimdAttrRequired,
                                       "using pto.simd.* requires attrs: "
                                       "pto.simd.level / pto.simd.lanes");
          return;
        }
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(storePU.getValue().getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              storePU, kErrLanesMismatch,
              "simd.store_pu lanes mismatch with pto.simd.lanes");
        }
        return;
      }

      auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr);
      if (!slotAttr)
        return;

      if (slotAttr.getValue() != entry.coreSlot) {
        status = emitFailureWithCode(
            op, kErrCoreSlot, "core slot value mismatches seed.core_slot");
        return;
      }
      if (!isBinaryCore(op)) {
        status = emitFailureWithCode(op, kErrCoreSlot,
                                     "core slot op must be one of "
                                     "arith.addf/subf/mulf/divf/maximumf/"
                                     "minimumf/addi/subi/muli/divsi/divui");
        return;
      }
      if (op->getNumResults() != 1) {
        status = emitFailureWithCode(
            op, kErrCoreSlot, "core slot op must have exactly one result");
        return;
      }
      if (entry.simdLanes > 0) {
        FailureOr<int64_t> lanes =
            getFixedVectorLanes(op->getResult(0).getType());
        if (failed(lanes) || *lanes != entry.simdLanes) {
          status = emitFailureWithCode(
              op, kErrLanesMismatch,
              "core slot vector lanes mismatch with pto.simd.lanes");
          return;
        }
      }
      coreSeq = preorder[op];
      ++coreCount;
    });

    if (failed(status))
      return failure();

    if (requiresUnifiedLevel3Simd && !sawLevel3VectorValue) {
      return emitFailureWithCode(
          imported.getOperation(), kErrLanesMismatch,
          Twine("Level-3 template kind=") + entry.kind +
              " must materialize a supported A5 SIMD body");
    }

    // Canonicalization may rewrite vector.maskedload/store to vector.load/store
    // and drop op attrs. Keep function-local and enclosing-op fallback tokens
    // on imported OP-Lib regions so EmitC lowering can still recover vld/vst
    // tokens from ancestors after later rewrites.
    if (!inferredVldDist.empty() || !inferredVstDist.empty() ||
        !inferredExecMode.empty()) {
      auto *ctx = imported.getContext();
      if (!inferredVldDist.empty() && !imported->getAttr(kSimdVldDistAttr)) {
        imported->setAttr(kSimdVldDistAttr,
                          StringAttr::get(ctx, inferredVldDist));
      }
      if (!inferredVstDist.empty() && !imported->getAttr(kSimdVstDistAttr)) {
        imported->setAttr(kSimdVstDistAttr,
                          StringAttr::get(ctx, inferredVstDist));
      }
      if (!inferredExecMode.empty() && !imported->getAttr(kSimdExecModeAttr)) {
        imported->setAttr(kSimdExecModeAttr,
                          StringAttr::get(ctx, inferredExecMode));
      }
      imported.walk([&](Operation *op) {
        if (!inferredVldDist.empty() && !op->getAttr(kSimdVldDistAttr)) {
          op->setAttr(kSimdVldDistAttr, StringAttr::get(ctx, inferredVldDist));
        }
        if (!inferredVstDist.empty() && !op->getAttr(kSimdVstDistAttr)) {
          op->setAttr(kSimdVstDistAttr, StringAttr::get(ctx, inferredVstDist));
        }
        if (!inferredExecMode.empty() && !op->getAttr(kSimdExecModeAttr)) {
          op->setAttr(kSimdExecModeAttr,
                      StringAttr::get(ctx, inferredExecMode));
        }
      });
    }

    if (entry.role == EntryRole::Seed && coreCount != 1) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "seed template must contain exactly one core slot op");
    }
    if (entry.hasSimdBridgeOps && !hasPackedCmpBridgeOnly && coreCount != 1) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "template using pto.simd.* must contain exactly one core slot op");
    }
    if (coreCount > 1) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "template must not contain multiple core slot ops");
    }
    if (entry.hasSimdBridgeOps && !hasPackedCmpBridgeOnly &&
        firstLoad == std::numeric_limits<int64_t>::max()) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "template must contain simd.load/simd.load_pu");
    }
    if (entry.hasSimdBridgeOps && !hasPackedCmpBridgeOnly &&
        firstStore == std::numeric_limits<int64_t>::max()) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "template must contain simd.store/simd.store_pu");
    }
    if (entry.hasSimdBridgeOps && !hasPackedCmpBridgeOnly &&
        !(firstLoad < coreSeq && coreSeq < firstStore)) {
      return emitFailureWithCode(
          imported.getLoc(), kErrCoreSlot,
          "template ordering must satisfy load -> core -> store");
    }

    return success();
  }

  LogicalResult registerImportedEntry(func::FuncOp imported,
                                      StringRef sourceTag, bool validateBody,
                                      bool allowLoweredMemRefAbi = false) {
    auto kindAttr = imported->getAttrOfType<StringAttr>(kOpLibAttrKind);
    if (!kindAttr) {
      imported.emitError("missing required attr: pto.oplib.kind");
      return failure();
    }

    auto roleAttr = imported->getAttrOfType<StringAttr>(kOpLibAttrEntryRole);
    if (!roleAttr) {
      imported.emitError("missing required attr: pto.oplib.entry_role");
      return failure();
    }

    auto entry = std::make_unique<TemplateEntry>();
    entry->symbol = imported;
    entry->kind = kindAttr.getValue().str();

    if (!validateTemplateSignature(imported, entry->kind, entry->argRoles,
                                   allowLoweredMemRefAbi)) {
      imported.emitError() << "invalid OP-Lib signature for kind="
                           << entry->kind;
      return failure();
    }

    if (failed(parseCommonAttrs(imported, entry)))
      return failure();

    if (roleAttr.getValue() == kEntryRoleVariant) {
      if (failed(parseVariantAttrs(imported, entry)))
        return failure();
    } else if (roleAttr.getValue() == kEntryRoleSeed) {
      if (failed(parseSeedAttrs(imported, entry)))
        return failure();
    } else {
      imported.emitError("invalid pto.oplib.entry_role, expected variant|seed");
      return failure();
    }

    bool hasSimdOps = false;
    imported.walk([&](Operation *bodyOp) {
      if (isSimdBridgeOp(bodyOp))
        hasSimdOps = true;
    });

    if (failed(parseSimdAttrs(imported, entry, hasSimdOps)))
      return failure();

    if (validateBody && failed(validateTemplateBody(imported, *entry)))
      return failure();

    if (debug) {
      llvm::errs() << "[op-fusion] imported OP-Lib entry: role="
                   << (entry->role == EntryRole::Variant ? "variant" : "seed");
      if (entry->role == EntryRole::Variant) {
        llvm::errs() << " op=" << entry->op
                     << " variant_id=" << entry->variantId;
      } else {
        llvm::errs() << " seed_id=" << entry->seedId;
      }
      llvm::errs() << " symbol=@" << imported.getSymName();
      if (!sourceTag.empty())
        llvm::errs() << " source=" << sourceTag;
      llvm::errs() << "\n";
    }

    entries.push_back(std::move(entry));
    return success();
  }

  LogicalResult loadFromModuleEntries(Location loc, int &importedCount) {
    importedCount = 0;
    for (func::FuncOp fn : module.getOps<func::FuncOp>()) {
      auto kindAttr = fn->getAttrOfType<StringAttr>(kOpLibAttrKind);
      if (!kindAttr)
        continue;
      if (failed(getTemplateArgRolesForKind(kindAttr.getValue()))) {
        fn.emitError() << "unsupported pto.oplib.kind: " << kindAttr.getValue();
        return failure();
      }
      if (failed(registerImportedEntry(fn, "module", /*validateBody=*/false,
                                       /*allowLoweredMemRefAbi=*/true)))
        return failure();
      ++importedCount;
    }

    if (importedCount > 0)
      return success();

    (void)loc;
    return success();
  }

  LogicalResult loadFromDir(StringRef opLibDir, MLIRContext *ctx,
                            Location loc) {
    if (opLibDir.empty()) {
      return emitFailure(
          loc,
          "OP-Lib directory must be resolved before OP-LIB lowering runs");
    }

    llvm::SmallVector<std::string, 16> mlirFiles;
    std::error_code ec;
    for (llvm::sys::fs::directory_iterator it(opLibDir, ec), end;
         it != end && !ec; it.increment(ec)) {
      if (!llvm::sys::fs::is_regular_file(it->path()))
        continue;
      if (llvm::sys::path::extension(it->path()) == ".mlir")
        mlirFiles.push_back(it->path());
    }
    if (ec) {
      return emitFailure(loc, Twine("failed to iterate --op-lib-dir '") +
                                  opLibDir + "': " + ec.message());
    }

    llvm::sort(mlirFiles);
    if (mlirFiles.empty()) {
      return emitFailure(loc, Twine("no .mlir files found in --op-lib-dir '") +
                                  opLibDir + "'");
    }

    int importedCount = 0;
    for (const std::string &path : mlirFiles) {
      auto fileOrErr = llvm::MemoryBuffer::getFile(path);
      if (!fileOrErr) {
        return emitFailure(loc, Twine("failed to read OP-Lib file '") + path +
                                    "': " + fileOrErr.getError().message());
      }

      llvm::SourceMgr sourceMgr;
      sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
      OwningOpRef<ModuleOp> libModule =
          parseSourceFile<ModuleOp>(sourceMgr, ctx);
      if (!libModule)
        return emitFailure(loc,
                           Twine("failed to parse OP-Lib file '") + path + "'");

      for (func::FuncOp libFunc : libModule->getOps<func::FuncOp>()) {
        auto kindAttr = libFunc->getAttrOfType<StringAttr>(kOpLibAttrKind);
        if (!kindAttr)
          continue;
        SmallVector<TemplateArgRole, 4> argRoles;
        if (failed(getTemplateArgRolesForKind(kindAttr.getValue()))) {
          libFunc.emitError()
              << "unsupported pto.oplib.kind: " << kindAttr.getValue();
          return failure();
        }
        if (!validateTemplateSignature(libFunc, kindAttr.getValue(), argRoles,
                                       /*allowLoweredMemRefAbi=*/false)) {
          libFunc.emitError()
              << "invalid OP-Lib signature for kind=" << kindAttr.getValue();
          return failure();
        }

        FailureOr<func::FuncOp> importedOr = cloneIntoModule(libFunc);
        if (failed(importedOr)) {
          libFunc.emitError("failed to import template function into module");
          return failure();
        }
        func::FuncOp imported = *importedOr;
        if (failed(registerImportedEntry(imported, path, /*validateBody=*/true,
                                         /*allowLoweredMemRefAbi=*/false)))
          return failure();
        ++importedCount;
      }
    }

    if (importedCount == 0)
      return emitFailure(loc,
                         "no valid OP-Lib entries imported from --op-lib-dir");

    return success();
  }

  static bool matchDim(int64_t pattern, int64_t target) {
    if (pattern == -1)
      return true;
    if (target == -1)
      return false;
    return pattern == target;
  }

  static bool matchLayout(StringRef pattern, StringRef target) {
    if (pattern == kDefaultAny)
      return true;
    return pattern == target;
  }

  static bool areDegenerateMajorLayoutsCompatible(const MatchKey &pattern,
                                                  const MatchKey &request) {
    auto isMajorLayout = [](StringRef layout) {
      return layout == "row_major" || layout == "col_major";
    };
    if (!isMajorLayout(pattern.blayout) || !isMajorLayout(request.blayout))
      return false;
    if (pattern.blayout == request.blayout)
      return true;
    return request.rows == 1 || request.cols == 1;
  }

  static bool matchCommon(const TemplateEntry &entry,
                          const MatchRequest &target) {
    if (entry.kind != target.kind)
      return false;
    if (entry.argRoles.size() != target.argTypes.size() ||
        entry.argMatches.size() != target.argTypes.size())
      return false;

    for (unsigned i = 0; i < entry.argRoles.size(); ++i) {
      if (entry.argRoles[i] == TemplateArgRole::Scalar)
        continue;
      if (!entry.argMatches[i] || !target.argMatches[i])
        return false;
      const MatchKey &pattern = *entry.argMatches[i];
      const MatchKey &request = *target.argMatches[i];
      if (!matchDim(pattern.rows, request.rows))
        return false;
      if (!matchDim(pattern.cols, request.cols))
        return false;
      if (!matchDim(pattern.fractal, request.fractal))
        return false;
      if (!matchLayout(pattern.blayout, request.blayout) &&
          !areDegenerateMajorLayoutsCompatible(pattern, request))
        return false;
      if (!matchLayout(pattern.slayout, request.slayout))
        return false;
    }

    if (entry.scalarPos &&
        (!target.scalarPos || *entry.scalarPos != *target.scalarPos))
      return false;
    if (entry.cmpMode && (!target.cmpMode || *entry.cmpMode != *target.cmpMode))
      return false;
    if (entry.isBinary &&
        (!target.isBinary || *entry.isBinary != *target.isBinary))
      return false;
    return true;
  }

  FailureOr<SelectedVariant> selectVariantFor(const MatchRequest &request,
                                              Location loc) {
    SmallVector<SelectedVariant, 16> candidates;
    for (std::unique_ptr<TemplateEntry> &entryPtr : entries) {
      TemplateEntry &entry = *entryPtr;
      if (!matchCommon(entry, request))
        continue;

      if (entry.role == EntryRole::Variant) {
        if (entry.op != request.op)
          continue;
        if (entry.matchDType != request.dtype)
          continue;
        if (request.requiredVariantId &&
            entry.variantId != *request.requiredVariantId)
          continue;

        SelectedVariant selected;
        selected.entry = &entry;
        selected.kind = entry.kind;
        selected.variantId = entry.variantId;
        selected.op = entry.op;
        selected.dtype = entry.matchDType;
        selected.fromSeed = false;
        selected.coreSlot = entry.coreSlot;
        selected.cost = entry.cost;
        selected.priority = entry.priority;
        if (request.scalarPos)
          selected.instanceKeyAttrs.push_back(
              "scalar_pos=" + std::to_string(*request.scalarPos));
        if (request.cmpMode)
          selected.instanceKeyAttrs.push_back("cmp_mode=" + *request.cmpMode);
        if (request.isBinary)
          selected.instanceKeyAttrs.push_back(
              "is_binary=" + std::string(*request.isBinary ? "true" : "false"));
        candidates.push_back(std::move(selected));
        continue;
      }

      if (request.requiredVariantId)
        continue;
      if (!canUseSeedRewriteFor(entry.kind, request.op))
        continue;
      if (!containsString(entry.supportOps, request.op))
        continue;
      if (!containsString(entry.supportDTypes, request.dtype))
        continue;

      std::string variantId =
          "__seed__" + entry.seedId + "__" + request.op + "__" + request.dtype;
      SelectedVariant selected;
      selected.entry = &entry;
      selected.kind = entry.kind;
      selected.variantId = variantId;
      selected.op = request.op;
      selected.dtype = request.dtype;
      selected.fromSeed = true;
      selected.seedId = entry.seedId;
      selected.coreSlot = entry.coreSlot;
      selected.cost = entry.cost;
      selected.priority = entry.priority;
      if (request.scalarPos)
        selected.instanceKeyAttrs.push_back("scalar_pos=" +
                                            std::to_string(*request.scalarPos));
      if (request.cmpMode)
        selected.instanceKeyAttrs.push_back("cmp_mode=" + *request.cmpMode);
      if (request.isBinary)
        selected.instanceKeyAttrs.push_back(
            "is_binary=" + std::string(*request.isBinary ? "true" : "false"));
      candidates.push_back(std::move(selected));
    }

    if (candidates.empty()) {
      (void)emitFailure(loc, Twine("no matching OP-Lib entry for kind=") +
                                 request.kind + " op=" + request.op +
                                 " dtype=" + request.dtype);
      return failure();
    }

    llvm::sort(candidates,
               [](const SelectedVariant &lhs, const SelectedVariant &rhs) {
                 if (lhs.cost != rhs.cost)
                   return lhs.cost < rhs.cost;
                 if (lhs.priority != rhs.priority)
                   return lhs.priority > rhs.priority;
                 return lhs.variantId < rhs.variantId;
               });

    if (debug) {
      const SelectedVariant &best = candidates.front();
      llvm::errs() << "[op-fusion] selected variant: kind=" << best.kind
                   << " op=" << best.op << " dtype=" << best.dtype
                   << " variant_id=" << best.variantId << " cost=" << best.cost
                   << " priority=" << best.priority
                   << (best.fromSeed ? " source=seed" : " source=variant")
                   << "\n";
    }

    return candidates.front();
  }

  FailureOr<func::FuncOp> getOrCreateInstance(const SelectedVariant &selected,
                                              ArrayRef<Type> concreteArgTypes,
                                              Location loc) {
    std::string key = selected.kind;
    key += "|";
    key += selected.op;
    key += "|";
    key += selected.variantId;
    for (const std::string &piece : selected.instanceKeyAttrs) {
      key += "|";
      key += piece;
    }
    for (Type ty : concreteArgTypes) {
      key += "|";
      key += typeToString(ty);
    }

    auto cached = instanceByKey.find(key);
    if (cached != instanceByKey.end())
      return cached->second;

    SmallVector<Type, 3> argTypes;
    for (Type ty : concreteArgTypes)
      argTypes.push_back(ty);

    std::string symBase = "__pto_oplib_inst_" +
                          sanitizeSymbolComponent(selected.kind) + "_" +
                          sanitizeSymbolComponent(selected.op) + "_" +
                          sanitizeSymbolComponent(selected.variantId);
    std::string sym = symBase;
    int suffix = 0;
    while (SymbolTable::lookupSymbolIn(module, sym))
      sym = symBase + "_" + std::to_string(++suffix);

    OpBuilder modBuilder(module.getContext());
    modBuilder.setInsertionPointToStart(module.getBody());
    auto inst = modBuilder.create<func::FuncOp>(
        loc, sym, FunctionType::get(module.getContext(), argTypes, {}));
    inst.setPrivate();
    inst->setAttr(kOpLibAttrInstKind,
                  StringAttr::get(module.getContext(), selected.kind));
    inst->setAttr(kOpLibAttrInstVariantId,
                  StringAttr::get(module.getContext(), selected.variantId));
    inst->setAttr(kOpLibAttrInstOp,
                  StringAttr::get(module.getContext(), selected.op));
    inst->setAttr(kOpLibAttrInstDType,
                  StringAttr::get(module.getContext(), selected.dtype));
    inst->setAttr(kOpLibAttrInstSource,
                  FlatSymbolRefAttr::get(module.getContext(),
                                         selected.entry->symbol.getSymName()));
    inst->setAttr(kOpLibAttrInstFromSeed,
                  BoolAttr::get(module.getContext(), selected.fromSeed));
    if (selected.fromSeed) {
      inst->setAttr(kOpLibAttrInstSeedId,
                    StringAttr::get(module.getContext(), selected.seedId));
      inst->setAttr(kOpLibAttrInstCoreSlot,
                    StringAttr::get(module.getContext(), selected.coreSlot));
    }
    inst->setAttr(kOpLibAttrSync,
                  BoolAttr::get(module.getContext(), selected.entry->sync));
    if (auto levelAttr =
            selected.entry->symbol->getAttrOfType<StringAttr>(kSimdLevelAttr)) {
      inst->setAttr(kSimdLevelAttr, levelAttr);
    }
    if (auto lanesAttr = selected.entry->symbol->getAttrOfType<IntegerAttr>(
            kSimdLanesAttr)) {
      inst->setAttr(kSimdLanesAttr, lanesAttr);
    }
    if (auto vldAttr = selected.entry->symbol->getAttrOfType<StringAttr>(
            kSimdVldDistAttr)) {
      inst->setAttr(kSimdVldDistAttr, vldAttr);
    }
    if (auto vstAttr = selected.entry->symbol->getAttrOfType<StringAttr>(
            kSimdVstDistAttr)) {
      inst->setAttr(kSimdVstDistAttr, vstAttr);
    }
    if (auto execModeAttr = selected.entry->symbol->getAttrOfType<StringAttr>(
            kSimdExecModeAttr)) {
      inst->setAttr(kSimdExecModeAttr, execModeAttr);
    }

    func::FuncOp source = selected.entry->symbol;
    if (source.isExternal() || source.empty() ||
        source.front().without_terminator().empty()) {
      (void)emitFailureWithCode(loc, kErrEmptyBody,
                                Twine("source template has empty body: @") +
                                    source.getSymName());
      return failure();
    }
    if (!source.getBody().hasOneBlock()) {
      (void)emitFailureWithCode(
          loc, kErrCoreSlot,
          Twine("source template must have a single entry block: @") +
              source.getSymName());
      return failure();
    }

    Block &srcEntry = source.front();
    Block *dstEntry = inst.addEntryBlock();
    OpBuilder bodyBuilder = OpBuilder::atBlockBegin(dstEntry);
    IRMapping mapping;
    for (auto [srcArg, dstArg, role] :
         llvm::zip(srcEntry.getArguments(), dstEntry->getArguments(),
                   selected.entry->argRoles)) {
      if (srcArg.getType() == dstArg.getType()) {
        mapping.map(srcArg, dstArg);
        continue;
      }

      if (role == TemplateArgRole::Scalar &&
          canRemapScalarViaCarrierCast(dstArg.getType(), srcArg.getType())) {
        auto cast = bodyBuilder.create<UnrealizedConversionCastOp>(
            loc, TypeRange{srcArg.getType()}, ValueRange{dstArg});
        mapping.map(srcArg, cast.getResult(0));
        continue;
      }

      mapping.map(srcArg, dstArg);
    }

    for (Operation &op : srcEntry.without_terminator()) {
      if (auto bridge = dyn_cast<pto::SimdTileToMemrefOp>(&op)) {
        Value mappedSrc = mapping.lookupOrNull(bridge.getSrc());
        if (!mappedSrc) {
          (void)emitFailureWithCode(
              loc, kErrCoreSlot,
              Twine(
                  "failed to remap simd.tile_to_memref source in instance @") +
                  inst.getSymName());
          return failure();
        }

        auto templateMemTy = dyn_cast<MemRefType>(bridge.getDst().getType());
        if (auto mappedTileTy =
                dyn_cast<pto::TileBufType>(mappedSrc.getType())) {
          FailureOr<MemRefType> inferredTyOr =
              inferSimdBridgeMemRefType(mappedTileTy, module.getContext());
          if (failed(inferredTyOr)) {
            (void)emitFailureWithCode(
                loc, kErrLayout,
                Twine("failed to infer simd.tile_to_memref memref type for "
                      "instance @") +
                    inst.getSymName());
            return failure();
          }

          auto inferredTy = *inferredTyOr;
          auto newBridge = bodyBuilder.create<pto::SimdTileToMemrefOp>(
              bridge.getLoc(), inferredTy, mappedSrc);
          if (templateMemTy && inferredTy != templateMemTy &&
              canRemapSimdBridgeViaCarrierCast(inferredTy, templateMemTy)) {
            auto carrierTy =
                remapMemRefToTemplateCarrier(inferredTy, templateMemTy);
            auto cast = bodyBuilder.create<UnrealizedConversionCastOp>(
                bridge.getLoc(), TypeRange{carrierTy},
                ValueRange{newBridge.getDst()});
            mapping.map(bridge.getDst(), cast.getResult(0));
          } else {
            mapping.map(bridge.getDst(), newBridge.getDst());
          }
          continue;
        }

        auto mappedMemTy = dyn_cast<MemRefType>(mappedSrc.getType());
        auto dstMemTy = templateMemTy;
        if (!mappedMemTy || !dstMemTy) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("unsupported simd.tile_to_memref remap in instance @") +
                  inst.getSymName());
          return failure();
        }

        if (mappedMemTy.getRank() != dstMemTy.getRank()) {
          (void)emitFailureWithCode(loc, kErrLayout,
                                    Twine("incompatible memref remap for "
                                          "simd.tile_to_memref in instance @") +
                                        inst.getSymName());
          return failure();
        }

        // Keep simd.tile_to_memref as a backend marker even in memref-world.
        // Use the concrete mapped memref type as the bridge result to avoid
        // re-introducing unrealized casts for shape-only differences.
        auto newBridge = bodyBuilder.create<pto::SimdTileToMemrefOp>(
            bridge.getLoc(), mappedMemTy, mappedSrc);
        if (mappedMemTy.getElementType() == dstMemTy.getElementType()) {
          mapping.map(bridge.getDst(), newBridge.getDst());
          continue;
        }
        if (!canRemapSimdBridgeViaCarrierCast(mappedMemTy, dstMemTy)) {
          (void)emitFailureWithCode(loc, kErrLayout,
                                    Twine("incompatible memref remap for "
                                          "simd.tile_to_memref in instance @") +
                                        inst.getSymName());
          return failure();
        }
        auto carrierTy = remapMemRefToTemplateCarrier(mappedMemTy, dstMemTy);
        auto cast = bodyBuilder.create<UnrealizedConversionCastOp>(
            bridge.getLoc(), TypeRange{carrierTy},
            ValueRange{newBridge.getDst()});
        mapping.map(bridge.getDst(), cast.getResult(0));
        continue;
      }

      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(&op)) {
        if (cast->getNumOperands() != 1 || cast->getNumResults() != 1) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("unsupported unrealized_conversion_cast in instance source "
                    "@") +
                  source.getSymName());
          return failure();
        }

        Value mappedSrc = mapping.lookupOrNull(cast.getOperand(0));
        if (!mappedSrc) {
          (void)emitFailureWithCode(
              loc, kErrLayout,
              Twine("failed to remap unrealized_conversion_cast source in "
                    "instance @") +
                  inst.getSymName());
          return failure();
        }

        // Source templates may contain transient conversion casts introduced by
        // earlier legalization. Treat them as transparent while cloning
        // instance bodies so concrete memref signatures do not reintroduce
        // unresolved cast chains.
        mapping.map(cast.getResult(0), mappedSrc);
        continue;
      }

      Operation *cloned = bodyBuilder.clone(op, mapping);
      for (auto [oldRes, newRes] :
           llvm::zip(op.getResults(), cloned->getResults()))
        mapping.map(oldRes, newRes);
    }
    bodyBuilder.create<func::ReturnOp>(loc);

    if (selected.fromSeed) {
      SmallVector<Operation *, 4> coreOps;
      inst.walk([&](Operation *op) {
        auto slotAttr = op->getAttrOfType<StringAttr>(kSimdCoreSlotAttr);
        if (!slotAttr)
          return;
        if (slotAttr.getValue() == selected.coreSlot)
          coreOps.push_back(op);
      });
      if (coreOps.size() != 1) {
        (void)emitFailureWithCode(
            loc, kErrCoreSlot,
            Twine("seed instance expects exactly one core slot op in @") +
                inst.getSymName());
        return failure();
      }

      Operation *core = coreOps.front();
      if (core->getNumOperands() != 2 || core->getNumResults() != 1) {
        (void)emitFailureWithCode(
            core, kErrCoreSlot,
            "core slot op must be binary op with one result");
        return failure();
      }

      OpBuilder b(core);
      Value lhs = core->getOperand(0);
      Value rhs = core->getOperand(1);
      Operation *newCore = nullptr;
      Type coreTy = lhs.getType();
      auto isIntLikeCoreType = [&]() {
        if (isa<IntegerType, IndexType>(coreTy))
          return true;
        if (auto vecTy = dyn_cast<VectorType>(coreTy))
          return isa<IntegerType>(vecTy.getElementType());
        return false;
      };

      if (selected.op == "tadd" || selected.op == "tpartadd" ||
          selected.op == "tadds" || selected.op == "trowsum") {
        if (isIntLikeCoreType())
          newCore = b.create<arith::AddIOp>(core->getLoc(), lhs, rhs);
        else
          newCore = b.create<arith::AddFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tsub" || selected.op == "tsubs") {
        if (isIntLikeCoreType())
          newCore = b.create<arith::SubIOp>(core->getLoc(), lhs, rhs);
        else
          newCore = b.create<arith::SubFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmul" || selected.op == "tmuls") {
        if (isIntLikeCoreType())
          newCore = b.create<arith::MulIOp>(core->getLoc(), lhs, rhs);
        else
          newCore = b.create<arith::MulFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tdiv") {
        if (isa<arith::DivUIOp>(core))
          newCore = b.create<arith::DivUIOp>(core->getLoc(), lhs, rhs);
        else if (isa<arith::DivSIOp>(core))
          newCore = b.create<arith::DivSIOp>(core->getLoc(), lhs, rhs);
        else
          newCore = b.create<arith::DivFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmax" || selected.op == "tpartmax" ||
                 selected.op == "tmaxs" || selected.op == "trowmax" ||
                 selected.op == "tcolmax") {
        newCore = b.create<arith::MaximumFOp>(core->getLoc(), lhs, rhs);
      } else if (selected.op == "tmin" || selected.op == "tpartmin" ||
                 selected.op == "tmins" || selected.op == "trowmin" ||
                 selected.op == "tcolmin") {
        newCore = b.create<arith::MinimumFOp>(core->getLoc(), lhs, rhs);
      } else {
        (void)emitFailureWithCode(core, kErrCoreSlot,
                                  Twine("unsupported seed target op: ") +
                                      selected.op);
        return failure();
      }
      newCore->setAttrs(core->getAttrs());
      core->replaceAllUsesWith(newCore->getResults());
      core->erase();
    }

    instanceByKey[key] = inst;

    if (debug) {
      llvm::errs() << "[op-fusion] instantiate candidate: variant_id="
                   << selected.variantId << " -> @" << inst.getSymName()
                   << "\n";
    }

    return inst;
  }
};

static FailureOr<MatchKey> buildMatchKeyFromTileBuf(pto::TileBufType dstTy) {
  if (dstTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = dstTy.getShape();
  MatchKey key;
  key.rows = shape[0] == ShapedType::kDynamic ? -1 : shape[0];
  key.cols = shape[1] == ShapedType::kDynamic ? -1 : shape[1];
  auto cfg = dstTy.getConfigAttr();
  if (!cfg)
    cfg = pto::TileBufConfigAttr::getDefault(dstTy.getContext());
  key.blayout = toBLayoutName(dstTy.getBLayoutValueI32());
  key.slayout = toSLayoutName(dstTy.getSLayoutValueI32());
  key.fractal = getFractalOrWildcard(cfg);
  return key;
}

static FailureOr<MatchKey> buildMatchKeyFromMemRef(MemRefType dstTy,
                                                   Value dst) {
  if (dstTy.getRank() != 2)
    return failure();

  ArrayRef<int64_t> shape = dstTy.getShape();
  MatchKey key;
  key.rows = shape[0] == ShapedType::kDynamic ? -1 : shape[0];
  key.cols = shape[1] == ShapedType::kDynamic ? -1 : shape[1];

  auto cfg = lookupTileConfigForValue(dst, dstTy.getContext());
  int32_t bl = 0;
  int32_t sl = 0;
  (void)readBLayoutI32(cfg.getBLayout(), bl);
  (void)readSLayoutI32(cfg.getSLayout(), sl);
  key.blayout = toBLayoutName(bl);
  key.slayout = toSLayoutName(sl);
  key.fractal = getFractalOrWildcard(cfg);
  return key;
}

static FailureOr<std::pair<Type, MatchKey>> getTileOperandInfo(Value operand) {
  Type ty = operand.getType();
  if (auto tileTy = dyn_cast<pto::TileBufType>(ty)) {
    if (tileTy.getRank() != 2 ||
        !isOplibTileElementType(tileTy.getElementType()))
      return failure();
    FailureOr<MatchKey> keyOr = buildMatchKeyFromTileBuf(tileTy);
    if (failed(keyOr))
      return failure();
    return std::make_pair(tileTy.getElementType(), *keyOr);
  }
  if (auto memTy = dyn_cast<MemRefType>(ty)) {
    if (memTy.getRank() != 2 || !isOplibTileElementType(memTy.getElementType()))
      return failure();
    FailureOr<MatchKey> keyOr = buildMatchKeyFromMemRef(memTy, operand);
    if (failed(keyOr))
      return failure();
    return std::make_pair(memTy.getElementType(), *keyOr);
  }
  return failure();
}

static LogicalResult addTileOperandToRequest(MatchRequest &request,
                                             Value operand,
                                             std::optional<Type> &elemTy) {
  FailureOr<std::pair<Type, MatchKey>> infoOr = getTileOperandInfo(operand);
  if (failed(infoOr))
    return failure();
  if (elemTy && *elemTy != infoOr->first)
    return failure();
  elemTy = infoOr->first;
  request.argTypes.push_back(operand.getType());
  request.operands.push_back(operand);
  request.argMatches.push_back(infoOr->second);
  return success();
}

static LogicalResult
addScalarOperandToRequest(MatchRequest &request, Value operand,
                          std::optional<Type> elemTy,
                          bool requireElemTypeMatch = true) {
  Type ty = operand.getType();
  if (!isOplibScalarType(ty))
    return failure();
  if (requireElemTypeMatch && elemTy && ty != *elemTy)
    return failure();
  request.argTypes.push_back(ty);
  request.operands.push_back(operand);
  request.argMatches.push_back(std::nullopt);
  return success();
}

static MatchRequest
createMatchRequest(StringRef kind, StringRef opName,
                   std::optional<int64_t> scalarPos = std::nullopt,
                   std::optional<StringRef> requiredVariantId = std::nullopt) {
  MatchRequest request;
  request.kind = kind.str();
  request.op = opName.str();
  request.scalarPos = scalarPos;
  if (requiredVariantId)
    request.requiredVariantId = requiredVariantId->str();
  return request;
}

static LogicalResult
addOperandToRequest(MatchRequest &request, Value operand, TemplateArgRole role,
                    std::optional<Type> &elemTy,
                    bool requireScalarElemTypeMatch = true) {
  if (role == TemplateArgRole::Tile)
    return addTileOperandToRequest(request, operand, elemTy);
  return addScalarOperandToRequest(request, operand, elemTy,
                                   requireScalarElemTypeMatch);
}

static FailureOr<MatchRequest>
finalizeMatchRequest(MatchRequest request, std::optional<Type> elemTy) {
  if (!elemTy)
    return failure();
  request.dtype = dtypeToString(*elemTy);
  return request;
}

static FailureOr<MatchRequest> buildTypedMatchRequest(
    StringRef kind, StringRef opName, ArrayRef<Value> operands,
    ArrayRef<TemplateArgRole> argRoles,
    std::optional<int64_t> scalarPos = std::nullopt,
    std::optional<StringRef> requiredVariantId = std::nullopt,
    bool requireScalarElemTypeMatch = true) {
  if (operands.size() != argRoles.size())
    return failure();

  MatchRequest request =
      createMatchRequest(kind, opName, scalarPos, requiredVariantId);
  std::optional<Type> elemTy;
  for (auto [operand, role] : llvm::zip(operands, argRoles)) {
    if (failed(addOperandToRequest(request, operand, role, elemTy,
                                   requireScalarElemTypeMatch)))
      return failure();
  }
  return finalizeMatchRequest(std::move(request), elemTy);
}

static void appendTileOperandUnchecked(MatchRequest &request, Value operand,
                                       const MatchKey &key) {
  request.argTypes.push_back(operand.getType());
  request.operands.push_back(operand);
  request.argMatches.push_back(key);
}

static void appendScalarOperandUnchecked(MatchRequest &request, Value operand) {
  request.argTypes.push_back(operand.getType());
  request.operands.push_back(operand);
  request.argMatches.push_back(std::nullopt);
}

static FailureOr<MatchRequest> buildCmpTileTileMatchRequest(
    StringRef kind, StringRef opName, Value src0, Value src1, Value dst,
    std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> src0InfoOr = getTileOperandInfo(src0);
  FailureOr<std::pair<Type, MatchKey>> src1InfoOr = getTileOperandInfo(src1);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(src0InfoOr) || failed(src1InfoOr) || failed(dstInfoOr))
    return failure();
  if (src0InfoOr->first != src1InfoOr->first)
    return failure();
  auto dstIntTy = dyn_cast<IntegerType>(dstInfoOr->first);
  if (!dstIntTy || dstIntTy.getWidth() != 8)
    return failure();

  MatchRequest request =
      createMatchRequest(kind, opName,
                         /*scalarPos=*/std::nullopt, requiredVariantId);
  appendTileOperandUnchecked(request, src0, src0InfoOr->second);
  appendTileOperandUnchecked(request, src1, src1InfoOr->second);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(src0InfoOr->first);
  request.maskContract = kCanonicalByteMaskContract.str();
  return request;
}

static FailureOr<MatchRequest> buildCmpTileScalarMatchRequest(
    StringRef kind, StringRef opName, Value src, Value scalar, Value dst,
    std::optional<int64_t> scalarPos = std::nullopt,
    std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> srcInfoOr = getTileOperandInfo(src);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(srcInfoOr) || failed(dstInfoOr))
    return failure();
  if (!isOplibScalarType(scalar.getType()) ||
      scalar.getType() != srcInfoOr->first)
    return failure();
  auto dstIntTy = dyn_cast<IntegerType>(dstInfoOr->first);
  if (!dstIntTy || dstIntTy.getWidth() != 8)
    return failure();

  MatchRequest request = createMatchRequest(kind, opName, scalarPos.value_or(1),
                                            requiredVariantId);
  appendTileOperandUnchecked(request, src, srcInfoOr->second);
  appendScalarOperandUnchecked(request, scalar);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(srcInfoOr->first);
  request.maskContract = kCanonicalByteMaskContract.str();
  return request;
}

static FailureOr<MatchRequest> buildSelectMaskMatchRequest(
    Operation *consumer, StringRef kind, StringRef opName, Value mask,
    Value src0, Value src1, Value dst,
    std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> maskInfoOr = getTileOperandInfo(mask);
  FailureOr<std::pair<Type, MatchKey>> src0InfoOr = getTileOperandInfo(src0);
  FailureOr<std::pair<Type, MatchKey>> src1InfoOr = getTileOperandInfo(src1);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(maskInfoOr) || failed(src0InfoOr) || failed(src1InfoOr) ||
      failed(dstInfoOr))
    return failure();
  auto maskIntTy = dyn_cast<IntegerType>(maskInfoOr->first);
  if (!maskIntTy || maskIntTy.getWidth() != 8) {
    if (consumer) {
      consumer->emitOpError(
          "select mask must use the approved compare/select byte-mask "
          "contract with 8-bit mask lanes");
    }
    return failure();
  }
  if (src0InfoOr->first != src1InfoOr->first ||
      src0InfoOr->first != dstInfoOr->first)
    return failure();

  MatchRequest request =
      createMatchRequest(kind, opName,
                         /*scalarPos=*/std::nullopt, requiredVariantId);
  appendTileOperandUnchecked(request, mask, maskInfoOr->second);
  appendTileOperandUnchecked(request, src0, src0InfoOr->second);
  appendTileOperandUnchecked(request, src1, src1InfoOr->second);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(src0InfoOr->first);
  request.maskContract = kCanonicalByteMaskContract.str();
  return request;
}

static FailureOr<MatchRequest> buildSelectScalarMatchRequest(
    StringRef kind, StringRef opName, Value src0, Value src1, Value selectMode,
    Value dst, std::optional<int64_t> scalarPos = std::nullopt,
    std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> src0InfoOr = getTileOperandInfo(src0);
  FailureOr<std::pair<Type, MatchKey>> src1InfoOr = getTileOperandInfo(src1);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(src0InfoOr) || failed(src1InfoOr) || failed(dstInfoOr))
    return failure();
  if (!isOplibScalarType(selectMode.getType()))
    return failure();
  if (src0InfoOr->first != src1InfoOr->first ||
      src0InfoOr->first != dstInfoOr->first)
    return failure();

  MatchRequest request = createMatchRequest(kind, opName, scalarPos.value_or(2),
                                            requiredVariantId);
  appendTileOperandUnchecked(request, src0, src0InfoOr->second);
  appendTileOperandUnchecked(request, src1, src1InfoOr->second);
  appendScalarOperandUnchecked(request, selectMode);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(src0InfoOr->first);
  return request;
}

static bool isKnownStaticDim(int64_t dim) { return dim >= 0; }

static bool isFullTileRole(const MatchKey &src, const MatchKey &dst) {
  return isKnownStaticDim(src.rows) && isKnownStaticDim(src.cols) &&
         isKnownStaticDim(dst.rows) && isKnownStaticDim(dst.cols) &&
         src.rows == dst.rows && src.cols == dst.cols;
}

static bool isRowBroadcastRole(const MatchKey &src, const MatchKey &dst) {
  return isKnownStaticDim(src.rows) && isKnownStaticDim(src.cols) &&
         isKnownStaticDim(dst.rows) && src.rows == dst.rows && src.cols == 1;
}

static bool isRowSourceRole(const MatchKey &src, const MatchKey &dst) {
  return isRowBroadcastRole(src, dst) || isFullTileRole(src, dst);
}

static std::optional<std::pair<int64_t, int64_t>>
inferRowBroadcastOperandPositions(const MatchKey &src0, const MatchKey &src1,
                                  const MatchKey &dst) {
  const bool src0Full = isFullTileRole(src0, dst);
  const bool src1Full = isFullTileRole(src1, dst);
  const bool src0Row = isRowBroadcastRole(src0, dst);
  const bool src1Row = isRowBroadcastRole(src1, dst);
  if (src0Full && src1Row)
    return std::make_pair(0, 1);
  if (src1Full && src0Row)
    return std::make_pair(1, 0);
  if (src0Full && src1Full)
    return std::make_pair(0, 1);
  return std::nullopt;
}

static FailureOr<MatchRequest> buildRowBroadcastBinaryMatchRequest(
    StringRef kind, StringRef opName, Value src0, Value src1, Value dst,
    std::optional<int64_t> fullTilePos = std::nullopt,
    std::optional<int64_t> rowBroadcastPos = std::nullopt,
    std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> src0InfoOr = getTileOperandInfo(src0);
  FailureOr<std::pair<Type, MatchKey>> src1InfoOr = getTileOperandInfo(src1);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(src0InfoOr) || failed(src1InfoOr) || failed(dstInfoOr))
    return failure();
  if (src0InfoOr->first != src1InfoOr->first ||
      src0InfoOr->first != dstInfoOr->first)
    return failure();

  const std::pair<Type, MatchKey> srcInfos[] = {*src0InfoOr, *src1InfoOr};
  Value srcValues[] = {src0, src1};
  if (!fullTilePos || !rowBroadcastPos) {
    auto inferred = inferRowBroadcastOperandPositions(
        srcInfos[0].second, srcInfos[1].second, dstInfoOr->second);
    if (!inferred)
      return failure();
    fullTilePos = inferred->first;
    rowBroadcastPos = inferred->second;
  }
  if (*fullTilePos == *rowBroadcastPos || *fullTilePos < 0 ||
      *fullTilePos > 1 || *rowBroadcastPos < 0 || *rowBroadcastPos > 1)
    return failure();
  if (!isFullTileRole(srcInfos[*fullTilePos].second, dstInfoOr->second) ||
      !isRowSourceRole(srcInfos[*rowBroadcastPos].second, dstInfoOr->second))
    return failure();

  MatchRequest request =
      createMatchRequest(kind, opName,
                         /*scalarPos=*/std::nullopt, requiredVariantId);
  appendTileOperandUnchecked(request, srcValues[*fullTilePos],
                             srcInfos[*fullTilePos].second);
  appendTileOperandUnchecked(request, srcValues[*rowBroadcastPos],
                             srcInfos[*rowBroadcastPos].second);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(src0InfoOr->first);
  request.fullTilePos = *fullTilePos;
  request.rowBroadcastPos = *rowBroadcastPos;
  return request;
}

static FailureOr<MatchRequest> buildPReluMatchRequest(
    StringRef kind, StringRef opName, Value src0, Value src1, Value tmp,
    Value dst, std::optional<StringRef> requiredVariantId = std::nullopt) {
  FailureOr<std::pair<Type, MatchKey>> src0InfoOr = getTileOperandInfo(src0);
  FailureOr<std::pair<Type, MatchKey>> src1InfoOr = getTileOperandInfo(src1);
  FailureOr<std::pair<Type, MatchKey>> tmpInfoOr = getTileOperandInfo(tmp);
  FailureOr<std::pair<Type, MatchKey>> dstInfoOr = getTileOperandInfo(dst);
  if (failed(src0InfoOr) || failed(src1InfoOr) || failed(tmpInfoOr) ||
      failed(dstInfoOr))
    return failure();

  if (src0InfoOr->first != src1InfoOr->first ||
      src0InfoOr->first != dstInfoOr->first)
    return failure();
  if (!isa<FloatType>(src0InfoOr->first))
    return failure();
  auto tmpIntTy = dyn_cast<IntegerType>(tmpInfoOr->first);
  if (!tmpIntTy || tmpIntTy.getWidth() != 8)
    return failure();

  MatchRequest request =
      createMatchRequest(kind, opName,
                         /*scalarPos=*/std::nullopt, requiredVariantId);
  appendTileOperandUnchecked(request, src0, src0InfoOr->second);
  appendTileOperandUnchecked(request, src1, src1InfoOr->second);
  appendTileOperandUnchecked(request, tmp, tmpInfoOr->second);
  appendTileOperandUnchecked(request, dst, dstInfoOr->second);
  request.dtype = dtypeToString(src0InfoOr->first);
  return request;
}

static FailureOr<MatchRequest> buildMatchRequestFromInterface(Operation *op) {
  auto iface = dyn_cast<pto::OpLibOpInterface>(op);
  if (!iface)
    return failure();

  FailureOr<pto::OpLibMatchDescriptor> descOr = iface.getOpLibMatchDescriptor();
  if (failed(descOr))
    return failure();

  const pto::OpLibMatchDescriptor &desc = *descOr;
  if (desc.operands.size() != desc.operandRoles.size())
    return failure();

  SmallVector<TemplateArgRole, 4> argRoles;
  argRoles.reserve(desc.operandRoles.size());
  for (int64_t role : desc.operandRoles) {
    switch (static_cast<pto::OpLibArgRole>(role)) {
    case pto::OpLibArgRole::Tile:
      argRoles.push_back(TemplateArgRole::Tile);
      break;
    case pto::OpLibArgRole::Scalar:
      argRoles.push_back(TemplateArgRole::Scalar);
      break;
    default:
      return failure();
    }
  }

  std::optional<StringRef> requiredVariantId = std::nullopt;
  if (desc.requiredVariantId)
    requiredVariantId = StringRef(*desc.requiredVariantId);

  FailureOr<MatchRequest> requestOr = failure();
  if (desc.kind == "l3_cmp_tile_tile_template" && desc.operands.size() == 3) {
    requestOr = buildCmpTileTileMatchRequest(
        desc.kind, desc.opName, desc.operands[0], desc.operands[1],
        desc.operands[2], requiredVariantId);
  } else if (desc.kind == "l3_cmp_tile_scalar_template" &&
             desc.operands.size() == 3) {
    requestOr = buildCmpTileScalarMatchRequest(
        desc.kind, desc.opName, desc.operands[0], desc.operands[1],
        desc.operands[2], desc.scalarPos, requiredVariantId);
  } else if (desc.kind == "l3_select_mask_template" &&
             desc.operands.size() == 4) {
    requestOr = buildSelectMaskMatchRequest(
        op, desc.kind, desc.opName, desc.operands[0], desc.operands[1],
        desc.operands[2], desc.operands[3], requiredVariantId);
  } else if (desc.kind == "l3_select_scalar_template" &&
             desc.operands.size() == 4) {
    requestOr = buildSelectScalarMatchRequest(
        desc.kind, desc.opName, desc.operands[0], desc.operands[1],
        desc.operands[2], desc.operands[3], desc.scalarPos, requiredVariantId);
  } else if (desc.kind == "l3_broadcast_row_binary_template" &&
             desc.operands.size() == 3) {
    requestOr = buildRowBroadcastBinaryMatchRequest(
        desc.kind, desc.opName, desc.operands[0], desc.operands[1],
        desc.operands[2], desc.fullTilePos, desc.rowBroadcastPos,
        requiredVariantId);
    if (failed(requestOr)) {
      op->emitOpError("broadcast_row_binary family requires at least one "
                      "dst-shaped input and a row-source operand before "
                      "template matching");
      return failure();
    }
  } else if (desc.kind == "l3_float_ternary_tile_template" &&
             desc.opName == "tprelu" && desc.operands.size() == 4) {
    requestOr = buildPReluMatchRequest(desc.kind, desc.opName, desc.operands[0],
                                       desc.operands[1], desc.operands[2],
                                       desc.operands[3], requiredVariantId);
  } else {
    requestOr =
        buildTypedMatchRequest(desc.kind, desc.opName, desc.operands, argRoles,
                               desc.scalarPos, requiredVariantId);
  }
  if (failed(requestOr))
    return failure();

  if (desc.cmpMode)
    requestOr->cmpMode = *desc.cmpMode;
  if (desc.maskContract)
    requestOr->maskContract = *desc.maskContract;
  if (desc.operandOrder)
    requestOr->operandOrder = *desc.operandOrder;
  if (desc.fullTilePos)
    requestOr->fullTilePos = *desc.fullTilePos;
  if (desc.rowBroadcastPos)
    requestOr->rowBroadcastPos = *desc.rowBroadcastPos;
  if (desc.isBinary)
    requestOr->isBinary = *desc.isBinary;
  return *requestOr;
}

static FailureOr<MatchRequest> buildMatchRequest(Operation *op) {
  return buildMatchRequestFromInterface(op);
}

static LogicalResult validateFamilyLogicRequest(const MatchRequest &request,
                                                Location loc) {
  if ((request.kind == "l3_cmp_tile_tile_template" ||
       request.kind == "l3_cmp_tile_scalar_template" ||
       request.kind == "l3_select_mask_template") &&
      (!request.maskContract ||
       *request.maskContract != kCanonicalByteMaskContract)) {
    mlir::emitError(loc)
        << "compare/select family logic requires mask_contract="
        << kCanonicalByteMaskContract;
    return failure();
  }

  if (request.op == "tdivs") {
    if (!request.operandOrder || (*request.operandOrder != "tile_scalar" &&
                                  *request.operandOrder != "scalar_tile")) {
      mlir::emitError(loc)
          << "tile_scalar family logic lost operand_order before template "
             "normalization";
      return failure();
    }
  }

  if (request.kind == "l3_broadcast_row_binary_template") {
    if (!request.fullTilePos || !request.rowBroadcastPos ||
        *request.fullTilePos == *request.rowBroadcastPos) {
      mlir::emitError(loc)
          << "broadcast_row_binary family logic lost full-tile vs "
             "row-source roles before matcher selection";
      return failure();
    }
  }

  if (request.kind == "l3_reduce_colsum_template") {
    if (!request.isBinary || !request.requiredVariantId) {
      mlir::emitError(loc)
          << "reduce_colsum family logic requires both isBinary and "
             "requiredVariantId";
      return failure();
    }
    StringRef expectedVariant = *request.isBinary ? "binary" : "linear";
    if (*request.requiredVariantId != expectedVariant) {
      mlir::emitError(loc)
          << "reduce_colsum family logic mismatch: expected variant_id="
          << expectedVariant
          << " for isBinary=" << (*request.isBinary ? "true" : "false");
      return failure();
    }
  }

  return success();
}

static LogicalResult
validateSelectedFamilyLogic(const MatchRequest &request,
                            const SelectedVariant &selected, Location loc) {
  if (request.requiredVariantId &&
      selected.variantId != *request.requiredVariantId) {
    mlir::emitError(loc) << "template selection collapsed required variant_id="
                         << *request.requiredVariantId << " to "
                         << selected.variantId;
    return failure();
  }

  if (request.op == "tdivs" && request.operandOrder &&
      !StringRef(selected.variantId).starts_with(*request.operandOrder)) {
    mlir::emitError(loc) << "tile_scalar family logic collapsed operand_order="
                         << *request.operandOrder
                         << " to variant_id=" << selected.variantId;
    return failure();
  }

  if (request.kind == "l3_reduce_colsum_template" && request.isBinary) {
    StringRef expectedVariant = *request.isBinary ? "binary" : "linear";
    if (selected.variantId != expectedVariant) {
      mlir::emitError(loc)
          << "reduce_colsum variant semantics collapsed to variant_id="
          << selected.variantId << " while expecting " << expectedVariant;
      return failure();
    }
  }

  return success();
}

static FailureOr<PlannedOpLowering>
planOneOpLowering(Operation *op, const A5OpLibManifest &manifest,
                  TemplateRegistry &registry, StringRef errorPrefix,
                  bool enableDebugLog = false) {
  StringRef opName = getPTOMnemonic(op);
  const A5ManifestEntry *manifestEntry = manifest.lookup(opName);
  if (!manifestEntry) {
    op->emitError() << errorPrefix << ": op=" << opName
                    << " is outside the A5 OpLib V1 manifest scope";
    return failure();
  }

  if (manifestEntry->status == A5ManifestStatus::Deferred) {
    if (enableDebugLog) {
      llvm::errs() << "[op-fusion] skip deferred manifest op from OP-Lib path: "
                   << formatManifestEntrySummary(opName, *manifestEntry)
                   << "\n";
    }
    return failure();
  }

  FailureOr<MatchRequest> requestOr = buildMatchRequest(op);
  if (failed(requestOr)) {
    op->emitError() << errorPrefix << ": manifest-implemented "
                    << formatManifestEntrySummary(opName, *manifestEntry)
                    << " failed to build MatchRequest from interface metadata";
    return failure();
  }

  if (enableDebugLog) {
    llvm::errs() << "[oplib-lowering] match path=interface op=" << op->getName()
                 << "\n";
  }

  const MatchRequest &request = *requestOr;
  if (failed(validateFamilyLogicRequest(request, op->getLoc()))) {
    op->emitError() << errorPrefix
                    << ": family-logic request validation failed for "
                    << formatManifestEntrySummary(opName, *manifestEntry);
    return failure();
  }

  FailureOr<SelectedVariant> selectedOr =
      registry.selectVariantFor(request, op->getLoc());
  if (failed(selectedOr)) {
    op->emitError() << errorPrefix << ": manifest-implemented "
                    << formatManifestEntrySummary(opName, *manifestEntry)
                    << " has no OP-Lib candidate for op=" << request.op
                    << " dtype=" << request.dtype;
    return failure();
  }

  if (failed(validateSelectedFamilyLogic(request, *selectedOr, op->getLoc()))) {
    op->emitError() << errorPrefix
                    << ": family-logic selection validation failed for "
                    << formatManifestEntrySummary(opName, *manifestEntry);
    return failure();
  }

  FailureOr<func::FuncOp> instanceOr =
      registry.getOrCreateInstance(*selectedOr, request.argTypes, op->getLoc());
  if (failed(instanceOr)) {
    op->emitError()
        << errorPrefix
        << ": failed to instantiate OP-Lib candidate for manifest-implemented "
        << formatManifestEntrySummary(opName, *manifestEntry)
        << " op=" << request.op << " dtype=" << request.dtype;
    return failure();
  }

  PlannedOpLowering planned;
  planned.op = op;
  planned.selected = *selectedOr;
  planned.operands.assign(request.operands.begin(), request.operands.end());
  planned.instance = *instanceOr;
  return planned;
}

static LogicalResult
rewriteOneGroupedOpAsCall(const PlannedOpLowering &planned) {
  Operation *op = planned.op;
  OpBuilder builder(op);
  auto call = builder.create<func::CallOp>(op->getLoc(), planned.instance,
                                           planned.operands);

  if (auto gid = op->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr))
    call->setAttr(kFusionGroupIdAttr, gid);
  if (auto order = op->getAttrOfType<IntegerAttr>(kFusionOrderAttr))
    call->setAttr(kFusionOrderAttr, order);

  op->erase();
  return success();
}

static void pruneUnusedOplibFuncs(ModuleOp module, bool debug) {
  auto hasDirectCallUse = [&](func::FuncOp fn) -> bool {
    bool used = false;
    StringRef sym = fn.getSymName();
    module.walk([&](func::CallOp call) {
      if (call.getCallee() == sym) {
        used = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return used;
  };

  SmallVector<func::FuncOp, 16> funcsToErase;
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    if (!func.isPrivate())
      continue;

    const bool isOplibEntry =
        func->hasAttr(kOpLibAttrEntryRole) || func->hasAttr(kOpLibAttrKind);
    const bool isOplibInstance = func->hasAttr(kOpLibAttrInstVariantId);
    if (!(isOplibEntry || isOplibInstance))
      continue;
    if (hasDirectCallUse(func))
      continue;
    funcsToErase.push_back(func);
  }

  for (func::FuncOp func : funcsToErase)
    func.erase();

  if (debug && !funcsToErase.empty()) {
    llvm::errs() << "[op-fusion] pruned " << funcsToErase.size()
                 << " unused OP-Lib function(s)\n";
  }
}

struct PTOLowerToOpLibCallsPass
    : public pto::impl::PTOInstantiateAndLowerToLibCallBase<
          PTOLowerToOpLibCallsPass> {
  using pto::impl::PTOInstantiateAndLowerToLibCallBase<
      PTOLowerToOpLibCallsPass>::PTOInstantiateAndLowerToLibCallBase;

  LogicalResult collectGroupedLoweringUnitFromFusionRegion(
      pto::FusionRegionOp fusionRegion, const A5OpLibManifest &manifest,
      SmallVectorImpl<GroupInfo> &groups) {
    GroupInfo group;
    group.region = fusionRegion;
    if (auto gidAttr =
            fusionRegion->getAttrOfType<IntegerAttr>(kFusionGroupIdAttr))
      group.groupId = gidAttr.getInt();

    SmallVector<Operation *, 8> unsupportedComputeOps;
    fusionRegion.walk([&](Operation *op) {
      if (op == fusionRegion.getOperation())
        return WalkResult::advance();
      if (isa<pto::YieldOp>(op))
        return WalkResult::advance();
      if (auto nestedFusionRegion = dyn_cast<pto::FusionRegionOp>(op);
          nestedFusionRegion && nestedFusionRegion != fusionRegion)
        return WalkResult::skip();

      if (!isa<pto::OpLibOpInterface>(op))
        return WalkResult::advance();

      if (!shouldLowerViaOpLib(op, manifest)) {
        unsupportedComputeOps.push_back(op);
        return WalkResult::advance();
      }

      group.ops.push_back(op);
      return WalkResult::advance();
    });

    if (!group.ops.empty() && !unsupportedComputeOps.empty()) {
      InFlightDiagnostic diag = fusionRegion.emitError(
          "fusion_region contains partially-supported OpLib lowering scope");
      if (group.groupId >= 0)
        diag << " for group_id=" << group.groupId;
      diag << ": active grouped lowering cannot mix supported compute op '"
           << getPTOMnemonic(group.ops.front()) << "' with unsupported compute "
           << "op '" << getPTOMnemonic(unsupportedComputeOps.front()) << "'";
      return failure();
    }

    if (!group.ops.empty())
      groups.push_back(std::move(group));
    return success();
  }

  LogicalResult collectGroupedLoweringUnitsInRegion(
      Region &region, const A5OpLibManifest &manifest,
      SmallVectorImpl<GroupInfo> &groups) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        Operation *curOp = &op;

        if (auto fusionRegion = dyn_cast<pto::FusionRegionOp>(curOp)) {
          if (failed(collectGroupedLoweringUnitFromFusionRegion(
                  fusionRegion, manifest, groups)))
            return failure();
          continue;
        }

        for (Region &nested : curOp->getRegions())
          if (failed(
                  collectGroupedLoweringUnitsInRegion(nested, manifest, groups)))
            return failure();
      }
    }
    return success();
  }

  LogicalResult
  planGroupLowerings(GroupInfo &group, const A5OpLibManifest &manifest,
                     TemplateRegistry &registry,
                     SmallVectorImpl<PlannedOpLowering> &planned) {
    for (Operation *op : group.ops) {
      FailureOr<PlannedOpLowering> oneOr = planOneOpLowering(
          op, manifest, registry, "fusion-group lowering", debug);
      if (failed(oneOr))
        return failure();
      planned.push_back(*oneOr);
    }
    return success();
  }

  LogicalResult lowerOneGroup(const A5OpLibManifest &manifest,
                              TemplateRegistry &registry, GroupInfo &group) {
    if (group.ops.empty())
      return success();

    Operation *anchor =
        group.region ? group.region.getOperation() : group.ops.front();

    SmallVector<PlannedOpLowering, 8> planned;
    if (failed(planGroupLowerings(group, manifest, registry, planned))) {
      InFlightDiagnostic diag =
          anchor->emitError("fusion-region lowering required but failed");
      if (group.groupId >= 0)
        diag << " for group_id=" << group.groupId;
      return failure();
    }

    for (const PlannedOpLowering &plan : planned) {
      if (failed(rewriteOneGroupedOpAsCall(plan))) {
        anchor->emitError()
            << "fusion-region lowering required but failed to rewrite selected "
               "OP-Lib call(s)";
        return failure();
      }
    }

    if (debug) {
      llvm::errs() << "[op-fusion] lowered fusion_region";
      if (group.groupId >= 0)
        llvm::errs() << " group_id=" << group.groupId;
      llvm::errs() << " to OP-Lib calls (" << planned.size() << " op(s))\n";
    }

    return success();
  }

  LogicalResult lowerSingleOps(func::FuncOp func,
                               const A5OpLibManifest &manifest,
                               TemplateRegistry &registry) {
    SmallVector<Operation *, 32> toRewrite;

    func.walk([&](Operation *op) {
      if (op->getParentOfType<pto::FusionRegionOp>())
        return;
      if (!isA5OpLibV1TargetOp(op, manifest))
        return;
      if (!shouldLowerViaOpLib(op, manifest))
        return;
      if (hasFusionGroupAttrs(op))
        return;
      toRewrite.push_back(op);
    });

    for (Operation *op : toRewrite) {
      if (!op || !op->getBlock())
        continue;

      FailureOr<PlannedOpLowering> plannedOr = planOneOpLowering(
          op, manifest, registry, "single-op lowering", debug);
      if (failed(plannedOr))
        return failure();

      OpBuilder builder(op);
      builder.create<func::CallOp>(op->getLoc(), plannedOr->instance,
                                   plannedOr->operands);
      op->erase();

      if (debug) {
        llvm::errs() << "[op-fusion] materialized single op="
                     << plannedOr->selected.op << " in @" << func.getSymName()
                     << "\n";
      }
    }

    return success();
  }

  LogicalResult verifyNoRemainingTargetOps(func::FuncOp func,
                                           const A5OpLibManifest &manifest) {
    bool hasFailure = false;
    func.walk([&](Operation *op) {
      const A5ManifestEntry *manifestEntry =
          manifest.lookup(getPTOMnemonic(op));
      if (!manifestEntry ||
          manifestEntry->status != A5ManifestStatus::Implemented ||
          !shouldLowerViaOpLib(op, manifest))
        return;
      op->emitError() << "OP-Lib lowering is required for manifest-implemented "
                         "PTO IR 4.5~4.9 "
                         "op, but this op was not lowered: "
                      << formatManifestEntrySummary(getPTOMnemonic(op),
                                                    *manifestEntry);
      hasFailure = true;
    });
    return failure(hasFailure);
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    SymbolTable symbolTable(module);

    std::string manifestPath =
        resolveA5OpLibManifestPath(opLibManifest, opLibDir, module.getLoc());
    if (manifestPath.empty()) {
      emitError(module.getLoc())
          << "failed to resolve A5 OpLib V1 manifest path; set "
             "--op-lib-manifest "
             "or keep oplib/level3/families/a5_oplib_v1_manifest.yaml "
             "available";
      signalPassFailure();
      return;
    }

    llvm::Expected<A5OpLibManifest> manifestOrErr =
        loadA5OpLibManifest(manifestPath);
    if (!manifestOrErr) {
      emitError(module.getLoc()) << llvm::toString(manifestOrErr.takeError());
      signalPassFailure();
      return;
    }
    A5OpLibManifest manifest = std::move(*manifestOrErr);

    TemplateRegistry registry{module, symbolTable, debug};
    int importedFromModule = 0;
    if (failed(registry.loadFromModuleEntries(module.getLoc(),
                                              importedFromModule))) {
      signalPassFailure();
      return;
    }
    if (importedFromModule == 0) {
      if (failed(registry.loadFromDir(opLibDir, ctx, module.getLoc()))) {
        signalPassFailure();
        return;
      }
    } else if (debug) {
      llvm::errs() << "[op-fusion] using " << importedFromModule
                   << " pre-imported OP-Lib entries from module\n";
      llvm::errs() << "[op-fusion] loaded A5 OpLib manifest from "
                   << manifestPath << "\n";
    }

    for (func::FuncOp func : module.getOps<func::FuncOp>()) {
      if (func.isExternal())
        continue;
      if (func->hasAttr(kOpLibAttrKind) ||
          func->hasAttr(kOpLibAttrInstVariantId))
        continue;
      if (func.getSymName().starts_with("__pto_oplib_"))
        continue;

      SmallVector<GroupInfo, 8> groups;
      if (failed(
              collectGroupedLoweringUnitsInRegion(func.getRegion(), manifest,
                                                  groups))) {
        signalPassFailure();
        return;
      }
      if (debug && !groups.empty()) {
        llvm::errs() << "[op-fusion] found " << groups.size()
                     << " fusion_region lowering unit(s) in @"
                     << func.getSymName() << "\n";
      }

      for (GroupInfo &group : groups) {
        if (failed(lowerOneGroup(manifest, registry, group))) {
          signalPassFailure();
          return;
        }
      }

      if (failed(lowerSingleOps(func, manifest, registry))) {
        signalPassFailure();
        return;
      }

      if (failed(verifyNoRemainingTargetOps(func, manifest))) {
        signalPassFailure();
        return;
      }
    }

    pruneUnusedOplibFuncs(module, debug);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::pto::createPTOInstantiateAndLowerToLibCallPass(
    const PTOInstantiateAndLowerToLibCallOptions &options) {
  return std::make_unique<PTOLowerToOpLibCallsPass>(options);
}

LogicalResult mlir::pto::importPTOOpLibTemplates(ModuleOp module,
                                                 StringRef opLibDir,
                                                 bool debug) {
  SymbolTable symbolTable(module);
  TemplateRegistry registry{module, symbolTable, debug};
  return registry.loadFromDir(opLibDir, module.getContext(), module.getLoc());
}
