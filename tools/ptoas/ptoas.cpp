//===- ptoas.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "PTO/Transforms/Passes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/EmitC/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Target/Cpp/CppEmitter.h"
#include "ptobc/ptobc_decode.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <memory>
#include <string>

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

static LogicalResult reorderEmitCFunctions(ModuleOp module) {
  SmallVector<emitc::FuncOp> declarations;
  SmallVector<emitc::FuncOp> definitions;
  llvm::DenseMap<StringAttr, emitc::FuncOp> definitionsByName;

  for (auto func : module.getOps<emitc::FuncOp>()) {
    if (func.isDeclaration()) {
      declarations.push_back(func);
      continue;
    }
    definitions.push_back(func);
    definitionsByName[func.getSymNameAttr()] = func;
  }

  llvm::DenseMap<Operation *, unsigned> indegree;
  llvm::DenseMap<Operation *, SmallVector<Operation *>> outgoing;
  for (auto func : definitions)
    indegree[func.getOperation()] = 0;

  for (auto caller : definitions) {
    Operation *callerOp = caller.getOperation();
    llvm::SmallPtrSet<Operation *, 8> seenCallees;
    bool hasCycle = false;
    caller.walk([&](emitc::CallOp call) {
      auto calleeAttr = call.getCalleeAttr();
      if (!calleeAttr)
        return;
      auto it = definitionsByName.find(calleeAttr.getLeafReference());
      if (it == definitionsByName.end())
        return;
      Operation *calleeOp = it->second.getOperation();
      if (calleeOp == callerOp) {
        hasCycle = true;
        return;
      }
      if (!seenCallees.insert(calleeOp).second)
        return;
      outgoing[calleeOp].push_back(callerOp);
      ++indegree[callerOp];
    });
    if (hasCycle) {
      return caller.emitOpError()
             << "recursive function calls are not supported for EmitC C++ "
                "emission";
    }
  }

  SmallVector<Operation *> ready;
  for (auto func : definitions) {
    if (indegree[func.getOperation()] == 0)
      ready.push_back(func.getOperation());
  }

  SmallVector<emitc::FuncOp> sortedDefinitions;
  while (!ready.empty()) {
    Operation *next = ready.front();
    ready.erase(ready.begin());
    auto nextFunc = cast<emitc::FuncOp>(next);
    sortedDefinitions.push_back(nextFunc);

    for (Operation *user : outgoing[next]) {
      unsigned &userIndegree = indegree[user];
      if (--userIndegree == 0)
        ready.push_back(user);
    }
  }

  if (sortedDefinitions.size() != definitions.size()) {
    return module.emitError() << "cyclic function call graph is not supported "
                                 "for EmitC C++ emission";
  }

  if (declarations.empty() && definitions.size() <= 1)
    return success();

  SmallVector<emitc::FuncOp> desiredOrder;
  desiredOrder.append(declarations.begin(), declarations.end());
  desiredOrder.append(sortedDefinitions.begin(), sortedDefinitions.end());

  Block &body = module.getBodyRegion().front();
  Operation *anchor = nullptr;
  for (Operation &op : body.getOperations()) {
    if (isa<emitc::FuncOp>(op)) {
      anchor = &op;
      break;
    }
  }
  if (!anchor)
    return success();

  auto advanceAnchor = [&]() {
    while (anchor) {
      anchor = anchor->getNextNode();
      if (!anchor || isa<emitc::FuncOp>(anchor))
        return;
    }
  };

  for (auto func : desiredOrder) {
    if (func.getOperation() == anchor) {
      advanceAnchor();
      continue;
    }
    if (anchor)
      func->moveBefore(anchor);
    else
      func->moveBefore(&body, body.end());
  }

  return success();
}

// #define ADD_CANONICALIZER_PASS \
//    CanonicalizerOptions options; \
//    options.enableExtendedPattern = true; \
//    std::vector<std::string> disabledPatterns{}; \
//    options.disabledPatterns = disabledPatterns; \
//    pm.addPass(createCanonicalizerPass(options))

// #define ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS \
//    pm.nest<func::FuncOp>().addPass(createCanonicalizerPass(options))

// static void canonicalizationPipeline(OpPassManager &pm) {
//    pm.addPass(createArithToAffineConversionPass());
//    ADD_CANONICALIZER_PASS;
//    pm.addPass(createSCFForLoopCanonicalizationPass());
//    pm.addPass(createCSEPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    //pm.nest<func::FuncOp>().addPass(createHIVMOptSinglePointPass());
//    ADD_CANONICALIZER_PASS_WITHOUT_OPTION_DEFS;
//    pm.nest<func::FuncOp>().addPass(memref::createDeadStoreEliminationPass());
// }

static void bufferizationPipeline(OpPassManager &pm) {
  bufferization::OneShotBufferizationOptions oneShotOptions;
  oneShotOptions.bufferizeFunctionBoundaries = true;
  oneShotOptions.setFunctionBoundaryTypeConversion(
      bufferization::LayoutMapOption::IdentityLayoutMap);
  oneShotOptions.allowReturnAllocsFromLoops = true;
  oneShotOptions.allowUnknownOps = true;
  pm.addPass(bufferization::createOneShotBufferizePass(oneShotOptions));
  // pm.addPass(bufferization::createOneShotBufferizePass());

  // if (hivmPipelineOptions.enableVfMerge) {
  //    pm.addPass(hfusion::createMergeVecScopePass());
  // }
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  // canonicalizationPipeline(pm);
  // pm.addPass(bufferization::createDropEquivalentBufferResultsPass());
  pm.addPass(createConvertToPTOOpPass());
}

// --------------------------------------------------------------------------
// Command Line Options
// --------------------------------------------------------------------------
static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string>
    outputFilename("o", llvm::cl::desc("Output filename"),
                   llvm::cl::value_desc("filename"), llvm::cl::init("-"));

static llvm::cl::opt<bool> enableInsertSync(
    "enable-insert-sync",
    llvm::cl::desc("Enable automatic synchronization insertion pass"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> enableOpFusion(
    "enable-op-fusion",
    llvm::cl::desc("Enable OP fusion passes (A5 only: "
                   "create-groups/outline/low-level-loop-fusion)"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string>
    opLibDir("op-lib-dir",
             llvm::cl::desc("Directory containing OP-Lib template .mlir files "
                            "for OP-LIB lowering (A5 only)"),
             llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    opFusionDebug("op-fusion-debug",
                  llvm::cl::desc("Enable verbose debug logs for OP fusion "
                                 "(grouping/materialization/loop fusion)"),
                  llvm::cl::init(false));

static llvm::cl::opt<bool> printIRAfterAll(
    "print-ir-after-all",
    llvm::cl::desc("Print MLIR IR after each pass in all PTOAS pass pipelines "
                   "for user-related functions"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> printIRAfterAllFuncFilter(
    "print-ir-after-all-func-filter",
    llvm::cl::desc("When --print-ir-after-all is enabled, only print dumps for "
                   "func.func whose symbol name contains this substring "
                   "(overrides the default user-related filtering)"),
    llvm::cl::value_desc("substring"), llvm::cl::init(""));

static llvm::cl::opt<bool> disableInferLayout(
    "disable-infer-layout",
    llvm::cl::desc("Disable PTO layout inference pass (static-only)"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> emitAddPtrTrace(
    "emit-addptr-trace",
    llvm::cl::desc("Emit addptr trace comments in generated C++ output"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> ptoTargetArch(
    "pto-arch",
    llvm::cl::desc(
        "Target Ascend architecture for codegen: a3 or a5 (default: a3)"),
    llvm::cl::value_desc("a3|a5"), llvm::cl::init("a3"));

static llvm::cl::opt<std::string>
    ptoBuildLevel("pto-level",
                  llvm::cl::desc("Build level for pass pipeline: level1, "
                                 "level2, or level3 (default: level2)"),
                  llvm::cl::value_desc("level1|level2|level3"),
                  llvm::cl::init("level2"));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

enum class PTOTargetArch {
  A3,
  A5,
};

static std::string asciiLowercaseCopy(llvm::StringRef text) {
  std::string lowered = text.str();
  for (char &c : lowered)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return lowered;
}

static PTOBuildLevel defaultBuildLevel() { return PTOBuildLevel::Level2; }

static bool parseBuildLevel(llvm::StringRef levelStr, PTOBuildLevel &out) {
  std::string s = asciiLowercaseCopy(levelStr);
  if (s == "level1") {
    out = PTOBuildLevel::Level1;
    return true;
  }
  if (s == "level2") {
    out = PTOBuildLevel::Level2;
    return true;
  }
  if (s == "level3") {
    out = PTOBuildLevel::Level3;
    return true;
  }
  return false;
}

static constexpr llvm::StringLiteral kAutoSyncTailPolicyBarrierAll =
    "barrier_all";
static constexpr llvm::StringLiteral kAutoSyncTailPolicyMte3ToSEvent0 =
    "setwait_mte3_to_s_event0";

static bool parseAutoSyncTailHint(llvm::StringRef hintStr,
                                  std::string &normalized) {
  std::string s = hintStr.str();
  for (char &c : s)
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "barrier-all" || s == "barrier_all" || s == "default") {
    normalized = kAutoSyncTailPolicyBarrierAll.str();
    return true;
  }
  if (s == "mte3-to-s-event0" || s == "mte3_to_s_event0" ||
      s == "setwait-mte3-to-s-event0" || s == "setwait_mte3_to_s_event0") {
    normalized = kAutoSyncTailPolicyMte3ToSEvent0.str();
    return true;
  }
  return false;
}

static bool parseTargetArch(llvm::StringRef archStr, PTOTargetArch &out) {
  std::string s = asciiLowercaseCopy(archStr);
  if (s == "a3") {
    out = PTOTargetArch::A3;
    return true;
  }
  if (s == "a5") {
    out = PTOTargetArch::A5;
    return true;
  }
  return false;
}

namespace {
static void
printIRDumpHeader(PassManager::IRPrinterConfig::PrintCallbackFn printCallback,
                  llvm::raw_ostream &out) {
  std::string dumpText;
  llvm::raw_string_ostream dumpStream(dumpText);
  printCallback(dumpStream);
  dumpStream.flush();

  llvm::StringRef headerLine = dumpText;
  if (size_t newlinePos = headerLine.find('\n');
      newlinePos != std::string::npos)
    headerLine = headerLine.take_front(newlinePos);
  out << headerLine << "\n";
}

class SelectedFuncsIRPrinterConfig : public PassManager::IRPrinterConfig {
public:
  SelectedFuncsIRPrinterConfig(llvm::raw_ostream &out)
      : IRPrinterConfig(/*printModuleScope=*/false,
                        /*printAfterOnlyOnChange=*/false,
                        /*printAfterOnlyOnFailure=*/false),
        out(out) {}

  void printBeforeIfEnabled(Pass *, Operation *, PrintCallbackFn) override {}

protected:
  void printSelectedOp(Operation *op, PrintCallbackFn printCallback) {
    printIRDumpHeader(printCallback, out);
    op->print(out, getOpPrintingFlags());
    out << "\n\n";
  }

  template <typename OpT, typename SelectorT>
  static bool
  appendSelectedFuncs(ModuleOp moduleOp,
                      llvm::SmallVectorImpl<Operation *> &matchedOps,
                      SelectorT selector) {
    bool added = false;
    for (OpT funcOp : moduleOp.getOps<OpT>()) {
      if (!selector(funcOp))
        continue;
      matchedOps.push_back(funcOp.getOperation());
      added = true;
    }
    return added;
  }

  void
  printSelectedFuncs(ModuleOp moduleOp, PrintCallbackFn printCallback,
                     llvm::function_ref<bool(func::FuncOp)> funcSelector,
                     llvm::function_ref<bool(emitc::FuncOp)> emitcSelector) {
    llvm::SmallVector<Operation *, 8> matchedOps;
    bool hasMatches =
        appendSelectedFuncs<func::FuncOp>(moduleOp, matchedOps, funcSelector);
    hasMatches |=
        appendSelectedFuncs<emitc::FuncOp>(moduleOp, matchedOps, emitcSelector);
    if (!hasMatches)
      return;

    printIRDumpHeader(printCallback, out);
    for (Operation *op : matchedOps) {
      op->print(out, getOpPrintingFlags());
      out << "\n";
    }
    out << "\n";
  }

  llvm::raw_ostream &out;
};

class FuncFilteredIRPrinterConfig final : public SelectedFuncsIRPrinterConfig {
public:
  FuncFilteredIRPrinterConfig(std::string funcFilter, llvm::raw_ostream &out)
      : SelectedFuncsIRPrinterConfig(out), funcFilter(std::move(funcFilter)),
        out(out) {}

  void printBeforeIfEnabled(Pass *, Operation *, PrintCallbackFn) override {}

  void printAfterIfEnabled(Pass *, Operation *op,
                           PrintCallbackFn printCallback) override {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (!funcOp.getSymName().contains(funcFilter))
        return;
      printSelectedOp(funcOp, printCallback);
      return;
    }
    if (auto emitcFuncOp = dyn_cast<emitc::FuncOp>(op)) {
      if (!emitcFuncOp.getSymName().contains(funcFilter))
        return;
      printSelectedOp(emitcFuncOp, printCallback);
      return;
    }

    auto moduleOp = dyn_cast<ModuleOp>(op);
    if (!moduleOp)
      return;

    printSelectedFuncs(
        moduleOp, printCallback,
        [&](func::FuncOp funcOp) {
          return funcOp.getSymName().contains(funcFilter);
        },
        [&](emitc::FuncOp funcOp) {
          return funcOp.getSymName().contains(funcFilter);
        });
  }

private:
  std::string funcFilter;
  llvm::raw_ostream &out;
};

class UserRelevantIRPrinterConfig final : public SelectedFuncsIRPrinterConfig {
public:
  UserRelevantIRPrinterConfig(const llvm::StringSet<> &userFuncNames,
                              llvm::raw_ostream &out)
      : SelectedFuncsIRPrinterConfig(out), userFuncNames(userFuncNames) {}

  void printAfterIfEnabled(Pass *, Operation *op,
                           PrintCallbackFn printCallback) override {
    if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
      if (!shouldPrintFunction(funcOp))
        return;
      printSelectedOp(funcOp, printCallback);
      return;
    }
    if (auto emitcFuncOp = dyn_cast<emitc::FuncOp>(op)) {
      if (!shouldPrintFunction(emitcFuncOp))
        return;
      printSelectedOp(emitcFuncOp, printCallback);
      return;
    }

    auto moduleOp = dyn_cast<ModuleOp>(op);
    if (!moduleOp)
      return;

    printSelectedFuncs(
        moduleOp, printCallback,
        [&](func::FuncOp funcOp) { return shouldPrintFunction(funcOp); },
        [&](emitc::FuncOp funcOp) { return shouldPrintFunction(funcOp); });
  }

private:
  static bool shouldPrintFunctionName(llvm::StringRef symName,
                                      const llvm::StringSet<> &userFuncNames) {
    return userFuncNames.contains(symName) ||
           symName.starts_with("__pto_oplib_inst_") ||
           symName.starts_with("__pto_fused_group_");
  }

  bool shouldPrintFunction(func::FuncOp funcOp) const {
    return shouldPrintFunctionName(funcOp.getSymName(), userFuncNames);
  }

  bool shouldPrintFunction(emitc::FuncOp funcOp) const {
    return shouldPrintFunctionName(funcOp.getSymName(), userFuncNames);
  }

  llvm::StringSet<> userFuncNames;
};
} // namespace

static void maybeEnablePrintIRAfterAll(PassManager &pm,
                                       const llvm::StringSet<> &userFuncNames) {
  if (!printIRAfterAll)
    return;
  std::string funcFilter = printIRAfterAllFuncFilter;
  if (!funcFilter.empty()) {
    pm.enableIRPrinting(std::make_unique<FuncFilteredIRPrinterConfig>(
        std::move(funcFilter), llvm::errs()));
    return;
  }

  pm.enableIRPrinting(std::make_unique<UserRelevantIRPrinterConfig>(
      userFuncNames, llvm::errs()));
}
// --------------------------------------------------------------------------
// Post-process C++ output: rewrite marker calls into Tile member calls.
//
// We emit marker calls in EmitC IR because EmitC currently does not provide a
// first-class op for member-function invocation. After translation, we rewrite:
//   PTOAS__TILE_SET_VALUE(dst, offset, val) -> dst.SetValue(offset, val)
//   PTOAS__TILE_GET_VALUE(src, offset)      -> src.GetValue(offset)
//   PTOAS__TILE_DATA(obj)                  -> obj.data()
//   PTOAS__TILE_SET_VALIDSHAPE(obj, r, c)  -> obj.SetValidShape(r, c)
//   PTOAS__TILE_GET_VALID_ROW(obj)         -> obj.GetValidRow()
//   PTOAS__TILE_GET_VALID_COL(obj)         -> obj.GetValidCol()
//   PTOAS__PTR_LOAD(ptr, offset)           -> ptr[offset]
//   PTOAS__PTR_STORE(ptr, offset, val)     -> ptr[offset] = val
// --------------------------------------------------------------------------
struct MarkerCallMatch {
  size_t markerPos = std::string::npos;
  size_t rparenPos = std::string::npos;
  llvm::SmallVector<llvm::StringRef, 4> args;
};

static llvm::SmallVector<llvm::StringRef, 4>
splitTopLevelCallArgs(llvm::StringRef argsRef) {
  llvm::SmallVector<llvm::StringRef, 4> args;
  size_t partBegin = 0;
  int parenDepth = 0;
  for (size_t i = 0; i < argsRef.size(); ++i) {
    char c = argsRef[i];
    if (c == '(') {
      ++parenDepth;
    } else if (c == ')') {
      if (parenDepth > 0)
        --parenDepth;
    } else if (c == ',' && parenDepth == 0) {
      args.push_back(argsRef.slice(partBegin, i).trim());
      partBegin = i + 1;
    }
  }
  if (partBegin <= argsRef.size())
    args.push_back(argsRef.drop_front(partBegin).trim());
  return args;
}

static bool parseMarkerCallAt(llvm::StringRef cpp, llvm::StringRef marker,
                              size_t markerPos, MarkerCallMatch &match,
                              bool &stopScanning) {
  stopScanning = false;

  size_t lparenPos = markerPos + marker.size();
  if (lparenPos >= cpp.size() || cpp[lparenPos] != '(')
    return false;

  size_t argsBegin = lparenPos + 1;
  int parenDepth = 0;
  size_t rparenPos = std::string::npos;
  for (size_t i = argsBegin; i < cpp.size(); ++i) {
    char c = cpp[i];
    if (c == '(') {
      ++parenDepth;
    } else if (c == ')') {
      if (parenDepth == 0) {
        rparenPos = i;
        break;
      }
      --parenDepth;
    }
  }
  if (rparenPos == std::string::npos) {
    stopScanning = true;
    return false;
  }

  match.markerPos = markerPos;
  match.rparenPos = rparenPos;
  match.args = splitTopLevelCallArgs(cpp.slice(argsBegin, rparenPos));
  return true;
}

template <typename RewriteFn>
static bool rewriteMarkerCalls(std::string &cpp, llvm::StringRef marker,
                               RewriteFn &&buildReplacement) {
  size_t searchPos = 0;
  bool changed = false;
  while (true) {
    llvm::StringRef cppRef(cpp);
    size_t markerPos = cppRef.find(marker, searchPos);
    if (markerPos == llvm::StringRef::npos)
      break;

    MarkerCallMatch match;
    bool stopScanning = false;
    if (!parseMarkerCallAt(cppRef, marker, markerPos, match, stopScanning)) {
      if (stopScanning)
        break;
      searchPos = markerPos + marker.size();
      continue;
    }

    size_t replaceEnd = match.rparenPos;
    std::string replacement;
    if (!buildReplacement(match, replacement, replaceEnd)) {
      searchPos = match.rparenPos + 1;
      continue;
    }

    cpp.replace(match.markerPos, (replaceEnd - match.markerPos) + 1,
                replacement);
    changed = true;
    searchPos = match.markerPos + replacement.size();
  }
  return changed;
}

static bool rewriteMarkerCallToMember(std::string &cpp, llvm::StringRef marker,
                                      llvm::StringRef memberName,
                                      unsigned expectedNumArgs) {
  return rewriteMarkerCalls(cpp, marker,
                            [&](const MarkerCallMatch &match,
                                std::string &replacement, size_t &) -> bool {
                              if (match.args.size() != expectedNumArgs)
                                return false;

                              replacement.clear();
                              replacement.append(match.args[0].str());
                              replacement.push_back('.');
                              replacement.append(memberName.str());
                              replacement.push_back('(');
                              if (expectedNumArgs == 2) {
                                replacement.append(match.args[1].str());
                              } else if (expectedNumArgs == 3) {
                                replacement.append(match.args[1].str());
                                replacement.append(", ");
                                replacement.append(match.args[2].str());
                              }
                              replacement.push_back(')');
                              return true;
                            });
}

static void rewriteTileGetSetValueMarkers(std::string &cpp) {
  // Keep applying until fixed-point in case rewrites shift subsequent matches.
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_SET_VALUE",
                                         "SetValue", /*expectedNumArgs=*/3);
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_GET_VALUE",
                                         "GetValue", /*expectedNumArgs=*/2);
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_DATA", "data",
                                         /*expectedNumArgs=*/1);
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_SET_VALIDSHAPE",
                                         "SetValidShape",
                                         /*expectedNumArgs=*/3);
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_GET_VALID_ROW",
                                         "GetValidRow", /*expectedNumArgs=*/1);
    changed |= rewriteMarkerCallToMember(cpp, "PTOAS__TILE_GET_VALID_COL",
                                         "GetValidCol", /*expectedNumArgs=*/1);
  }
}

// --------------------------------------------------------------------------
// EmitC cleanup: drop empty emitc.expression ops.
//
// After FormExpressions + CSE, EmitC expressions can become empty when their
// root op is CSE'd with an equivalent dominating value outside the expression
// region. Such expressions crash mlir::emitc::translateToCpp because
// ExpressionOp::getRootOp() returns nullptr.
// --------------------------------------------------------------------------
static void dropEmptyEmitCExpressions(Operation *rootOp) {
  llvm::SmallVector<emitc::ExpressionOp, 8> toErase;
  rootOp->walk([&](emitc::ExpressionOp expr) {
    if (expr.getRootOp())
      return;
    Block *body = expr.getBody();
    if (!body)
      return;
    auto yield = dyn_cast<emitc::YieldOp>(body->getTerminator());
    if (!yield || yield.getNumOperands() != 1)
      return;
    Value yielded = yield.getOperand(0);
    expr.getResult().replaceAllUsesWith(yielded);
    toErase.push_back(expr);
  });
  for (emitc::ExpressionOp expr : llvm::reverse(toErase))
    expr.erase();
}

static Attribute getDefaultEmitCVariableInitAttr(OpBuilder &builder,
                                                 Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return builder.getIntegerAttr(intTy, 0);
  if (isa<IndexType>(type))
    return builder.getIndexAttr(0);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return builder.getFloatAttr(floatTy, 0.0);
  if (isa<emitc::OpaqueType, emitc::PointerType>(type))
    return emitc::OpaqueAttr::get(builder.getContext(), "");
  return Attribute{};
}

// FormExpressions may inline conditions into emitc.expression, but the C++
// emitter prints cf.br/cf.cond_br operands by variable name rather than by
// recursively emitting an expression. Materialize such operands so CFG-based
// lowering (e.g. scf.while -> cf.*) stays valid.
static void materializeControlFlowOperands(Operation *rootOp) {
  llvm::SmallVector<Operation *, 16> branches;
  rootOp->walk([&](Operation *op) {
    if (isa<cf::BranchOp, cf::CondBranchOp>(op))
      branches.push_back(op);
  });

  OpBuilder builder(rootOp->getContext());
  for (Operation *op : branches) {
    builder.setInsertionPoint(op);
    for (OpOperand &operand : op->getOpOperands()) {
      Value value = operand.get();
      auto expr = dyn_cast_or_null<emitc::ExpressionOp>(value.getDefiningOp());
      if (!expr)
        continue;

      Attribute initAttr =
          getDefaultEmitCVariableInitAttr(builder, value.getType());
      if (!initAttr)
        continue;

      Value tmp = builder
                      .create<emitc::VariableOp>(op->getLoc(), value.getType(),
                                                 initAttr)
                      .getResult();
      builder.create<emitc::AssignOp>(op->getLoc(), tmp, value);
      operand.set(tmp);
    }
  }
}

static bool rewriteMarkerCallToSubscript(std::string &cpp,
                                         llvm::StringRef marker,
                                         unsigned expectedNumArgs,
                                         bool isStore) {
  return rewriteMarkerCalls(
      cpp, marker,
      [&](const MarkerCallMatch &match, std::string &replacement,
          size_t &) -> bool {
        if (match.args.size() != expectedNumArgs)
          return false;

        if (isStore) {
          replacement =
              (match.args[0] + "[" + match.args[1] + "] = " + match.args[2])
                  .str();
        } else {
          replacement = (match.args[0] + "[" + match.args[1] + "]").str();
        }
        return true;
      });
}

static void rewritePtrScalarMarkers(std::string &cpp) {
  bool changed = true;
  while (changed) {
    changed = false;
    changed |= rewriteMarkerCallToSubscript(
        cpp, "PTOAS__PTR_LOAD", /*expectedNumArgs=*/2, /*isStore=*/false);
    changed |= rewriteMarkerCallToSubscript(
        cpp, "PTOAS__PTR_STORE", /*expectedNumArgs=*/3, /*isStore=*/true);
  }
}

static bool rewriteAddPtrTraceMarkers(std::string &cpp, bool showTrace) {
  return rewriteMarkerCalls(
      cpp, "PTOAS__ADDPTR_TRACE",
      [&](const MarkerCallMatch &match, std::string &replacement,
          size_t &replaceEnd) -> bool {
        if (match.args.size() != 3)
          return false;

        replacement.clear();
        if (showTrace) {
          replacement.reserve(64);
          replacement.append("/* ADDPTR_TRACE: ");
          replacement.append(match.args[0].str());
          replacement.append(" = ");
          replacement.append(match.args[1].str());
          replacement.append(" + ");
          replacement.append(match.args[2].str());
          replacement.append(" */");
        } else {
          size_t i = match.rparenPos + 1;
          while (i < cpp.size() &&
                 std::isspace(static_cast<unsigned char>(cpp[i]))) {
            ++i;
          }
          if (i < cpp.size() && cpp[i] == ';')
            replaceEnd = i;
        }
        return true;
      });
}

static void rewriteHoistedGlobalTensorDecls(std::string &cpp) {
  // When `declareVariablesAtTop` is enabled, the C++ emitter hoists SSA value
  // declarations to the top of the function and emits assignments later. This
  // requires the C++ type to be default-constructible.
  //
  // `GlobalTensor<...>` from pto-isa does NOT have a default constructor, so a
  // hoisted declaration like:
  //   GlobalTensor<...> v42;
  // fails to compile. Initialize those hoisted temporaries with a null pointer
  // so they are constructible:
  //   GlobalTensor<...> v42(nullptr);
  //
  // We keep the assignment later; the null-initialized value is never used.
  std::string out;
  out.reserve(cpp.size() + 64);

  llvm::StringRef ref(cpp);
  while (!ref.empty()) {
    auto split = ref.split('\n');
    llvm::StringRef line = split.first;
    llvm::StringRef rest = split.second;

    llvm::StringRef trimmed = line.trim();
    bool rewritten = false;
    if (trimmed.starts_with("GlobalTensor<") && trimmed.ends_with(";") &&
        !trimmed.contains('=') && !trimmed.contains('(')) {
      llvm::StringRef decl = trimmed.drop_back().rtrim();
      size_t lastWs = decl.find_last_of(" \t");
      if (lastWs != llvm::StringRef::npos) {
        llvm::StringRef varName = decl.drop_front(lastWs + 1);
        if (varName.starts_with("v") && varName.size() > 1) {
          bool allDigits = true;
          for (char c : varName.drop_front(1)) {
            if (c < '0' || c > '9') {
              allDigits = false;
              break;
            }
          }
          if (allDigits) {
            size_t indentLen = line.find_first_not_of(" \t");
            if (indentLen == std::string::npos)
              indentLen = 0;
            llvm::StringRef indent = line.take_front(indentLen);

            out.append(indent.str());
            out.append(decl.str());
            out.append("(nullptr);");
            rewritten = true;
          }
        }
      }
    }

    if (!rewritten)
      out.append(line.str());
    if (!rest.empty())
      out.push_back('\n');
    ref = rest;
  }

  cpp.swap(out);
}

int main(int argc, char **argv) {
  DialectRegistry registry;
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::cf::ControlFlowDialect>();
  registry.insert<mlir::bufferization::BufferizationDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::vector::VectorDialect>();

  registry.insert<mlir::pto::PTODialect>();
  // mlir::registerAllDialects(registry);
  arith::registerBufferizableOpInterfaceExternalModels(registry);
  tensor::registerBufferizableOpInterfaceExternalModels(registry);
  // func::registerBufferizableOpInterfaceExternalModels(registry);
  pto::registerBufferizableOpInterfaceExternalModels(registry);

  registry.insert<emitc::EmitCDialect>();
  registry.insert<mlir::LLVM::LLVMDialect>();

  llvm::cl::SetVersionPrinter(printPTOASVersion);

  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  // Read whole input first (so we can auto-detect .ptobc by magic).
  auto fileOrErr = llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (!fileOrErr) {
    llvm::errs() << "Error: Could not open input file: "
                 << fileOrErr.getError().message() << "\n";
    return 1;
  }

  MLIRContext context(registry);
  // Be tolerant: ptobc decode may materialize ops from dialects that aren't
  // explicitly registered/loaded in this tool yet.
  context.allowUnregisteredDialects(true);
  if (printIRAfterAll)
    context.disableMultithreading();

  context.getOrLoadDialect<emitc::EmitCDialect>();
  context.getOrLoadDialect<mlir::pto::PTODialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<math::MathDialect>();
  context.getOrLoadDialect<memref::MemRefDialect>();
  context.getOrLoadDialect<affine::AffineDialect>();
  context.getOrLoadDialect<vector::VectorDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();

  PTOTargetArch effectiveArch = PTOTargetArch::A3;
  if (!parseTargetArch(ptoTargetArch, effectiveArch)) {
    llvm::errs() << "Error: invalid --pto-arch='" << ptoTargetArch
                 << "'. Expected 'a3' or 'a5'.\n";
    return 1;
  }
  std::string arch = (effectiveArch == PTOTargetArch::A5) ? std::string("a5")
                                                          : std::string("a3");

  OwningOpRef<ModuleOp> module;
  llvm::StringRef buf = (*fileOrErr)->getBuffer();
  const bool isPTOBC =
      (buf.size() >= 6 && std::memcmp(buf.data(), "PTOBC\0", 6) == 0);

  if (isPTOBC) {
    // Decode PTO bytecode directly into an MLIR module.
    llvm::ArrayRef<uint8_t> bytes(reinterpret_cast<const uint8_t *>(buf.data()),
                                  buf.size());
#if defined(__cpp_exceptions) || defined(__EXCEPTIONS)
    try {
      module = ptobc::decodePTOBCToModule(bytes, context);
    } catch (...) {
      llvm::errs() << "Error: Failed to decode PTOBC.\n";
      return 1;
    }
#else
    module = ptobc::decodePTOBCToModule(bytes, context);
#endif
    if (!module) {
      llvm::errs() << "Error: Failed to decode PTOBC.\n";
      return 1;
    }
  } else {
    // Parse textual MLIR (.pto).
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    pto::ScopedPTOParserTargetArch scopedParserArch(
        arch == "a5" ? pto::PTOParserTargetArch::A5
                     : pto::PTOParserTargetArch::A3);
    module = parseSourceFile<ModuleOp>(sourceMgr, &context);
    if (!module) {
      llvm::errs() << "Error: Failed to parse MLIR.\n";
      return 1;
    }
  }

  llvm::StringSet<> inputFuncNames;
  for (func::FuncOp funcOp : module->getOps<func::FuncOp>())
    inputFuncNames.insert(funcOp.getSymName());

  PTOBuildLevel effectiveLevel = defaultBuildLevel();
  if (!parseBuildLevel(ptoBuildLevel, effectiveLevel)) {
    llvm::errs() << "Error: invalid --pto-level='" << ptoBuildLevel
                 << "'. Expected 'level1', 'level2', or 'level3'.\n";
    return 1;
  }

  bool invalidAutoSyncTailHint = false;
  module->walk([&](mlir::func::FuncOp func) {
    auto hintAttr =
        func->getAttrOfType<mlir::StringAttr>("pto.auto_sync_tail_hint");
    if (!hintAttr)
      return;

    std::string normalizedHint;
    if (!parseAutoSyncTailHint(hintAttr.getValue(), normalizedHint)) {
      func.emitError("invalid pto.auto_sync_tail_hint '")
          << hintAttr.getValue()
          << "'. Expected 'barrier-all' (or 'default') or "
             "'mte3-to-s-event0'.";
      invalidAutoSyncTailHint = true;
      return;
    }
    func->setAttr("pto.auto_sync_tail_hint",
                  mlir::StringAttr::get(&context, normalizedHint));
  });
  if (invalidAutoSyncTailHint)
    return 1;

  module->getOperation()->setAttr("pto.target_arch",
                                  mlir::StringAttr::get(&context, arch));
  const bool enableA5OplibPipeline = (effectiveArch == PTOTargetArch::A5);

  if (effectiveLevel == PTOBuildLevel::Level3) {
    bool missing = false;
    module->walk([&](pto::AllocTileOp op) {
      if (!op.getAddr()) {
        op.emitError("requires 'addr' operand when --pto-level=level3");
        missing = true;
      }
    });
    if (missing)
      return 1;
  } else {
    bool hasAddr = false;
    module->walk([&](pto::AllocTileOp op) {
      if (op.getAddr()) {
        op.emitError("unexpected 'addr' operand: only supported when "
                     "--pto-level=level3");
        hasAddr = true;
      }
    });
    if (hasAddr)
      return 1;
  }

  if (enableA5OplibPipeline && opLibDir.empty()) {
    llvm::errs() << "Error: --op-lib-dir is required.\n";
    return 1;
  }

  if (!enableA5OplibPipeline && enableOpFusion) {
    llvm::errs() << "Warning: --enable-op-fusion is ignored because "
                    "--pto-arch!=a5.\n";
  }

  if (!printIRAfterAll && !printIRAfterAllFuncFilter.empty()) {
    llvm::errs() << "Warning: --print-ir-after-all-func-filter has no effect "
                    "without --print-ir-after-all.\n";
  }

  // [Fix] ToolOutputFile Usage
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  // Stage 1: front-end lowering to memref-level IR.
  if (enableA5OplibPipeline) {
    if (failed(
            pto::importPTOOpLibTemplates(*module, opLibDir, opFusionDebug))) {
      llvm::errs() << "Error: Failed to import OP-Lib templates.\n";
      return 1;
    }
  }

  PassManager preCodegenPm(&context);
  maybeEnablePrintIRAfterAll(preCodegenPm, inputFuncNames);

  // preCodegenPm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertCVMovPass());
  // preCodegenPm.addNestedPass<mlir::func::FuncOp>(pto::createPTOConvertToDPSPass());
  // preCodegenPm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertLoadStoreForMixCVPass());
  preCodegenPm.addNestedPass<mlir::func::FuncOp>(
      pto::createLoweringSyncToPipePass());
  if (enableA5OplibPipeline) {
    preCodegenPm.addPass(pto::createPTOValidateSimdIRPass());
  }

  preCodegenPm.addPass(pto::createPTOViewToMemrefPass());

  if (failed(preCodegenPm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  // Stage 2: remaining optimization + codegen pipeline.
  PassManager pm(&context);
  maybeEnablePrintIRAfterAll(pm, inputFuncNames);
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addPass(pto::createPTOViewToMemrefPass());
  // bufferizationPipeline(pm);
  // pm.addPass(createInferPTOMemScopePass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }

  // Conditionally add Sync pass based on flag.
  if (enableInsertSync)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());

  if (enableA5OplibPipeline) {
    if (enableOpFusion) {
      pm.addNestedPass<mlir::func::FuncOp>(
          pto::createPTOCreateFusionGroupsPass());

      pto::PTOOutlineFusionGroupsOptions outlineGroupsOptions;
      outlineGroupsOptions.debug = opFusionDebug;
      pm.addPass(pto::createPTOOutlineFusionGroupsPass(outlineGroupsOptions));
    }

    pto::PTOInstantiateAndLowerToLibCallOptions instantiateLowerOptions;
    instantiateLowerOptions.opLibDir = opLibDir;
    instantiateLowerOptions.debug = opFusionDebug;
    pm.addPass(pto::createPTOInstantiateAndLowerToLibCallPass(
        instantiateLowerOptions));

    pto::PTOInlineLibCallOptions inlineLibCallOptions;
    inlineLibCallOptions.debug = opFusionDebug;
    pm.addPass(pto::createPTOInlineLibCallPass(inlineLibCallOptions));

    if (enableOpFusion) {
      pto::PTOLowLevelLoopFusionOptions loopFusionOptions;
      loopFusionOptions.debug = opFusionDebug;
      pm.addPass(pto::createPTOLowLevelLoopFusionPass(loopFusionOptions));
    }

    // Keep OP-Lib lowered loop nests intact until PTOLowLevelLoopFusion runs.
    // In particular, avoid pre-fusion canonicalization folding away
    // single-trip loops, otherwise the fusion pass no longer sees the regular
    // loop structure emitted by OP-Lib lowering.
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createCSEPass());
  }
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());

  if (failed(pm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTORemoveRedundantBarrierPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOHighDimLoweringPass());
  // pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVFloopGatherPass());

  PassManager codegenPm(&context);
  maybeEnablePrintIRAfterAll(codegenPm, inputFuncNames);
  codegenPm.addPass(createCSEPass());
  if (effectiveArch == PTOTargetArch::A3) {
    codegenPm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A3));
  } else {
    codegenPm.addPass(pto::createEmitPTOManualPass(pto::PTOArch::A5));
  }
  codegenPm.addPass(emitc::createFormExpressionsPass());
  codegenPm.addPass(mlir::createCSEPass());

  if (failed(codegenPm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  dropEmptyEmitCExpressions(module.get());
  materializeControlFlowOperands(module.get());
  if (failed(reorderEmitCFunctions(module.get()))) {
    llvm::errs()
        << "Error: Failed to order emitted functions for C++ emission.\n";
    return 1;
  }

  // Emit C++ to string, then post-process, then write to output file.
  std::string cppOutput;
  llvm::raw_string_ostream cppOS(cppOutput);
  // CFG-style lowering (e.g. scf.while -> cf.br/cf.cond_br) may introduce
  // multiple blocks, requiring variables to be declared at the top for valid
  // C++ emission.
  bool declareVariablesAtTop = false;
  for (auto func : module->getOps<func::FuncOp>()) {
    if (func.getBlocks().size() > 1) {
      declareVariablesAtTop = true;
      break;
    }
  }
  if (!declareVariablesAtTop) {
    for (auto func : module->getOps<emitc::FuncOp>()) {
      if (func.getBlocks().size() > 1) {
        declareVariablesAtTop = true;
        break;
      }
    }
  }
  if (failed(emitc::translateToCpp(
          *module, cppOS,
          /*declareVariablesAtTop=*/declareVariablesAtTop))) {
    llvm::errs() << "Error: Failed to emit C++.\n";
    return 1;
  }
  cppOS.flush();
  rewriteTileGetSetValueMarkers(cppOutput);
  rewritePtrScalarMarkers(cppOutput);
  rewriteAddPtrTraceMarkers(cppOutput, emitAddPtrTrace);
  rewriteHoistedGlobalTensorDecls(cppOutput);
  outputFile.os() << cppOutput;

  outputFile.keep(); // Success, keep the file

  return 0;
}
