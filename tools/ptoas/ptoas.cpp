//===- ptoas.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTO.h"
#include "PTO/Transforms/VPTOLowering.h"
#include "PTO/Transforms/VPTOLLVMEmitter.h"
#include "PTO/Transforms/VPTOTextEmitter.h"
#include "PTO/Transforms/BufferizableOpInterfaceImpl.h"
#include "PTO/Transforms/CppEmitter.h"
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
#include "mlir/Pass/PassRegistry.h"
#include "ptobc/ptobc_decode.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h" // [Fix] Required for OF_None
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include <cctype>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <system_error>

using namespace mlir;
using namespace pto;

#ifndef PTOAS_RELEASE_VERSION
#define PTOAS_RELEASE_VERSION "unknown"
#endif

static void printPTOASVersion(llvm::raw_ostream &os) {
  os << "ptoas " << PTOAS_RELEASE_VERSION << "\n";
}

namespace {
} // namespace

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
    llvm::cl::desc("Enable the VPTO fusion pipeline "
                   "(FusionPlan/OpScheduling/FusionRegionGen/"
                   "PTOToVPTO/LowLevelLoopFusion/FlattenFusionRegion). "
                   "Ignored by EmitC."),
    llvm::cl::init(false));

static llvm::cl::opt<bool> testOnlyFusionRegionGen(
    "test-only-fusion-region-gen",
    llvm::cl::desc("Run only PTOFusionRegionGen and exit (test-only)"),
    llvm::cl::Hidden, llvm::cl::init(false));

static llvm::cl::opt<bool> testOnlyOpScheduling(
    "test-only-op-scheduling",
    llvm::cl::desc("Run only FusionPlan + OpScheduling and exit (test-only)"),
    llvm::cl::Hidden, llvm::cl::init(false));

static llvm::cl::opt<std::string>
    opLibDir("op-lib-dir",
             llvm::cl::desc("Deprecated OP-Lib template directory flag. "
                            "Ignored by both EmitC and VPTO backends."),
             llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool>
    opFusionDebug("op-fusion-debug",
                  llvm::cl::desc("Enable verbose debug logs for the VPTO "
                                 "fusion pipeline"),
                  llvm::cl::init(false));

static llvm::cl::opt<unsigned> postFusionLoopUnrollFactor(
    "post-fusion-loop-unroll-factor",
    llvm::cl::desc("Forward a fixed unroll factor into "
                   "PTOPostFusionLoopUnroll when the A5 VPTO fusion mainline "
                   "is enabled. Accepted values: 0 (auto), 2, 4"),
    llvm::cl::value_desc("0|2|4"), llvm::cl::init(0));

static llvm::cl::opt<bool> printIRAfterAll(
    "print-ir-after-all",
    llvm::cl::desc("Print MLIR IR after each pass in all PTOAS pass pipelines "
                   "for user-related functions"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> dumpPreFusionAnalysis(
    "dump-pre-fusion-analysis",
    llvm::cl::desc("Run pre-fusion analysis in tile_buf world, print a stable "
                   "text dump, and exit"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> printIRAfterAllFuncFilter(
    "print-ir-after-all-func-filter",
    llvm::cl::desc("When --print-ir-after-all is enabled, only print dumps for "
                   "func.func whose symbol name contains this substring "
                   "(overrides the default user-related filtering)"),
    llvm::cl::value_desc("substring"), llvm::cl::init(""));

static mlir::PassPipelineCLParser passPipeline(
    "", "Run a custom MLIR pass pipeline and exit");

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

static llvm::cl::opt<std::string> ptoBackend(
    "pto-backend",
    llvm::cl::desc("Final PTOAS backend: emitc or vpto (default: emitc)"),
    llvm::cl::value_desc("emitc|vpto"), llvm::cl::init("emitc"));

static llvm::cl::opt<bool> emitVPTO(
    "emit-vpto",
    llvm::cl::desc("Write final post-pass VPTO IR to -o"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoPrintIR(
    "vpto-print-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> vptoLoweringStrategy(
    "vpto-lowering-strategy",
    llvm::cl::desc("VPTO vector lowering strategy: post-update or no-post-update"),
    llvm::cl::value_desc("post-update|no-post-update"),
    llvm::cl::init("post-update"));

static llvm::cl::opt<bool> dumpVPTOIR(
    "dump-vpto-ir",
    llvm::cl::desc("Print post-pass VPTO backend IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> ptoPrintSeamIR(
    "pto-print-seam-ir",
    llvm::cl::desc("Print shared pre-backend seam IR to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> ptoSeamIRFile(
    "pto-seam-ir-file",
    llvm::cl::desc("Write shared pre-backend seam IR to a file"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<bool> vptoPrintIntrinsics(
    "vpto-print-intrinsics",
    llvm::cl::desc("Print VPTO intrinsic selection decisions to stderr"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoEmitHIVMText(
    "vpto-emit-hivm-text",
    llvm::cl::desc(
        "After lowering to VPTO IR, emit textual LLVM/HIVM instead of raw "
        "VPTO IR"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoEmitHIVMOfficialLLVM(
    "vpto-emit-hivm-llvm",
    llvm::cl::desc("After lowering to VPTO IR, emit textual LLVM/HIVM via "
                   "the official LLVM dialect export path"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoEmitHIVMOfficialBitcode(
    "vpto-emit-hivm-bc",
    llvm::cl::desc("After lowering to VPTO IR, emit LLVM bitcode via the "
                   "official LLVM dialect export path"),
    llvm::cl::init(false));

static llvm::cl::opt<bool> vptoAllowUnresolved(
    "vpto-allow-unresolved",
    llvm::cl::desc("Emit explicit unresolved VPTO comments instead of failing"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> vptoUnresolvedReport(
    "vpto-unresolved-report",
    llvm::cl::desc("Write unresolved VPTO mappings to a sidecar report"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

static llvm::cl::opt<std::string> hivmUnresolvedReport(
    "hivm-unresolved-report",
    llvm::cl::desc("Write unresolved HIVM mappings to a sidecar report"),
    llvm::cl::value_desc("path"), llvm::cl::init(""));

enum class PTOBuildLevel {
  Level1,
  Level2,
  Level3,
};

enum class PTOTargetArch {
  A3,
  A5,
};

enum class PTOBackend {
  EmitC,
  VPTO,
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

static bool parseBackend(llvm::StringRef backendStr, PTOBackend &out) {
  std::string s = asciiLowercaseCopy(backendStr);
  if (s == "emitc") {
    out = PTOBackend::EmitC;
    return true;
  }
  if (s == "vpto") {
    out = PTOBackend::VPTO;
    return true;
  }
  return false;
}

static bool isSupportedPostFusionLoopUnrollFactor(unsigned factor) {
  return factor == 0 || factor == 2 || factor == 4;
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

static void addSharedPreBackendPasses(OpPassManager &pm,
                                      PTOBuildLevel effectiveLevel) {
  pm.addNestedPass<mlir::func::FuncOp>(pto::createLoweringSyncToPipePass());

  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  pm.addPass(pto::createPTOViewToMemrefPass());

  if (effectiveLevel != PTOBuildLevel::Level3) {
    PlanMemoryOptions planMemoryOption;
    planMemoryOption.memMode = MemPlanMode::LOCAL_MEM_PLAN;
    planMemoryOption.enableGlobalReuse = false;
    planMemoryOption.enablePrintMemoryAllocatedSize = false;
    pm.addPass(pto::createPlanMemoryPass(planMemoryOption));
  }

  if (enableInsertSync) {
    if (effectiveLevel == PTOBuildLevel::Level3) {
      llvm::errs() << "Warning: --enable-insert-sync is ignored because "
                      "--pto-level=level3.\n";
    } else {
      pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOInsertSyncPass());
    }
  }

  pm.addPass(createCSEPass());
}

static void addA5FusionRegionMainlinePreBackendPasses(OpPassManager &pm) {
  pm.addNestedPass<mlir::func::FuncOp>(pto::createFusionPlanPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createOpSchedulingPass());
  pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOFusionRegionGenPass());
}

static void addVPTOBackendMainlinePasses(OpPassManager &pm,
                                         bool enableFusionMainline,
                                         unsigned forwardedUnrollFactor) {
  // Keep the A5 backend lowering boundary explicit:
  //   FusionRegionGen -> shared pre-backend normalization
  //   -> PTOVPTOVersionSelection -> PTOToVPTO
  //   -> PTOValidateVPTOIR
  //   -> PTOVPTOIfCanonicalize
  //   -> PTOFusionMergeVecScope
  //   -> PTOLowLevelLoopFusion -> Canonicalize
  //   -> CSE -> PTOFusionPredicateElision
  //   -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll
  //   -> PTOFlattenFusionRegion
  //   -> backend emission.
  if (enableFusionMainline) {
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOVPTOVersionSelectionPass());
    pm.addPass(pto::createLowerPTOToVPTOPass());
  } else {
    pm.addPass(pto::createLowerPTOToVPTOPass(vptoLoweringStrategy));
  }

  pm.addPass(pto::createPTOValidateVPTOIRPass());

  if (enableFusionMainline) {
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOVPTOIfCanonicalizePass());
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionMergeVecScopePass());
    pto::PTOLowLevelLoopFusionOptions loopFusionOptions;
    loopFusionOptions.debug = opFusionDebug;
    pm.addPass(pto::createPTOLowLevelLoopFusionPass(loopFusionOptions));
    GreedyRewriteConfig canonicalizeConfig;
    pm.addPass(createCanonicalizerPass(
        canonicalizeConfig,
        {"SimplifyTrivialLoops",
         "{anonymous}::SimplifyTrivialLoops",
         "mlir::(anonymous namespace)::SimplifyTrivialLoops"}));
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOVPTOTrivialLoopCanonicalizePass());
    pm.addPass(createCSEPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionPredicateElisionPass());
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFusionLoadStoreElisionPass());
    pto::PTOPostFusionLoopUnrollOptions postFusionLoopUnrollOptions;
    postFusionLoopUnrollOptions.forcedUnrollFactor = forwardedUnrollFactor;
    pm.addNestedPass<mlir::func::FuncOp>(pto::createPTOPostFusionLoopUnrollPass(
        postFusionLoopUnrollOptions));
    pm.addNestedPass<mlir::func::FuncOp>(
        pto::createPTOFlattenFusionRegionPass());
  }

  pm.addPass(mlir::createCSEPass());
}

static LogicalResult runVPTOAuthoringValidation(MLIRContext &context,
                                                ModuleOp module,
                                                const llvm::StringSet<> &userFuncNames) {
  PassManager validationPM(&context);
  maybeEnablePrintIRAfterAll(validationPM, userFuncNames);
  validationPM.addPass(pto::createPTOValidateVPTOIRPass());
  return validationPM.run(module);
}

static void printVPTOIROpSummary(ModuleOp module, llvm::raw_ostream &os) {
  for (func::FuncOp func : module.getOps<func::FuncOp>()) {
    for (Operation &op : func.getBody().front().getOperations()) {
      os << "VPTO IR op: " << op.getName().getStringRef() << "\n";
      for (Region &region : op.getRegions()) {
        for (Block &block : region) {
          for (Operation &nested : block.getOperations())
            os << "VPTO IR op: " << nested.getName().getStringRef() << "\n";
        }
      }
    }
  }
}

static LogicalResult emitSharedPreBackendSeamIR(ModuleOp module,
                                                llvm::StringRef outputPath) {
  if (outputPath.empty())
    return success();

  if (outputPath == "-") {
    module->print(llvm::outs());
    llvm::outs() << "\n";
    llvm::outs().flush();
    return success();
  }

  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputPath, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << "Error: failed to open seam IR file '" << outputPath
                 << "': " << ec.message() << "\n";
    return failure();
  }

  module->print(outputFile.os());
  outputFile.os() << "\n";
  outputFile.keep();
  return success();
}

static bool containsVPTOIR(llvm::StringRef input) {
  llvm::StringRef rest = input;
  while (!rest.empty()) {
    auto split = rest.split('\n');
    llvm::StringRef line = split.first.trim();
    if (!line.starts_with("//") &&
        (line.contains("!pto.vec<") || line.contains("!pto.mask") ||
         line.contains("!pto.align") || line.contains("pto.copy_") ||
         line.contains("pto.set_loop") || line.contains("pto.v") ||
         line.contains("pto.plt_") || line.contains("pto.pset_") ||
         line.contains("pto.psts") || line.contains("pto.pdintlv_") ||
         line.contains("pto.set_flag") || line.contains("pto.wait_flag") ||
         line.contains("pto.pipe_barrier") || line.contains("pto.get_buf") ||
         line.contains("pto.rls_buf")))
      return true;
    rest = split.second;
  }
  return false;
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
  mlir::registerAllPasses();
  ::registerPTOPostFusionLoopUnroll();
  mlir::registerPassManagerCLOptions();

  llvm::cl::SetVersionPrinter(printPTOASVersion);

  // Parse command line options
  llvm::cl::ParseCommandLineOptions(argc, argv, "PTO Assembler (ptoas)\n");

  PTOBackend effectiveBackend = PTOBackend::EmitC;
  if (!parseBackend(ptoBackend, effectiveBackend)) {
    llvm::errs() << "Error: invalid --pto-backend='" << ptoBackend
                 << "'. Expected 'emitc' or 'vpto'.\n";
    return 1;
  }

  if (vptoEmitHIVMOfficialLLVM && vptoEmitHIVMOfficialBitcode) {
    llvm::errs() << "Error: --vpto-emit-hivm-llvm and --vpto-emit-hivm-bc "
                    "cannot be used together.\n";
    return 1;
  }

  if (emitVPTO &&
      (vptoEmitHIVMText || vptoEmitHIVMOfficialLLVM ||
       vptoEmitHIVMOfficialBitcode)) {
    llvm::errs() << "Error: --emit-vpto cannot be used together with HIVM "
                    "emission flags.\n";
    return 1;
  }

  if (effectiveBackend != PTOBackend::VPTO &&
      (vptoEmitHIVMText || vptoEmitHIVMOfficialLLVM ||
       vptoEmitHIVMOfficialBitcode || emitVPTO ||
       vptoPrintIntrinsics || vptoAllowUnresolved ||
       !vptoUnresolvedReport.empty() || !hivmUnresolvedReport.empty() ||
       ptoPrintSeamIR || !ptoSeamIRFile.empty())) {
    llvm::errs() << "Error: VPTO-specific flags require "
                    "--pto-backend=vpto.\n";
    return 1;
  }

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
  if (printIRAfterAll || dumpPreFusionAnalysis)
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
  const bool inputIsVPTOIR = containsVPTOIR(buf);

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

  if (dumpPreFusionAnalysis) {
    PassManager analysisPm(&context);
    analysisPm.addNestedPass<mlir::func::FuncOp>(
        pto::createPrintPreFusionAnalysisPass());
    if (failed(analysisPm.run(*module))) {
      llvm::errs() << "Error: Pre-fusion analysis dump failed.\n";
      return 1;
    }
    return 0;
  }

  const bool useVPTOBackendPipeline =
      (effectiveBackend == PTOBackend::VPTO);
  const bool enableA5FusionMainline =
      (enableOpFusion && effectiveArch == PTOTargetArch::A5 &&
       effectiveBackend == PTOBackend::VPTO);

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

  if (effectiveArch != PTOTargetArch::A5 && enableOpFusion) {
    llvm::errs() << "Warning: --enable-op-fusion is ignored because "
                    "--pto-arch!=a5.\n";
  }

  if (effectiveBackend == PTOBackend::EmitC &&
      effectiveArch == PTOTargetArch::A5 && enableOpFusion) {
    llvm::errs() << "Warning: --enable-op-fusion is ignored by the EmitC "
                    "backend.\n";
  }

  if (effectiveBackend == PTOBackend::EmitC &&
      effectiveArch == PTOTargetArch::A5 && opFusionDebug) {
    llvm::errs() << "Warning: --op-fusion-debug is ignored by the EmitC "
                    "backend.\n";
  }

  if (!opLibDir.empty()) {
    llvm::errs() << "Warning: --op-lib-dir is deprecated and ignored.\n";
  }

  if (useVPTOBackendPipeline) {
    if (opFusionDebug && !enableA5FusionMainline) {
      llvm::errs() << "Warning: --op-fusion-debug is ignored because "
                      "VPTO fusion mainline is not enabled.\n";
    }
    if (postFusionLoopUnrollFactor.getNumOccurrences() > 0 &&
        !enableA5FusionMainline) {
      llvm::errs()
          << "Warning: --post-fusion-loop-unroll-factor is ignored because "
             "VPTO fusion mainline is not enabled.\n";
    }
    if (enableA5FusionMainline && vptoLoweringStrategy.getNumOccurrences() > 0) {
      llvm::errs() << "Warning: --vpto-lowering-strategy is ignored because "
                      "VPTO fusion mainline uses per-op "
                      "pto.lowering_choice.\n";
    }
  }

  if (!isSupportedPostFusionLoopUnrollFactor(postFusionLoopUnrollFactor)) {
    llvm::errs() << "Error: invalid --post-fusion-loop-unroll-factor='"
                 << postFusionLoopUnrollFactor << "'. Expected 0, 2, or 4.\n";
    return 1;
  }

  if (!printIRAfterAll && !printIRAfterAllFuncFilter.empty()) {
    llvm::errs() << "Warning: --print-ir-after-all-func-filter has no effect "
                    "without --print-ir-after-all.\n";
  }

  if (passPipeline.hasAnyOccurrences() &&
      postFusionLoopUnrollFactor.getNumOccurrences() > 0) {
    llvm::errs() << "Warning: --post-fusion-loop-unroll-factor is ignored "
                    "when --pass-pipeline is used; set the pass option inside "
                    "the textual pipeline instead.\n";
  }

  if (testOnlyFusionRegionGen) {
    PassManager testPm(&context);
    maybeEnablePrintIRAfterAll(testPm, inputFuncNames);
    testPm.addNestedPass<mlir::func::FuncOp>(pto::createPTOFusionRegionGenPass());
    if (failed(testPm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return 1;
    }
    return 0;
  }

  if (testOnlyOpScheduling) {
    PassManager testPm(&context);
    maybeEnablePrintIRAfterAll(testPm, inputFuncNames);
    testPm.addNestedPass<mlir::func::FuncOp>(pto::createFusionPlanPass());
    testPm.addNestedPass<mlir::func::FuncOp>(pto::createOpSchedulingPass());
    if (failed(testPm.run(*module))) {
      llvm::errs() << "Error: Pass execution failed.\n";
      return 1;
    }
    return 0;
  }

  if (passPipeline.hasAnyOccurrences()) {
    PassManager customPm(&context);
    maybeEnablePrintIRAfterAll(customPm, inputFuncNames);
    if (failed(mlir::applyPassManagerCLOptions(customPm))) {
      llvm::errs() << "Error: Failed to apply pass-manager CLI options.\n";
      return 1;
    }
    if (failed(passPipeline.addToPipeline(customPm, [&](const Twine &msg) {
          llvm::errs() << "Error: " << msg << "\n";
          return failure();
        }))) {
      return 1;
    }
    if (failed(customPm.run(*module))) {
      llvm::errs() << "Error: custom pass pipeline execution failed.\n";
      return 1;
    }

    std::error_code ec;
    llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << ec.message() << "\n";
      return 1;
    }
    module->print(outputFile.os());
    outputFile.keep();
    return 0;
  }

  // A5 backend mainline: lower through PTOToVPTO and never wire the legacy
  // OP-Lib passes into this backend branch.
  if (useVPTOBackendPipeline) {
    const bool skipPreBackendPasses = inputIsVPTOIR;

    if (!skipPreBackendPasses) {
      PassManager preBackendPM(&context);
      maybeEnablePrintIRAfterAll(preBackendPM, inputFuncNames);
      if (enableA5FusionMainline)
        addA5FusionRegionMainlinePreBackendPasses(preBackendPM);
      addSharedPreBackendPasses(preBackendPM, effectiveLevel);
      module->getOperation()->setAttr("pto.target_arch",
                                      mlir::StringAttr::get(&context, arch));
      if (failed(preBackendPM.run(*module))) {
        llvm::errs() << "Error: shared pre-backend pass execution failed.\n";
        return 1;
      }
    }

    if (ptoPrintSeamIR || !ptoSeamIRFile.empty()) {
      if (skipPreBackendPasses) {
        llvm::errs() << "Error: shared pre-backend seam IR is unavailable when "
                        "the input is already VPTO IR.\n";
        return 1;
      }
      if (ptoPrintSeamIR) {
        module->print(llvm::errs());
        llvm::errs() << "\n";
      }
      if (failed(emitSharedPreBackendSeamIR(*module, ptoSeamIRFile)))
        return 1;
    }

    if (!skipPreBackendPasses) {
      PassManager backendPM(&context);
      maybeEnablePrintIRAfterAll(backendPM, inputFuncNames);
      addVPTOBackendMainlinePasses(backendPM, enableA5FusionMainline,
                                   postFusionLoopUnrollFactor);
      backendPM.addNestedPass<mlir::func::FuncOp>(
          pto::createPTOVPTOExpandBridgeOpsPass());
      if (enableA5FusionMainline) {
        backendPM.addPass(createCSEPass());
      }
      if (failed(backendPM.run(*module))) {
        llvm::errs() << "Error: VPTO backend lowering pass execution failed.\n";
        return 1;
      }
    } else if (failed(runVPTOAuthoringValidation(context, *module,
                                                 inputFuncNames))) {
      llvm::errs() << "Error: VPTO authoring-stage legality verification "
                      "failed.\n";
      return 1;
    }

    std::error_code ec;
    llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << ec.message() << "\n";
      return 1;
    }

    if (vptoPrintIR || dumpVPTOIR) {
      printVPTOIROpSummary(*module, llvm::errs());
      module->print(llvm::errs());
      llvm::errs() << "\n";
    }

    pto::VPTOEmissionOptions options;
    options.dumpVPTOIR = vptoPrintIR || dumpVPTOIR;
    options.printIntrinsicSelections = vptoPrintIntrinsics;
    options.allowUnresolved = vptoAllowUnresolved;
    options.unresolvedReportPath = !hivmUnresolvedReport.empty()
                                       ? hivmUnresolvedReport
                                       : vptoUnresolvedReport;
    if (arch == "a5") {
      options.targetTriple = "hiipu64-hisilicon-cce";
      options.march = "dav-c310-vec";
      options.aicoreArch = "dav-c310-vec";
      options.defaultTargetCPU = "dav-c310-vec";
      options.defaultTargetFeatures =
          "+ATOMIC,+ArchV130,+AregRedefinable,+ArithmeticBf16,+AtomicForB8 ,"
          "+F8e4m3,+F8e5m2,+F8e8m0,+FFTSBlk,+Fp4e1m2x2,+Fp4e2m1x2,+LDExtRefine,"
          "+MOVX8,+SPR7bits,+SyncV,+dav-c310-vec";
    }

    if (emitVPTO || vptoEmitHIVMText ||
        (!vptoEmitHIVMOfficialLLVM && !vptoEmitHIVMOfficialBitcode)) {
      FailureOr<OwningOpRef<ModuleOp>> emissionModule =
          pto::prepareVPTOEmissionModule(*module, &llvm::errs());
      if (failed(emissionModule)) {
        llvm::errs() << "Error: VPTO emission preparation failed.\n";
        return 1;
      }

      if (emitVPTO || (!vptoEmitHIVMText && !vptoEmitHIVMOfficialLLVM &&
                       !vptoEmitHIVMOfficialBitcode)) {
        (*emissionModule)->print(outputFile.os());
        outputFile.os() << "\n";
        outputFile.keep();
        return 0;
      }

      if (failed(pto::translateVPTOModuleToText(**emissionModule,
                                                outputFile.os(), options,
                                                llvm::errs()))) {
        llvm::errs() << "Error: Failed to emit VPTO text.\n";
        return 1;
      }
      outputFile.keep();
      return 0;
    }

    LogicalResult emissionStatus =
        vptoEmitHIVMOfficialBitcode
            ? pto::translateVPTOModuleToLLVMBitcode(*module,
                                                    outputFile.os(),
                                                    options, llvm::errs())
            : pto::translateVPTOModuleToLLVMText(*module, outputFile.os(),
                                                 options, llvm::errs());
    if (failed(emissionStatus)) {
      llvm::errs() << "Error: Failed to emit VPTO text.\n";
      return 1;
    }
    outputFile.keep();
    return 0;
  }

  // [Fix] ToolOutputFile Usage
  std::error_code ec;
  llvm::ToolOutputFile outputFile(outputFilename, ec, llvm::sys::fs::OF_None);
  if (ec) {
    llvm::errs() << ec.message() << "\n";
    return 1;
  }

  PassManager preCodegenPm(&context);
  maybeEnablePrintIRAfterAll(preCodegenPm, inputFuncNames);

  preCodegenPm.addNestedPass<mlir::func::FuncOp>(
      pto::createLoweringSyncToPipePass());
  preCodegenPm.addPass(pto::createPTOViewToMemrefPass());

  if (failed(preCodegenPm.run(*module))) {
    llvm::errs() << "Error: Pass execution failed.\n";
    return 1;
  }

  // EmitC path: lower PTO ops directly to PTO-ISA/EmitC without any OP-Lib
  // compatibility stages.
  PassManager pm(&context);
  maybeEnablePrintIRAfterAll(pm, inputFuncNames);
  if (!disableInferLayout)
    pm.addNestedPass<mlir::func::FuncOp>(pto::createInferPTOLayoutPass());
  // Stage 1 has already lowered tile/view values into memref world.
  // Running PTOViewToMemref again here is not idempotent and can fail on IR
  // that already contains lowered bind_tile/partition_view structure.
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
  if (failed(pto::translateToCpp(
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
