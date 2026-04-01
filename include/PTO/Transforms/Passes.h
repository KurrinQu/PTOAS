//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// TODO.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H

#include "PTO/IR/PTO.h"
#include "PTO/IR/PTODialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace pto {

#define GEN_PASS_DECL
#include "PTO/Transforms/Passes.h.inc"

enum class PTOArch {
  A3,
  A5,
};

// EmitC remains the pass-driven backend; the VPTO backend is emitted directly
// from tools/ptoas at the final backend boundary.

std::unique_ptr<Pass> createPTOHighDimLoweringPass();
std::unique_ptr<Pass> createPTOVFloopGatherPass();
std::unique_ptr<Pass> createLoweringSyncToPipePass();

// Creates a pass for ...
std::unique_ptr<Pass> createPTOInsertSyncPass();
// Default arch is A3 unless overridden by callers.
std::unique_ptr<Pass> createEmitPTOManualPass();
// Explicitly select target arch for codegen.
std::unique_ptr<Pass> createEmitPTOManualPass(PTOArch arch);

/// Create a pass to convert ops from other dialects to PTO Ops.
std::unique_ptr<Pass> createConvertToPTOOpPass();

/// Create a pass to infer, propagate, and add memory scope information to
/// PTO Ops.
std::unique_ptr<Pass> createInferPTOMemScopePass();

/// Create a pass to plan memory.
std::unique_ptr<Pass>
createPlanMemoryPass(const PlanMemoryOptions &planMemoryOption = {});

std::unique_ptr<Pass> createPTOInsertCVMovPass();
std::unique_ptr<Pass> createPTOConvertToDPSPass();
std::unique_ptr<Pass> createPTORemoveRedundantBarrierPass();
std::unique_ptr<Pass> createPTOViewToMemrefPass();
std::unique_ptr<mlir::Pass> createPTOInsertLoadStoreForMixCVPass();
std::unique_ptr<Pass> createInferPTOLayoutPass();
std::unique_ptr<Pass> createPreFusionAnalysisPass();
std::unique_ptr<Pass> createPrintPreFusionAnalysisPass();
std::unique_ptr<Pass> createFusionPlanPass();
std::unique_ptr<Pass> createOpSchedulingPass();
std::unique_ptr<Pass> createPTOFusionRegionGenPass();
std::unique_ptr<Pass> createPTOVPTOVersionSelectionPass();
std::unique_ptr<Pass> createPTOVPTOIfCanonicalizePass();
std::unique_ptr<Pass> createPTOVPTOExpandBridgeOpsPass();
std::unique_ptr<Pass> createPTOVPTOPtrBoundaryPass();
LogicalResult importPTOOpLibTemplates(ModuleOp module, StringRef opLibDir,
                                      bool debug = false);
std::unique_ptr<Pass> createPTOInstantiateAndLowerToLibCallPass(
    const PTOInstantiateAndLowerToLibCallOptions &options = {});
std::unique_ptr<Pass>
createPTOInlineLibCallPass(const PTOInlineLibCallOptions &options = {});
std::unique_ptr<Pass> createPTOValidateVPTOIRPass();
std::unique_ptr<Pass> createPTOValidateVPTOEmissionIRPass();
std::unique_ptr<Pass> createPTOValidateSimdIRPass();
std::unique_ptr<Pass> createPTOLowerSimdToVectorPass();
std::unique_ptr<Pass> createPTOFusionMergeVecScopePass();
std::unique_ptr<Pass> createPTOLowLevelLoopFusionPass(
    const PTOLowLevelLoopFusionOptions &options = {});
std::unique_ptr<Pass> createPTOVPTOTrivialLoopCanonicalizePass();
std::unique_ptr<Pass> createPTOFusionPredicateElisionPass();
std::unique_ptr<Pass> createPTOFusionLoadStoreElisionPass();
std::unique_ptr<Pass> createPTOPostFusionLoopUnrollPass();
std::unique_ptr<Pass> createPTOPostFusionLoopUnrollPass(
    const PTOPostFusionLoopUnrollOptions &options);
std::unique_ptr<Pass> createPTOFlattenFusionRegionPass();
std::unique_ptr<Pass> createLowerPTOToVPTOPass();
std::unique_ptr<Pass> createLowerPTOToVPTOPass(StringRef loweringStrategy);

} // namespace pto
} // namespace mlir

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

#undef GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "PTO/Transforms/Passes.h.inc"

#endif // MLIR_DIALECT_PTO_TRANSFORMS_PASSES_H
