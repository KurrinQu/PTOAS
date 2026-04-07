// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

// The VPTO backend is emitted from tools/ptoas rather than a TableGen pass;
// these registrations continue to describe the shared pre-backend pipeline.

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

// Define PTOInsertSync Pass
def PTOInsertSync : Pass<"pto-insert-sync", "func::FuncOp"> {
  let summary = "Insert synchronization flags for PTO dialect";
  let description = [{
    Analyzes data dependencies between Cube, Vector and MTE pipelines
    and inserts explicit synchronization instructions.
  }];

  // Declare the constructor function name
  let constructor = "mlir::pto::createPTOInsertSyncPass()";

  let dependentDialects = ["mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect"];
}

def ConvertToPTOOp : Pass<"convert-to-pto-op"> {
  let summary = "Convert Ops from other dialects to PTO Ops";
  let constructor = "mlir::pto::createConvertToPTOOpPass()";
  let dependentDialects = ["arith::ArithDialect", "pto::PTODialect",
                           "tensor::TensorDialect",
  ];
}

def InferPTOLayout : Pass<"pto-infer-layout", "func::FuncOp"> {
  let summary = "Infer GlobalTensor layout (ND/DN/NZ) for make_tensor_view";
  let description = [{
    For each pto.make_tensor_view with fully static shape/stride, compute layout
    (ND/DN/NZ) per design rules and attach the optional `layout` attribute.
    Dynamic stride/shape are left unset.
  }];
  let constructor = "mlir::pto::createInferPTOLayoutPass()";
  let dependentDialects = ["pto::PTODialect", "arith::ArithDialect"];
}

def InferPTOMemScope : Pass<"pto-infer-mem-scope"> {
  let summary = "Infer memory scope for PTO Ops";
  let constructor = "mlir::pto::createInferPTOMemScopePass()";
  let dependentDialects = ["pto::PTODialect", "memref::MemRefDialect", ];
}

def PlanMemory : Pass<"pto-plan-memory", "ModuleOp"> {
  let summary = "Plan memory for PTO Ops";
  let constructor = "mlir::pto::createPlanMemoryPass()";
  let dependentDialects = ["pto::PTODialect", ];
  let options =
      [Option<"memMode", "mem-plan-mode", "pto::MemPlanMode",
              "pto::MemPlanMode::LOCAL_MEM_PLAN",
              "plan mem mode (default is LOCAL_MEM_PLAN)",
              [{::llvm::cl::values(
               clEnumValN(pto::MemPlanMode::LOCAL_MEM_PLAN, "local-mem-plan",
                          "plan mem mode is for memref.alloc"),
               clEnumValN(
                   pto::MemPlanMode::GLOBAL_WORKSPACE_PLAN,
                   "global-work-space-plan",
                   "plan mem mode is for memref_ext.alloc_workspace"))}]>,
       Option<"enableGlobalReuse", "enable-global-workspace-reuse", "bool",
              /*default=*/"false",
              "Enable global workspace reuse ,default : false">,
       Option<"enablePrintMemoryAllocatedSize",
              "enable-print-memory-allocated-size", "bool",
              /*default=*/"false",
              "Enable print memory allocated size, default : false">,
       Option<"restrictInplaceAsISA", "restrict-inplace-as-isa", "bool",
              /*default=*/"false",
              "restrict memory inplace as isa, default : false">,
  ];
}
def PTOLoweringSyncToPipe : Pass<"pto-lowering-sync-to-pipe", "func::FuncOp"> {
  let summary = "Lower high-level sync ops to low-level pipe ops";
  let description = [{
    Converts high-level `pto.record_event` and `pto.wait_event` operations
    into low-level `pto.set_flag` and `pto.wait_flag` operations based on
    hardware pipe mapping logic.
  }];

  let constructor = "mlir::pto::createLoweringSyncToPipePass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def FusionPlan : Pass<"pto-fusion-plan", "func::FuncOp"> {
  let summary = "Build conservative tile-fusion planning groups from block-local analysis";
  let description = [{
    Consumes PreFusionAnalysis results, forms conservative block-local fusion
    groups for currently supported compute ops, and annotates accepted group
    members with:
      - pto.fusion.group_id
      - pto.fusion.order
  }];
  let constructor = "mlir::pto::createFusionPlanPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def OpScheduling : Pass<"pto-op-scheduling", "func::FuncOp"> {
  let summary = "Compact planned fusion groups into block-local contiguous spans";
  let description = [{
    Consumes fusion planning metadata emitted by FusionPlanPass and performs
    block-local instruction scheduling to make each accepted fusion group a
    contiguous span without redefining group membership.
  }];
  let constructor = "mlir::pto::createOpSchedulingPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOFusionRegionGen : Pass<"pto-fusion-region-gen", "func::FuncOp"> {
  let summary = "Wrap one scheduled fusion span into one pto.fusion_region";
  let description = [{
    Consumes scheduled tile-fusion metadata and will encapsulate one contiguous
    fusion group span into one pto.fusion_region without redefining group
    membership or physical order.
  }];
  let constructor = "mlir::pto::createPTOFusionRegionGenPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOVPTOVersionSelection
    : Pass<"pto-vpto-version-selection", "func::FuncOp"> {
  let summary = "Select and materialize per-op VPTO lowering variants";
  let description = [{
    Traverses all VPTO candidate PTO ops in a function and annotates each op
    with a per-op `pto.lowering_choice` contract for downstream lowering.
  }];
  let constructor = "mlir::pto::createPTOVPTOVersionSelectionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect"];
}

def PreFusionAnalysis : Pass<"pto-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Analyze block-local tile-fusion candidates before planning";
  let description = [{
    Builds reusable pre-planning analysis state for tile-fusion work in
    tile_buf world. This pass is analysis-only and does not emit grouping or
    scheduling decisions.
  }];
  let constructor = "mlir::pto::createPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PrintPreFusionAnalysis
    : Pass<"pto-print-pre-fusion-analysis", "func::FuncOp"> {
  let summary = "Print pre-fusion analysis results for testing and debugging";
  let description = [{
    Computes the pre-fusion analysis result and prints a stable textual dump of
    block-local compute nodes, edges, liveness, iteration-domain classes, and
    local boundaries.
  }];
  let constructor = "mlir::pto::createPrintPreFusionAnalysisPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOInstantiateAndLowerToLibCall
    : Pass<"pto-lower-to-oplib-calls", "ModuleOp"> {
  let summary =
      "Instantiate OP-Lib variants and lower PTO binary ops to OP-Lib calls";
  let description = [{
    Reads OP-Lib template functions from --op-lib-dir (or a caller-resolved
    default directory), validates templates, selects/matches OP-Lib entries
    (variant/seed), creates OP-Lib instance symbols, and rewrites PTO binary
    elementwise ops to OP-Lib func.call.
    Template/instance/call interfaces are all !pto.tile_buf.
  }];
  let constructor = "mlir::pto::createPTOInstantiateAndLowerToLibCallPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect"];
  let options = [Option<"opLibDir", "op-lib-dir", "std::string",
                        /*default=*/"\"\"",
                        "Directory containing OP-Lib template .mlir files; "
                        "the caller may resolve a default when omitted">,
                 Option<"opLibManifest", "op-lib-manifest", "std::string",
                        /*default=*/"\"\"",
                        "Path to the A5 OpLib V1 manifest snapshot used to "
                        "classify in-scope and deferred ops">,
                 Option<"debug", "debug", "bool",
                        /*default=*/"false",
                        "Enable verbose debug logging for OP-Lib lowering">];
}

def PTOInlineLibCall : Pass<"pto-inline-libcall", "ModuleOp"> {
  let summary = "Materialize OP-Lib instance bodies and inline OP-Lib calls";
  let description = [{
    Resolves OP-Lib instance declarations generated by OP-Lib lowering,
    materializes instance bodies, and inlines OP-Lib calls into caller/fused
    helper functions. Function signatures stay in !pto.tile_buf form.
  }];
  let constructor = "mlir::pto::createPTOInlineLibCallPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for OP-Lib instantiation/inlining">];
}

def PTOValidateSimdIR : Pass<"pto-validate-simd-ir", "ModuleOp"> {
  let summary = "Validate functions that use pto.simd bridge IR";
  let description = [{
    Validates structural constraints for functions that contain pto.simd ops,
    including:
      - required pto.simd.level / pto.simd.lanes attributes
      - non-empty function body
      - single binary core slot in seed/templates
      - lane consistency between mask and vector values
      - load -> core -> store ordering
  }];
  let constructor = "mlir::pto::createPTOValidateSimdIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect"];
}

def PTOValidateVPTOIR : Pass<"pto-validate-vpto-ir", "ModuleOp"> {
  let summary =
      "Validate authoring-stage VPTO legality before ptr-boundary canonicalization";
  let description = [{
    Runs the authoring-stage VPTO legality verifier on post-mainline VPTO IR.
    This stage keeps the memref-first authoring surface legal, while checking
    the shared structural contracts that must hold before emission-boundary
    canonicalization.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOValidateVPTOEmissionIR
    : Pass<"pto-validate-vpto-emission-ir", "ModuleOp"> {
  let summary =
      "Validate emission-stage VPTO legality after ptr-boundary canonicalization";
  let description = [{
    Runs the emission-stage VPTO legality verifier on ptr-form VPTO IR after
    `PTOVPTOPtrBoundary`. This stage re-checks the shared authoring contracts
    and confirms the final emission surface no longer carries memref boundary
    state or residual bridge scaffold.
  }];
  let constructor = "mlir::pto::createPTOValidateVPTOEmissionIRPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOLowerSimdToVector : Pass<"pto-lower-simd-to-vector", "ModuleOp"> {
  let summary = "Lower pto.simd bridge ops to standard vector/memref/arith ops";
  let description = [{
    Rewrites pto.simd.predicate/load/store/load_pu/store_pu into
    vector/memref/arith equivalents so downstream canonicalization and loop
    fusion can proceed on standard MLIR ops.
  }];
  let constructor = "mlir::pto::createPTOLowerSimdToVectorPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect",
                           "mlir::vector::VectorDialect",
                           "mlir::memref::MemRefDialect"];
}

def PTOVPTOIfCanonicalize
    : Pass<"pto-vpto-if-canonicalize", "func::FuncOp"> {
  let summary =
      "Canonicalize only scf.if cleanup patterns before A5 low-level fusion";
  let description = [{
    Applies the builtin `scf.if` canonicalization patterns to pre-existing
    `scf.if` operations without running the global Canonicalizer. This keeps the
    pre-loop-fusion cleanup focused on removing residual constant-control-flow
    structure while preserving surrounding `scf.for` loop headers.
  }];
  let constructor = "mlir::pto::createPTOVPTOIfCanonicalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOVPTOExpandBridgeOps
    : Pass<"pto-vpto-expand-bridge-ops", "func::FuncOp"> {
  let summary =
      "Expand temporary VPTO bridge ops back to emission-ready VPTO IR";
  let description = [{
    Low-level fusion may keep temporary bridge ops in VPTO IR so legality and
    alias analysis can still see memref-form operands. This pass expands those
    bridge ops back to the existing emission-ready pointer-level VPTO forms
    before backend emission.
  }];
  let constructor = "mlir::pto::createPTOVPTOExpandBridgeOpsPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::memref::MemRefDialect",
                           "mlir::LLVM::LLVMDialect",
                           "mlir::pto::PTODialect"];
}

def PTOVPTOPtrBoundary
    : Pass<"pto-vpto-ptr-boundary", "ModuleOp"> {
  let summary =
      "Canonicalize the final VPTO emission boundary from memref-first IR to ptr ABI";
  let description = [{
    Runs the final emission-boundary ptr canonicalization after the backend
    mainline has finished its memref-first optimization pipeline. This pass
    rewrites eligible memref function arguments to same-space `!pto.ptr`,
    rejects memref function results, canonicalizes supported body-level VPTO
    buffer-like ops to ptr-form, and drops dead boundary/view scaffold such as
    trivial `pto.castptr`, `pto.bind_tile`, `memref.subview`,
    `memref.reinterpret_cast`, and `memref.memory_space_cast` once they become
    unused.
  }];
  let constructor = "mlir::pto::createPTOVPTOPtrBoundaryPass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::pto::PTODialect",
                           "mlir::memref::MemRefDialect"];
}

def PTOFusionMergeVecScope
    : Pass<"pto-fusion-merge-vecscope", "func::FuncOp"> {
  let summary =
      "Merge adjacent VPTO vecscope regions inside pto.fusion_region";
  let description = [{
    Coalesces adjacent sibling `pto.vecscope` regions that belong to the same
    `pto.fusion_region`, while also sinking pure inter-scope setup ops into the
    merged scope body. This normalization preserves one vector-scope carrier
    for each fusion-local stage run so `PTOLowLevelLoopFusion` can continue to
    operate on low-level `scf.for + pto.*` loop nests inside the region body of
    the merged scope.
  }];
  let constructor = "mlir::pto::createPTOFusionMergeVecScopePass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect"];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary =
      "Fuse adjacent VPTO post-lowering loop nests inside pto.fusion_region";
  let description = [{
    Conservatively fuses adjacent post-lowering loop stages for already-planned
    fusion groups after PTOToVPTO. The formal input contract for this pass is
    low-level `scf.for + pto.*` structure nested inside one
    `pto.vecscope` / `pto.strict_vecscope` carrier that remains inside
    `pto.fusion_region` until explicit flatten. A prior vecscope-merge
    normalization may collapse multiple sibling `pto.vecscope` into one
    carrier region first, but this pass MUST still only fuse stages within
    each individual vector-scope body and MUST NOT cross vecscope boundaries.
    Legacy `pto.simd.vec_scope` / `vector.masked_*` bridge IR is no longer the
    documented mainline contract for the A5 backend pipeline.
  }];
  let constructor = "mlir::pto::createPTOLowLevelLoopFusionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::vector::VectorDialect"];
  let options = [Option<
      "debug", "debug", "bool",
      /*default=*/"false",
      "Enable verbose debug logging for low-level loop fusion">];
}

def PTOVPTOTrivialLoopCanonicalize
    : Pass<"pto-vpto-trivial-loop-canonicalize", "func::FuncOp"> {
  let summary =
      "Simplify trivial scf.for loops while preserving aivector-scope carriers";
  let description = [{
    Re-applies the upstream scf.for trivial-loop simplifications after the
    VPTO fusion-mainline canonicalizer, but explicitly skips loops nested under
    `pto.vecscope` so the vector-scope body shape remains available to
    downstream VPTO passes and backend emission.
  }];
  let constructor = "mlir::pto::createPTOVPTOTrivialLoopCanonicalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOFusionPredicateElision
    : Pass<"pto-fusion-predicate-elision", "func::FuncOp"> {
  let summary =
      "Elide redundant fusion-local VPTO plt predicate materialization";
  let description = [{
    Scans `pto.fusion_region` bodies after the VPTO post-lowering CSE step and
    prepares fusion-local redundant `pto.plt_*` predicate elimination before
    load/store cleanup and explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOFusionPredicateElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::pto::PTODialect"];
}

def PTOFusionLoadStoreElision
    : Pass<"pto-fusion-load-store-elision", "func::FuncOp"> {
  let summary =
      "Elide fusion-local VPTO load/store round trips before flattening";
  let description = [{
    Removes local store materialization inside canonical fused loop bodies before
    explicit flatten. The formal input contract is VPTO post-lowering
    `pto.vecscope + pto.*`.
    It also performs conservative frontier-aware tail-store cleanup for
    non-escaping fusion-region internal buffers.
  }];
  let constructor = "mlir::pto::createPTOFusionLoadStoreElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::pto::PTODialect"];
}

def PTOPostFusionLoopUnroll
    : Pass<"pto-post-fusion-loop-unroll", "func::FuncOp"> {
  let summary =
      "Prepare fusion-local post-cleanup carrier loops for pre-flatten unroll";
  let description = [{
    Scans `pto.fusion_region` bodies after low-level loop fusion, CSE,
    predicate elision, and load/store cleanup, and reserves the formal
    pre-flatten insertion point for conservative post-fusion loop unroll.
    The default behavior uses a conservative auto policy over the v1 decision
    set `skip / x2 / x4`. For focused testing and debugging, callers may
    override the factor with `forced-unroll-factor=2|4` while keeping the
    pass scoped to the same fusion-local carrier-loop contract.
    This pass also establishes the fixed cleanup order
    `... -> PTOFusionLoadStoreElision -> PTOPostFusionLoopUnroll
    -> PTOFlattenFusionRegion -> CSE`.
    The formal input contract is fusion-local post-cleanup carrier loops
    represented as low-level `scf.for + pto.*` structure that still remains
    nested inside `pto.fusion_region`; residual non-fused parent-block VPTO ops
    are outside the scope of this pass.
  }];
  let constructor = "mlir::pto::createPTOPostFusionLoopUnrollPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect"];
  let options = [
    Option<"forcedUnrollFactor", "forced-unroll-factor", "unsigned",
           /*default=*/"0",
           "Override the auto strategy with a fixed factor. "
           "Accepted values: 0 (auto), 2, 4">
  ];
}

def PTOFlattenFusionRegion : Pass<"pto-flatten-fusion-region", "func::FuncOp"> {
  let summary = "Flatten pto.fusion_region wrappers back into the parent block";
  let description = [{
    Eliminates residual pto.fusion_region wrappers after region-local lowering
    is complete by moving body operations back to the parent block, replacing
    region results with pto.yield operands, and erasing both pto.yield and the
    wrapper op.
  }];
  let constructor = "mlir::pto::createPTOFlattenFusionRegionPass()";
  let dependentDialects = ["mlir::pto::PTODialect"];
}

def PTOToVPTO : Pass<"pto-to-vpto", "ModuleOp"> {
  let summary = "Lower PTO tile ops to VPTO backend ops";
  let description = [{
    Lowers PTO tile ops to VPTO backend ops. For already-planned fusion groups,
    the pass rewrites the `pto.fusion_region` body in place and preserves the
    wrapper until explicit flatten. Residual non-fused PTO ops may continue to
    be lowered directly in their parent block and are not wrapped into
    synthetic `pto.fusion_region` containers solely for backend lowering.
  }];
  let constructor = "mlir::pto::createLowerPTOToVPTOPass()";
  let options = [
    Option<"loweringStrategy", "pto-lowering-strategy", "std::string",
           "\"post-update\"",
           "vector lowering strategy: post-update or no-post-update">
  ];
  let dependentDialects = [
    "pto::PTODialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

def PTOWrapFunctionsInSections : Pass<"pto-wrap-functions-in-sections", "func::FuncOp"> {
  let summary = "Wrap attributed single-core functions in PTO cube/vector sections";
  let description = [{
    Rewrites frontend-marked functions into the existing PTO section form:
    - `pto.kernel_kind = #pto.kernel_kind<cube>` -> wrap the function body in
      `pto.section.cube`
    - `pto.kernel_kind = #pto.kernel_kind<vector>` -> wrap the function body
      in `pto.section.vector`
  }];

  let constructor = "mlir::pto::createPTOWrapFunctionsInSectionsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def MemrefToTileBuf : Pass<"pto-memref-to-tile-buf", "ModuleOp"> {
  let summary = "Recover tile_buf types from memref + pto.bind_tile metadata";
  let description = [{
    After PTOViewToMemref + PlanMemory + InsertSync, the IR uses memref types
    with pto.bind_tile ops carrying tile metadata (TileBufConfigAttr, valid
    dims).  This pass reconstructs tile_buf types from those anchors so that
    the subsequent Tile->Vector lowering (Expand TileOp) can operate on
    tile_buf semantics.

    The pass does NOT redo memory planning or synchronisation.
  }];
  let constructor = "mlir::pto::createMemrefToTileBufPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOLowerFrontendPipeOps : Pass<"pto-lower-frontend-pipe-ops", "func::FuncOp"> {
  let summary = "Lower frontend TPUSH/TPOP pipe ops to internal pipe ops";
  let description = [{
    Converts frontend `aic/aiv_initialize_pipe`, `tpush_to_*`,
    `tpop_from_*`, and `tfree_from_*` ops into the internal unified pipe IR:
    `initialize_l2l_pipe` / `initialize_l2g2l_pipe`, `tpush`,
    `declare_tile`, `tpop`, and `tfree`.
  }];

  let constructor = "mlir::pto::createPTOLowerFrontendPipeOpsPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect"
  ];
}

def PTOResolveReservedBuffers : Pass<"pto-resolve-reserved-buffers", "ModuleOp"> {
  let summary = "Resolve reserved local buffer addresses and peer pipe flag bases";
  let description = [{
    Runs after `pto-plan-memory`. Assumes `pto.reserve_buffer` base addresses
    have already been planned, then:
    - aligns missing `flag_base` attrs for peer internal pipe init ops
    - rejects internal pipe init ops without explicit `flag_base` when their
      `local_addr` cannot be traced back to `pto.reserve_buffer` /
      `pto.import_reserved_buffer`
    - replaces `pto.reserve_buffer` / `pto.import_reserved_buffer` results with
      resolved constant local addresses
    - erases the frontend-only reserve/import ops before EmitC lowering
  }];

  let constructor = "mlir::pto::createPTOResolveReservedBuffersPass()";

  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::func::FuncDialect",
    "mlir::arith::ArithDialect"
  ];
}

def ExpandTileOp : Pass<"pto-expand-tile-op", "ModuleOp"> {
  let summary = "Expand tile ops into calls to TileLang DSL template functions";
  let description = [{
    Expands tile-level operations (pto.tadd, pto.tsub, etc.) by invoking the
    TileLang Python DSL to instantiate template libraries. The generated
    template functions use tile_buf parameters and contain vector-level
    implementations (pto.vecscope, pto.vlds, pto.vadd, pto.vsts, etc.).

    Each tile op is replaced by a func.call to the generated template function,
    with tile_buf operands passed directly (no type bridging).

    After this pass, the Inline pass inlines template bodies, and
    FoldTileBufIntrinsics resolves tile_buf_addr / tile_valid_rows /
    tile_valid_cols.
  }];
  let constructor = "mlir::pto::createExpandTileOpPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::func::FuncDialect",
    "mlir::scf::SCFDialect",
    "mlir::vector::VectorDialect"
  ];
  let options = [
    Option<"tilelangPath", "tilelang-path", "std::string",
           /*default=*/"\"\"",
           "Path to directory of .py tilelang DSL template files">,
    Option<"tilelangPkgPath", "tilelang-pkg-path", "std::string",
           /*default=*/"\"\"",
           "PYTHONPATH for tilelang_dsl package (added to env)">,
    Option<"pythonExe", "python-exe", "std::string",
           /*default=*/"\"python3\"",
           "Python executable for tilelang DSL invocation">
  ];
}

def FoldTileBufIntrinsics : Pass<"pto-fold-tile-buf-intrinsics", "mlir::func::FuncOp"> {
  let summary = "Fold tile_buf_addr / tile_valid_rows / tile_valid_cols";
  let description = [{
    After TileLang DSL template functions are inlined, the IR contains
    pto.tile_buf_addr, pto.tile_valid_rows, and pto.tile_valid_cols ops
    whose tile_buf operands are now bound to concrete values.

    This pass resolves them:
    - pto.tile_buf_addr → replaced by pto.simd.tile_to_memref (extracts
      memref address from tile_buf)
    - pto.tile_valid_rows → folded to arith.constant if v_row is static,
      or replaced with the dynamic index value from tile_buf
    - pto.tile_valid_cols → same as above for v_col
  }];
  let constructor = "mlir::pto::createFoldTileBufIntrinsicsPass()";
  let dependentDialects = [
    "mlir::pto::PTODialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect"
  ];
}

def PTOVerifyTFree : Pass<"pto-verify-tfree", "func::FuncOp"> {
  let summary = "Verify explicit matching pto.tfree placement for pto.tpop";
  let description = [{
    For each `pto.tpop` in section.cube / section.vector, verify there is a
    matching later `pto.tfree(pipe_handle)` in the same block, the borrowed
    tile is not used after the free, and the same pipe does not accumulate
    multiple outstanding pops before the matching free.
  }];

  let constructor = "mlir::pto::createPTOVerifyTFreePass()";

  let dependentDialects = [
    "mlir::pto::PTODialect"
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
