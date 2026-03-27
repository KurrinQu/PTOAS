//===-- Passes.td - PTO pass definition file ---------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//===----------------------------------------------------------------------===//
//
// TODO
//
//===----------------------------------------------------------------------===//

// The A5VM backend is emitted from tools/ptoas rather than a TableGen pass;
// these registrations continue to describe the shared pre-backend pipeline.

#ifndef MLIR_DIALECT_PTO_PASSES
#define MLIR_DIALECT_PTO_PASSES

include "mlir/Pass/PassBase.td"

def PTOConvertToDPS : Pass<"pto-convert-to-dps", "func::FuncOp"> {
  let summary = "Convert PTO value-semantic ops to DPS ops";
  let description = [{
    Rewrites High-Level PTO ops (load/store/matmul) to their DPS variants 
    by allocating empty tensors as destinations.
  }];

  let constructor = "mlir::pto::createPTOConvertToDPSPass()";

  let dependentDialects = ["tensor::TensorDialect", "pto::PTODialect"];
}

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

def PTOA5VMVersionSelection
    : Pass<"pto-a5vm-version-selection", "func::FuncOp"> {
  let summary = "Select and materialize per-op A5VM lowering variants";
  let description = [{
    Traverses all A5VM candidate PTO ops in a function and annotates each op
    with a per-op `pto.a5vm_lowering_choice` contract for downstream lowering.
  }];
  let constructor = "mlir::pto::createPTOA5VMVersionSelectionPass()";
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

def PTOA5VMIfCanonicalize
    : Pass<"pto-a5vm-if-canonicalize", "func::FuncOp"> {
  let summary =
      "Canonicalize only scf.if cleanup patterns before A5 low-level fusion";
  let description = [{
    Applies the builtin `scf.if` canonicalization patterns to pre-existing
    `scf.if` operations without running the global Canonicalizer. This keeps the
    pre-loop-fusion cleanup focused on removing residual constant-control-flow
    structure while preserving surrounding `scf.for` loop headers.
  }];
  let constructor = "mlir::pto::createPTOA5VMIfCanonicalizePass()";
  let dependentDialects = ["mlir::func::FuncDialect",
                           "mlir::arith::ArithDialect",
                           "mlir::scf::SCFDialect"];
}

def PTOLowLevelLoopFusion : Pass<"pto-low-level-loop-fusion", "ModuleOp"> {
  let summary =
      "Fuse adjacent A5VM post-lowering loop nests inside pto.fusion_region";
  let description = [{
    Conservatively fuses adjacent post-lowering loop stages for already-planned
    fusion groups after PTOToA5VM. The formal input contract for this pass is
    low-level `scf.for + a5vm.*` structure that remains inside
    `pto.fusion_region` until explicit flatten.
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

def PTOFusionPredicateElision
    : Pass<"pto-fusion-predicate-elision", "func::FuncOp"> {
  let summary =
      "Elide redundant fusion-local A5VM plt predicate materialization";
  let description = [{
    Scans `pto.fusion_region` bodies after the A5VM post-lowering CSE step and
    prepares fusion-local redundant `a5vm.plt_*` predicate elimination before
    load/store cleanup and explicit flatten.
  }];
  let constructor = "mlir::pto::createPTOFusionPredicateElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::a5vm::A5VMDialect"];
}

def PTOFusionLoadStoreElision
    : Pass<"pto-fusion-load-store-elision", "func::FuncOp"> {
  let summary =
      "Elide fusion-local A5VM load/store round trips before flattening";
  let description = [{
    Removes local store materialization inside canonical fused loop bodies before
    explicit flatten. The formal input contract is A5VM post-lowering
    `scf.for + a5vm.*` (carrier loop marked with `llvm.loop.aivector_scope`).
    It also performs conservative frontier-aware tail-store cleanup for
    non-escaping fusion-region internal buffers.
  }];
  let constructor = "mlir::pto::createPTOFusionLoadStoreElisionPass()";
  let dependentDialects = ["mlir::func::FuncDialect", "mlir::pto::PTODialect",
                           "mlir::arith::ArithDialect", "mlir::scf::SCFDialect",
                           "mlir::a5vm::A5VMDialect"];
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

def PTOToA5VM : Pass<"pto-to-a5vm", "ModuleOp"> {
  let summary = "Lower PTO tile ops to A5VM backend ops";
  let description = [{
    Lowers PTO tile ops to A5VM backend ops. For already-planned fusion groups,
    the pass rewrites the `pto.fusion_region` body in place and preserves the
    wrapper until explicit flatten. Residual non-fused PTO ops may continue to
    be lowered directly in their parent block and are not wrapped into
    synthetic `pto.fusion_region` containers solely for backend lowering.
  }];
  let constructor = "mlir::pto::createLowerPTOToA5VMPass()";
  let options = [
    Option<"a5vmLoweringStrategy", "a5vm-lowering-strategy", "std::string",
           "\"post-update\"",
           "vector lowering strategy: post-update or no-post-update">
  ];
  let dependentDialects = [
    "pto::PTODialect",
    "a5vm::A5VMDialect",
    "func::FuncDialect",
    "arith::ArithDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect"
  ];
}

#endif // MLIR_DIALECT_PTO_PASSES
