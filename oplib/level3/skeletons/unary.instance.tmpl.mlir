  // family_id = @@FAMILY_ID@@
  // axes = @@AXIS_VALUES@@
  func.func private @@@FUNC_NAME@@(
@@ARG_DECLS@@
      ) attributes {
        pto.oplib.kind = "@@KIND@@",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "@@OP_NAME@@",
        pto.oplib.variant_id = "@@VARIANT_ID@@",
        pto.oplib.match.dtype = "@@MATCH_DTYPE@@",
@@MATCH_ATTRS@@
@@EXTRA_TEMPLATE_ATTRS@@
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 0 : index
    %c1 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 1 : index
    %cLanes = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} @@SIMD_LANES@@ : index
    %rows = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
    pto.simd.vec_scope {
      %passive = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} @@PASSIVE_VECTOR@@ : @@VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %remain = arith.subi %cols, %cidx {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %lt = arith.cmpi slt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %active = arith.select %lt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %mask = vector.create_mask %active {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@ into @@VECTOR_TYPE@@
@@COMPUTE@@          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@
        } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
      } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    return
  }
