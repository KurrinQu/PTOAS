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
        pto.oplib.match.cmp_mode = "@@CMP_MODE@@",
        pto.simd.level = "binary_ewise_v1",
        pto.simd.lanes = @@SIMD_LANES@@ : i64,
@@SCALAR_POS_ATTR@@@@MATCH_ATTRS@@
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
@@RHS_MEMREF_CAST@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant @@SIMD_LANES@@ : index
    %rows = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@INPUT_VECTOR_TYPE@@
@@RHS_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@ into @@INPUT_VECTOR_TYPE@@
@@RHS_LOAD@@@@COMPARE_STORE@@
        }
      }
    }
    return
  }
