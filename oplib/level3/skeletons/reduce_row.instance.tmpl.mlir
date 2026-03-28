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
        pto.oplib.cost = @@COST@@ : i64,
        pto.oplib.priority = @@PRIORITY@@ : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant @@SIMD_LANES@@ : index
    %rows = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %initVec = arith.constant @@PASSIVE_VECTOR@@ : @@INPUT_VECTOR_TYPE@@
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@INPUT_VECTOR_TYPE@@
      %outMask = vector.create_mask %c1 : @@MASK_VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        %reduced = scf.for %j = %c0 to %repeatTimes step %c1 iter_args(%acc = %initVec) -> (@@INPUT_VECTOR_TYPE@@) {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@ into @@INPUT_VECTOR_TYPE@@
@@COMPUTE@@          scf.yield %result : @@INPUT_VECTOR_TYPE@@
        }
        vector.maskedstore %md[%r, %c0], %outMask, %reduced {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@INPUT_VECTOR_TYPE@@
      }
    }
    return
  }
