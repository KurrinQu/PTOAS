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
    %m0 = pto.simd.tile_to_memref %src0 : @@SRC0_TILE_TYPE@@ to @@SRC0_MEMREF_TYPE@@
    %m1 = pto.simd.tile_to_memref %src1 : @@SRC1_TILE_TYPE@@ to @@SRC1_MEMREF_TYPE@@
    %m2 = pto.simd.tile_to_memref %src2 : @@SRC2_TILE_TYPE@@ to @@SRC2_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant @@SIMD_LANES@@ : index
    %rows = memref.dim %m1, %c0 : @@SRC1_MEMREF_TYPE@@
    %cols = memref.dim %m1, %c1 : @@SRC1_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      // Canonical byte-mask contract:
      // - active lane 0 => false
      // - active lane nonzero => true
      // - tail lanes load as zero via %laneMask/%passiveMask and must not
      //   affect the selected value
      %passiveMask = arith.constant dense<0> : @@SRC0_VECTOR_TYPE@@
      %zeroMask = arith.constant dense<0> : @@SRC0_VECTOR_TYPE@@
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@RESULT_VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %laneMask = vector.create_mask %active : @@MASK_VECTOR_TYPE@@
          %maskBytes = vector.maskedload %m0[%r, %cidx], %laneMask, %passiveMask {pto.simd.vld_dist = "NORM"} : @@SRC0_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@SRC0_VECTOR_TYPE@@ into @@SRC0_VECTOR_TYPE@@
          %maskVec = arith.cmpi ne, %maskBytes, %zeroMask : @@SRC0_VECTOR_TYPE@@
          %lhs = vector.maskedload %m1[%r, %cidx], %laneMask, %passive {pto.simd.vld_dist = "NORM"} : @@SRC1_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@SRC1_VECTOR_TYPE@@ into @@SRC1_VECTOR_TYPE@@
          %rhs = vector.maskedload %m2[%r, %cidx], %laneMask, %passive {pto.simd.vld_dist = "NORM"} : @@SRC2_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@SRC2_VECTOR_TYPE@@ into @@SRC2_VECTOR_TYPE@@
@@COMPUTE@@          vector.maskedstore %md[%r, %cidx], %laneMask, %result {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@RESULT_VECTOR_TYPE@@
        }
      }
    }
    return
  }
