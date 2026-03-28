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
    %m1 = pto.simd.tile_to_memref %src1 : @@INPUT_TILE_TYPE@@ to @@INPUT_MEMREF_TYPE@@
    %md = pto.simd.tile_to_memref %dst : @@RESULT_TILE_TYPE@@ to @@RESULT_MEMREF_TYPE@@

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant @@SIMD_LANES@@ : index
    %rows0 = memref.dim %m0, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols0 = memref.dim %m0, %c1 : @@INPUT_MEMREF_TYPE@@
    %rows1 = memref.dim %m1, %c0 : @@INPUT_MEMREF_TYPE@@
    %cols1 = memref.dim %m1, %c1 : @@INPUT_MEMREF_TYPE@@
    %rows = memref.dim %md, %c0 : @@RESULT_MEMREF_TYPE@@
    %cols = memref.dim %md, %c1 : @@RESULT_MEMREF_TYPE@@
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant @@PASSIVE_VECTOR@@ : @@VECTOR_TYPE@@
@@EXTRA_SETUP@@      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : @@MASK_VECTOR_TYPE@@

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : @@MASK_VECTOR_TYPE@@

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : @@MASK_VECTOR_TYPE@@

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@ into @@VECTOR_TYPE@@
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : @@INPUT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@ into @@VECTOR_TYPE@@
          %bothValid = arith.andi %lhsMask, %rhsMask : @@MASK_VECTOR_TYPE@@
@@COMPUTE@@          %carry = arith.select %lhsMask, %lhs, %rhs : @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@
          %merged = arith.select %bothValid, %result, %carry : @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : @@RESULT_MEMREF_TYPE@@, @@MASK_VECTOR_TYPE@@, @@VECTOR_TYPE@@
        }
      }
    }
    return
  }
