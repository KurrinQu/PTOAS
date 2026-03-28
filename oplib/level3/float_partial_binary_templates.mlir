// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: partial_binary
// Source template: skeletons/partial_binary.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from unified partial-binary skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = float_partial_binary
  // axes = dtype=i8, core_op=arith.addi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tpartadd_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.addi %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i16, core_op=arith.addi, variant_id=tile_i16
  func.func private @__pto_oplib_variant_tpartadd_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_i16",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.addi %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i32, core_op=arith.addi, variant_id=tile_i32
  func.func private @__pto_oplib_variant_tpartadd_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_i32",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.addi %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u8, core_op=arith.addi, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tpartadd_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.addi %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u16, core_op=arith.addi, variant_id=tile_u16
  func.func private @__pto_oplib_variant_tpartadd_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_u16",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.addi %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u32, core_op=arith.addi, variant_id=tile_u32
  func.func private @__pto_oplib_variant_tpartadd_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_u32",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.addi %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=bf16, core_op=arith.addf, variant_id=tile_bf16
  func.func private @__pto_oplib_variant_tpartadd_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_bf16",
        pto.oplib.match.dtype = "bf16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xbf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.addf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xbf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xbf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xbf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f16, core_op=arith.addf, variant_id=tile_f16
  func.func private @__pto_oplib_variant_tpartadd_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_f16",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.addf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f32, core_op=arith.addf, variant_id=tile_f32
  func.func private @__pto_oplib_variant_tpartadd_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartadd",
        pto.oplib.variant_id = "tile_f32",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.addf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xf32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i8, core_op=arith.maxsi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tpartmax_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.maxsi %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i16, core_op=arith.maxsi, variant_id=tile_i16
  func.func private @__pto_oplib_variant_tpartmax_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_i16",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.maxsi %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i32, core_op=arith.maxsi, variant_id=tile_i32
  func.func private @__pto_oplib_variant_tpartmax_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_i32",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.maxsi %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u8, core_op=arith.maxui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tpartmax_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.maxui %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u16, core_op=arith.maxui, variant_id=tile_u16
  func.func private @__pto_oplib_variant_tpartmax_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_u16",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.maxui %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u32, core_op=arith.maxui, variant_id=tile_u32
  func.func private @__pto_oplib_variant_tpartmax_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_u32",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.maxui %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=bf16, core_op=arith.maximumf, variant_id=tile_bf16
  func.func private @__pto_oplib_variant_tpartmax_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_bf16",
        pto.oplib.match.dtype = "bf16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xbf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.maximumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xbf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xbf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xbf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f16, core_op=arith.maximumf, variant_id=tile_f16
  func.func private @__pto_oplib_variant_tpartmax_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_f16",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.maximumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f32, core_op=arith.maximumf, variant_id=tile_f32
  func.func private @__pto_oplib_variant_tpartmax_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmax",
        pto.oplib.variant_id = "tile_f32",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.maximumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xf32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i8, core_op=arith.minsi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tpartmin_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.minsi %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i16, core_op=arith.minsi, variant_id=tile_i16
  func.func private @__pto_oplib_variant_tpartmin_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_i16",
        pto.oplib.match.dtype = "i16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.minsi %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=i32, core_op=arith.minsi, variant_id=tile_i32
  func.func private @__pto_oplib_variant_tpartmin_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_i32",
        pto.oplib.match.dtype = "i32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.minsi %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u8, core_op=arith.minui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tpartmin_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<256xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<256xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<256xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<256xi1>
          %result = arith.minui %lhs, %rhs : vector<256xi8>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<256xi1>, vector<256xi8>
          %merged = arith.select %bothValid, %result, %carry : vector<256xi1>, vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u16, core_op=arith.minui, variant_id=tile_u16
  func.func private @__pto_oplib_variant_tpartmin_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_u16",
        pto.oplib.match.dtype = "u16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.minui %lhs, %rhs : vector<128xi16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xi16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=u32, core_op=arith.minui, variant_id=tile_u32
  func.func private @__pto_oplib_variant_tpartmin_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_u32",
        pto.oplib.match.dtype = "u32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32> into vector<64xi32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.minui %lhs, %rhs : vector<64xi32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xi32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=bf16, core_op=arith.minimumf, variant_id=tile_bf16
  func.func private @__pto_oplib_variant_tpartmin_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_bf16",
        pto.oplib.match.dtype = "bf16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xbf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16> into vector<128xbf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.minimumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xbf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xbf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xbf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f16, core_op=arith.minimumf, variant_id=tile_f16
  func.func private @__pto_oplib_variant_tpartmin_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_f16",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %result = arith.minimumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f32, core_op=arith.minimumf, variant_id=tile_f32
  func.func private @__pto_oplib_variant_tpartmin_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tpartmin",
        pto.oplib.variant_id = "tile_f32",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %result = arith.minimumf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xf32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f16, core_op=arith.mulf, variant_id=tile_f16
  func.func private @__pto_oplib_variant_tprelu_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tprelu",
        pto.oplib.variant_id = "tile_f16",
        pto.oplib.match.dtype = "f16",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      %zeroVec = arith.constant dense<0.0> : vector<128xf16>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<128xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<128xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<128xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<128xi1>
          %positive = arith.cmpf ogt, %lhs, %zeroVec : vector<128xf16>
          %scaled = arith.mulf %rhs, %lhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          %result = arith.select %positive, %lhs, %scaled : vector<128xi1>, vector<128xf16>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<128xi1>, vector<128xf16>
          %merged = arith.select %bothValid, %result, %carry : vector<128xi1>, vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = float_partial_binary
  // axes = dtype=f32, core_op=arith.mulf, variant_id=tile_f32
  func.func private @__pto_oplib_variant_tprelu_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_float_partial_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tprelu",
        pto.oplib.variant_id = "tile_f32",
        pto.oplib.match.dtype = "f32",
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg1.rows = -1 : i64,
        pto.oplib.match.arg1.cols = -1 : i64,
        pto.oplib.match.arg1.blayout = "row_major",
        pto.oplib.match.arg1.slayout = "any",
        pto.oplib.match.arg1.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows0 = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols0 = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows1 = memref.dim %m1, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols1 = memref.dim %m1, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %rows = memref.dim %md, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %zeroVec = arith.constant dense<0.0> : vector<64xf32>
      scf.for %r = %c0 to %rows step %c1 {
        %lhsRowValid = arith.cmpi slt, %r, %rows0 : index
        %rhsRowValid = arith.cmpi slt, %r, %rows1 : index
        %lhsRowIndex = arith.select %lhsRowValid, %r, %c0 : index
        %rhsRowIndex = arith.select %rhsRowValid, %r, %c0 : index
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %dstRemain = arith.subi %cols, %cidx : index
          %dstTail = arith.cmpi slt, %dstRemain, %cLanes : index
          %dstActive = arith.select %dstTail, %dstRemain, %cLanes : index
          %mask = vector.create_mask %dstActive : vector<64xi1>

          %lhsColValid = arith.cmpi sgt, %cols0, %cidx : index
          %lhsColIndex = arith.select %lhsColValid, %cidx, %c0 : index
          %lhsRemainRaw = arith.subi %cols0, %cidx : index
          %lhsRemain = arith.select %lhsColValid, %lhsRemainRaw, %c0 : index
          %lhsTail = arith.cmpi slt, %lhsRemain, %cLanes : index
          %lhsActiveBase = arith.select %lhsTail, %lhsRemain, %cLanes : index
          %lhsActive = arith.select %lhsRowValid, %lhsActiveBase, %c0 : index
          %lhsMask = vector.create_mask %lhsActive : vector<64xi1>

          %rhsColValid = arith.cmpi sgt, %cols1, %cidx : index
          %rhsColIndex = arith.select %rhsColValid, %cidx, %c0 : index
          %rhsRemainRaw = arith.subi %cols1, %cidx : index
          %rhsRemain = arith.select %rhsColValid, %rhsRemainRaw, %c0 : index
          %rhsTail = arith.cmpi slt, %rhsRemain, %cLanes : index
          %rhsActiveBase = arith.select %rhsTail, %rhsRemain, %cLanes : index
          %rhsActive = arith.select %rhsRowValid, %rhsActiveBase, %c0 : index
          %rhsMask = vector.create_mask %rhsActive : vector<64xi1>

          %lhs = vector.maskedload %m0[%lhsRowIndex, %lhsColIndex], %lhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %rhs = vector.maskedload %m1[%rhsRowIndex, %rhsColIndex], %rhsMask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %bothValid = arith.andi %lhsMask, %rhsMask : vector<64xi1>
          %positive = arith.cmpf ogt, %lhs, %zeroVec : vector<64xf32>
          %scaled = arith.mulf %rhs, %lhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          %result = arith.select %positive, %lhs, %scaled : vector<64xi1>, vector<64xf32>
          %carry = arith.select %lhsMask, %lhs, %rhs : vector<64xi1>, vector<64xf32>
          %merged = arith.select %bothValid, %result, %carry : vector<64xi1>, vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %merged {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
