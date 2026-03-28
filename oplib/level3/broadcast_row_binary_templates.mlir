// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: broadcast_row_binary
// Source template: skeletons/broadcast_row_binary.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from broadcast-row-binary skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = broadcast_row_binary
  // axes = dtype=f16, core_op=arith.mulf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpandmul_linear_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpandmul",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<128xf16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %result = arith.mulf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row_binary
  // axes = dtype=f32, core_op=arith.mulf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpandmul_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpandmul",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<64xf32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = arith.mulf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = broadcast_row_binary
  // axes = dtype=f16, core_op=arith.divf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpanddiv_linear_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpanddiv",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<128xf16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %result = arith.divf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row_binary
  // axes = dtype=f32, core_op=arith.divf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpanddiv_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpanddiv",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<64xf32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = arith.divf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = broadcast_row_binary
  // axes = dtype=f16, core_op=arith.subf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpandsub_linear_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpandsub",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<128xf16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16> into vector<128xf16>
          %result = arith.subf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row_binary
  // axes = dtype=f32, core_op=arith.subf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpandsub_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_binary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpandsub",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    // %src0 is the canonical full-tile operand and %src1 is the canonical
    // row-broadcast operand; lowering may reorder external operands to satisfy
    // this family contract.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %m1 = pto.simd.tile_to_memref %src1 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m1[%r, %c0] : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %rhs = vector.splat %rowScalar : vector<64xf32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32> into vector<64xf32>
          %result = arith.subf %lhs, %rhs {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }
}
