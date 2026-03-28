// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: binary
// Source template: skeletons/binary.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from unified int binary skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.addi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tadd_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tadd",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.addi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.addi, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tadd_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tadd",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.addi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.subi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tsub_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tsub",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.subi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.subi, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tsub_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tsub",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.subi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.muli, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tmul_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmul",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.muli %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.muli, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tmul_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmul",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.muli %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.divsi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tdiv_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdiv",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divsi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.divui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tdiv_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdiv",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divui %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.maxsi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tmax_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmax",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.maxsi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.maxui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tmax_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmax",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.maxui %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.minsi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tmin_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmin",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.minsi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.minui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tmin_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmin",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.minui %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.andi, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tand_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tand",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.andi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.andi, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tand_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tand",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.andi %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.ori, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tor_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tor",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.ori %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.ori, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tor_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tor",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.ori %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.xori, variant_id=tile_i8
  func.func private @__pto_oplib_variant_txor_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "txor",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.xori %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.xori, variant_id=tile_u8
  func.func private @__pto_oplib_variant_txor_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "txor",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.xori %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.shli, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tshl_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshl",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shli %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.shli, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tshl_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshl",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shli %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=i8, core_op=arith.shrui, variant_id=tile_i8
  func.func private @__pto_oplib_variant_tshr_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshr",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shrui %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_binary_elementwise
  // axes = dtype=u8, core_op=arith.shrui, variant_id=tile_u8
  func.func private @__pto_oplib_variant_tshr_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %src1: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_binary_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshr",
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
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %rhs = vector.maskedload %m1[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shrui %lhs, %rhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }
}
