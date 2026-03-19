// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: tile_scalar
// Source template: skeletons/tile_scalar.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from unified int tile-scalar skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.divsi, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tdivs_tile_scalar_i8_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdivs",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divsi %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.divsi, variant_id=scalar_tile_i8
  func.func private @__pto_oplib_variant_tdivs_scalar_tile_i8_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdivs",
        pto.oplib.variant_id = "scalar_tile_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divsi %scalarVec, %lhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.divui, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tdivs_tile_scalar_u8_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdivs",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divui %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.divui, variant_id=scalar_tile_u8
  func.func private @__pto_oplib_variant_tdivs_scalar_tile_u8_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tdivs",
        pto.oplib.variant_id = "scalar_tile_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.divui %scalarVec, %lhs : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.maxsi, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tmaxs_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmaxs",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.maxsi %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.maxui, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tmaxs_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmaxs",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.maxui %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.minsi, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tmins_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmins",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.minsi %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.minui, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tmins_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tmins",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.minui %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.andi, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tands_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tands",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.andi %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.andi, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tands_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tands",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.andi %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.ori, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tors_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tors",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.ori %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.ori, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tors_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tors",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.ori %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.xori, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_txors_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "txors",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.xori %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.xori, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_txors_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "txors",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.xori %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.shli, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tshls_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshls",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shli %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.shli, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tshls_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshls",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shli %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=i8, core_op=arith.shrui, variant_id=tile_scalar_i8
  func.func private @__pto_oplib_variant_tshrs_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshrs",
        pto.oplib.variant_id = "tile_scalar_i8",
        pto.oplib.match.dtype = "i8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shrui %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = int_tile_scalar_elementwise
  // axes = dtype=u8, core_op=arith.shrui, variant_id=tile_scalar_u8
  func.func private @__pto_oplib_variant_tshrs_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %scalar: i8,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_tile_scalar_elementwise_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tshrs",
        pto.oplib.variant_id = "tile_scalar_u8",
        pto.oplib.match.dtype = "u8",
        pto.oplib.match.scalar_pos = 1 : i64,
        pto.oplib.match.arg0.rows = -1 : i64,
        pto.oplib.match.arg0.cols = -1 : i64,
        pto.oplib.match.arg0.blayout = "row_major",
        pto.oplib.match.arg0.slayout = "any",
        pto.oplib.match.arg0.fractal = -1 : i64,
        pto.oplib.match.arg2.rows = -1 : i64,
        pto.oplib.match.arg2.cols = -1 : i64,
        pto.oplib.match.arg2.blayout = "row_major",
        pto.oplib.match.arg2.slayout = "any",
        pto.oplib.match.arg2.fractal = -1 : i64,

        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    // variant_id / scalar_pos preserve external operand direction before this
    // canonical tile + splat-scalar skeleton is selected.
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %m0, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %scalarVec = vector.splat %scalar : vector<256xi8>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.vld_dist = "NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8> into vector<256xi8>
          %result = arith.shrui %lhs, %scalarVec : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }
}
