// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: broadcast_row
// Source template: skeletons/broadcast_row.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from broadcast-row skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = broadcast_row
  // axes = dtype=bf16, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_bf16(
      %src0: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=bf16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xbf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xbf16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xbf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<128xbf16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %result = arith.addf %lhs, %passive {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xbf16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xbf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xbf16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=f16, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_f16(
      %src0: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<128xf16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xf16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<128xf16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %result = arith.addf %lhs, %passive {pto.simd.exec_mode = "MODE_ZEROING"} : vector<128xf16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xf16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=f32, core_op=arith.addf, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_f32(
      %src0: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=f32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0.0> : vector<64xf32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xf32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<64xf32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %result = arith.addf %lhs, %passive {pto.simd.exec_mode = "MODE_ZEROING"} : vector<64xf32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xf32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xf32>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=i16, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<128xi16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %result = arith.addi %lhs, %passive : vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=i32, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_i32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<64xi32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %result = arith.addi %lhs, %passive : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=i8, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_i8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %rowMask = vector.create_mask %c1 : vector<256xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<256xi8>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %result = arith.addi %lhs, %passive : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=u16, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 128 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<128xi16>
      %rowMask = vector.create_mask %c1 : vector<128xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi16, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<128xi16>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<128xi1>
          %result = arith.addi %lhs, %passive : vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=u32, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_u32(
      %src0: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i32, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 64 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<64xi32>
      %rowMask = vector.create_mask %c1 : vector<64xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi32, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<64xi32>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<64xi1>
          %result = arith.addi %lhs, %passive : vector<64xi32>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi32, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<64xi1>, vector<64xi32>
        }
      }
    }
    return
  }

  // family_id = broadcast_row
  // axes = dtype=u8, core_op=arith.addi, variant_id=linear
  func.func private @__pto_oplib_variant_trowexpand_linear_u8(
      %src0: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_broadcast_row_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "trowexpand",
        pto.oplib.variant_id = "linear",
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

        pto.oplib.cost = 1 : i64,
        pto.oplib.priority = 1 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=1, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i8, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %cLanes = arith.constant 256 : index
    %rows = memref.dim %m0, %c0 : memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim %md, %c1 : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes : index
    pto.simd.vec_scope {
      %passive = arith.constant dense<0> : vector<256xi8>
      %rowMask = vector.create_mask %c1 : vector<256xi1>
      scf.for %r = %c0 to %rows step %c1 {
        %rowScalar = memref.load %m0[%r, %c0] : memref<?x?xi8, strided<[1, 1], offset: 0>, #pto.address_space<vec>>
        %lhs = vector.splat %rowScalar : vector<256xi8>
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes : index
          %remain = arith.subi %cols, %cidx : index
          %lt = arith.cmpi slt, %remain, %cLanes : index
          %active = arith.select %lt, %remain, %cLanes : index
          %mask = vector.create_mask %active : vector<256xi1>
          %result = arith.addi %lhs, %passive : vector<256xi8>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi8, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<256xi1>, vector<256xi8>
        }
      }
    }
    return
  }
}
