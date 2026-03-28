// -----------------------------------------------------------------------------
// AUTO-GENERATED: do not edit directly.
// Source pattern: unary
// Source template: skeletons/unary.instance.tmpl.mlir
// Axes: dtype, core_op, variant_id
// Output role: importer-active concrete templates synchronized from unified int unary skeleton source.
// -----------------------------------------------------------------------------
module {
  // family_id = int_unary
  // axes = dtype=i16, core_op=arith.xori, variant_id=tile_i16
  func.func private @__pto_oplib_variant_tnot_i16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_unary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tnot",
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


        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 0 : index
    %c1 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 1 : index
    %cLanes = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 128 : index
    %rows = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
    pto.simd.vec_scope {
      %passive = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} dense<0> : vector<128xi16>
      %allOnes = arith.constant dense<-1> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %remain = arith.subi %cols, %cidx {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %lt = arith.cmpi slt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %active = arith.select %lt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %mask = vector.create_mask %active {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : vector<128xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %result = arith.xori %lhs, %allOnes : vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
      } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    return
  }

  // family_id = int_unary
  // axes = dtype=u16, core_op=arith.xori, variant_id=tile_u16
  func.func private @__pto_oplib_variant_tnot_u16(
      %src0: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>,
      %dst: !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>
      ) attributes {
        pto.oplib.kind = "l3_int_unary_template",
        pto.oplib.entry_role = "variant",
        pto.oplib.op = "tnot",
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


        pto.oplib.cost = 10 : i64,
        pto.oplib.priority = 0 : i64
      } {
    %m0 = pto.simd.tile_to_memref %src0 : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %md = pto.simd.tile_to_memref %dst : !pto.tile_buf<loc=vec, dtype=i16, rows=32, cols=32, v_row=?, v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0> to memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>

    %c0 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 0 : index
    %c1 = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 1 : index
    %cLanes = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} 128 : index
    %rows = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c0 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %cols = memref.dim {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} %m0, %c1 : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>
    %repeatTimes = arith.ceildivsi %cols, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
    pto.simd.vec_scope {
      %passive = arith.constant {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} dense<0> : vector<128xi16>
      %allOnes = arith.constant dense<-1> : vector<128xi16>
      scf.for %r = %c0 to %rows step %c1 {
        scf.for %j = %c0 to %repeatTimes step %c1 {
          %cidx = arith.muli %j, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %remain = arith.subi %cols, %cidx {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %lt = arith.cmpi slt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %active = arith.select %lt, %remain, %cLanes {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : index
          %mask = vector.create_mask %active {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : vector<128xi1>
          %lhs = vector.maskedload %m0[%r, %cidx], %mask, %passive {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16> into vector<128xi16>
          %result = arith.xori %lhs, %allOnes : vector<128xi16>
          vector.maskedstore %md[%r, %cidx], %mask, %result {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"} : memref<?x?xi16, strided<[32, 1], offset: 0>, #pto.address_space<vec>>, vector<128xi1>, vector<128xi16>
        } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
      } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    } {pto.simd.exec_mode = "MODE_ZEROING", pto.simd.vld_dist = "NORM", pto.simd.vst_dist = "DIST_NORM"}
    return
  }
}
