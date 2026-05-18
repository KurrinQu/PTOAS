#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import re
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto
from ptodsl._bootstrap import make_context
from ptodsl._tracing import current_session
from mlir.ir import Location


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


@pto.jit(target="a5")
def host_vec_copy(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A, shape=A.shape, strides=A.strides)
    o_view = pto.make_tensor_view(O, shape=O.shape, strides=O.strides)
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tload(part, a_tile)
    pto.tstore(o_tile, out)


@pto.jit(target="a5")
def runtime_metadata_kernel(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A)
    o_view = pto.make_tensor_view(O)
    a_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    o_tile = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32, valid_shape=[rows, cols])
    part = pto.partition_view(a_view, offsets=[0, 0], sizes=[rows, cols])
    out = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    pto.tload(part, a_tile)
    pto.tstore(o_tile, out)


SUBKERNEL_OBSERVATIONS = []


@pto.simd
def nested_simd_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))


@pto.cube
def top_level_cube_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))


@pto.ukernel
def ukernel_probe():
    session = current_session()
    frame = session.current_subkernel
    SUBKERNEL_OBSERVATIONS.append((frame.role, frame.symbol_name, session.subkernel_stack_depth))
    nested_simd_probe()


@pto.jit(target="a5")
def shared_subkernel_lowering_probe(*, TRACE_TOKEN: pto.constexpr = 0):
    top_level_cube_probe()
    ukernel_probe()
    nested_simd_probe()


@pto.simt
def simt_tid_probe():
    pto.get_tid_x()
    pto.get_tid_y()
    pto.get_tid_z()


@pto.jit(target="a5")
def simt_helper_lowering_probe(*, TRACE_TOKEN: pto.constexpr = 0):
    simt_tid_probe()
    simt_tid_probe()


@pto.jit(target="a5")
def carry_loop_lowering_probe(*, BLOCK: pto.constexpr = 128):
    m_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_prev = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    m_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    l_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    o_next = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)

    m_prev.fill(0.0)
    l_prev.fill(0.0)
    o_prev.fill(0.0)

    kv_loop = pto.for_(0, 4, step=1).carry(m=m_prev, l=l_prev, o=o_prev)
    with kv_loop:
        kv_loop.m.fill(1.0)
        kv_loop.l.fill(2.0)
        kv_loop.o.fill(3.0)
        kv_loop.update(m=m_next, l=l_next, o=o_next)

    final_o = kv_loop.final("o")
    final_o.fill(4.0)


@pto.jit(target="a5")
def runtime_scalar_operator_probe(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 8,
):
    rows = A.shape[0]
    cols = A.shape[1]
    block_idx = pto.get_block_idx()
    o_view = pto.make_tensor_view(O)
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[rows, cols])
    o_ptr = o_part.as_ptr()

    batch_idx = block_idx // rows
    head_idx = block_idx % rows
    chunks = (cols + BLOCK - 1) // BLOCK
    tail = cols % BLOCK

    x = pto.const(2.0, dtype=pto.f32)
    y = (x + 1.0) * 2.0
    z = 4.0 - y
    w = 1.0 / z
    m = pto.scalar.max(w, x)
    e = pto.scalar.exp(m)
    pto.scalar.store(e, o_ptr + 0)

    _ = batch_idx
    _ = head_idx
    _ = chunks
    _ = tail
    _ = w
    _ = m
    _ = e


@pto.simd
def tile_slice_vector_probe(inp_tile: pto.Tile, out_tile: pto.Tile, row: pto.index):
    mask, _ = pto.plt_b32(pto.const(64, dtype=pto.i32))
    vec = pto.vlds(inp_tile[row, 0:])
    pto.vsts(vec, out_tile[row, 0:], mask)


@pto.jit(target="a5")
def tile_slice_surface_probe(*, BLOCK: pto.constexpr = 128):
    inp_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    out_tile = pto.alloc_tile(shape=[2, BLOCK], dtype=pto.f32)
    with pto.for_(0, 1, step=1) as row:
        tile_slice_vector_probe(inp_tile, out_tile, row)


@pto.jit(target="a5")
def tile_valid_shape_update_probe(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    *,
    BLOCK: pto.constexpr = 128,
):
    rows = A.shape[0]
    cols = A.shape[1]
    tile = pto.alloc_tile(
        shape=[1, BLOCK],
        dtype=pto.f32,
        valid_shape=[pto.const(1), cols],
    )
    tile.valid_shape = [rows, cols]


@pto.jit(target="a5")
def integer_loop_bound_probe(*, BLOCK: pto.constexpr = 8):
    row_start = pto.const(0, dtype=pto.i32)
    row_stop = pto.const(BLOCK, dtype=pto.i32)
    valid_dim = pto.const(BLOCK // 2, dtype=pto.i32)
    with pto.for_(row_start, row_stop, step=1) as row:
        with pto.for_(0, valid_dim, step=1) as col:
            _ = row
            _ = col


@pto.jit(target="a5")
def scalar_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 3])
    meta_ptr = meta_tile.as_ptr()
    pto.scalar.store(0, meta_ptr, 0)
    pto.scalar.store(1, meta_ptr, 1)
    pto.scalar.store(2, meta_ptr + 2)
    row_start = pto.scalar.load(meta_ptr, 0)
    row_stop = pto.scalar.load(meta_ptr, 1)
    valid_cols = pto.scalar.load(meta_ptr + 2)
    _ = row_start
    _ = row_stop
    _ = valid_cols


@pto.simt
def simt_pointer_offset_helper(meta_ptr: pto.ptr(pto.i32, pto.MemorySpace.UB)):
    pto.scalar.store(7, meta_ptr + 0)
    pto.scalar.store(9, meta_ptr + 1)


@pto.jit(target="a5")
def simt_pointer_offset_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 2])
    simt_pointer_offset_helper(meta_tile.as_ptr())
    first = pto.scalar.load(meta_tile.as_ptr() + 0)
    second = pto.scalar.load(meta_tile.as_ptr() + 1)
    _ = first
    _ = second


@pto.jit(target="a5")
def scalar_store_element_coercion_probe():
    meta_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.i32, valid_shape=[1, 4])
    meta_ptr = meta_tile.as_ptr()
    row_start = pto.const(0)
    row_stop = pto.const(4)
    pto.scalar.store(row_start, meta_ptr + 0)
    pto.scalar.store(row_stop, meta_ptr + 1)
    pto.scalar.store(pto.const(2, dtype=pto.i64), meta_ptr + 2)
    pto.scalar.store(3, meta_ptr + 3)


@pto.simd
def public_vector_surface_probe(inp_tile: pto.Tile, out_tile: pto.Tile, stats_tile: pto.Tile):
    col_mask = pto.make_mask(pto.f32, pto.const(16, dtype=pto.i32))
    row = pto.const(0)
    s_row = pto.vlds(inp_tile[row, 0:])
    row_max = pto.vcgmax(s_row, col_mask)
    s_shifted = pto.vsubs(s_row, row_max, col_mask)
    p_row = pto.vexp(s_shifted, col_mask)
    row_sum = pto.vcgadd(p_row, col_mask)
    pto.vsts(p_row, out_tile[row, 0:], col_mask)
    pto.scalar.store(row_max, stats_tile[row, 0])
    pto.scalar.store(row_sum, stats_tile[row, 1])


@pto.cube
def public_cube_surface_probe(
    lhs_tile: pto.Tile,
    rhs_tile: pto.Tile,
    lhs_l0a: pto.Tile,
    rhs_l0b: pto.Tile,
    acc_tile: pto.Tile,
    out_tile: pto.Tile,
):
    m = pto.const(16)
    k = pto.const(16)
    n = pto.const(16)
    pto.mte_l1_l0a(lhs_tile.as_ptr(), lhs_l0a.as_ptr(), m, k)
    pto.mte_l1_l0b(rhs_tile.as_ptr(), rhs_l0b.as_ptr(), k, n, transpose=True)
    pto.mad(lhs_l0a.as_ptr(), rhs_l0b.as_ptr(), acc_tile.as_ptr(), m, n, k)
    pto.mte_l0c_ub(acc_tile.as_ptr(), out_tile.as_ptr(), m, n, n, n, 0)


@pto.ukernel
def public_mte_surface_probe(
    inp_part: pto.PartitionTensorView,
    out_part: pto.PartitionTensorView,
    dma_tile: pto.Tile,
):
    pto.mte_load(inp_part, dma_tile)
    pto.pipe_barrier(pto.Pipe.ALL)
    pto.mte_store(dma_tile, out_part)
    pto.mem_bar(pto.BarrierType.VST_VLD)
    pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def public_surface_exports_probe(
    A: pto.tensor_spec(rank=2, dtype=pto.f32),
    O: pto.tensor_spec(rank=2, dtype=pto.f32),
):
    cols = A.shape[1]
    a_view = pto.make_tensor_view(A)
    o_view = pto.make_tensor_view(O)
    a_part = pto.partition_view(a_view, offsets=[0, 0], sizes=[1, cols])
    o_part = pto.partition_view(o_view, offsets=[0, 0], sizes=[1, cols])

    dma_tile = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, cols])
    public_mte_surface_probe(a_part, o_part, dma_tile)

    vec_in = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    vec_out = pto.alloc_tile(shape=[1, 128], dtype=pto.f32, valid_shape=[1, 16])
    stats_tile = pto.alloc_tile(shape=[1, 8], dtype=pto.f32, valid_shape=[1, 2])
    public_vector_surface_probe(vec_in, vec_out, stats_tile)

    lhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    rhs_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.MAT,
        valid_shape=[16, 16],
    )
    lhs_l0a = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.LEFT,
        valid_shape=[16, 16],
    )
    rhs_l0b = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f16,
        memory_space=pto.MemorySpace.RIGHT,
        valid_shape=[16, 16],
    )
    acc_tile = pto.alloc_tile(
        shape=[16, 16],
        dtype=pto.f32,
        memory_space=pto.MemorySpace.ACC,
        valid_shape=[16, 16],
    )
    cube_out = pto.alloc_tile(shape=[16, 16], dtype=pto.f32, valid_shape=[16, 16])
    public_cube_surface_probe(lhs_tile, rhs_tile, lhs_l0a, rhs_l0b, acc_tile, cube_out)


class _FakeTensor:
    def __init__(self, shape):
        self.shape = tuple(shape)

    def new_empty(self, shape):
        return _FakeTensor(shape)


def main() -> None:
    expected_public_exports = [
        "make_mask",
        "vexp",
        "vcgmax",
        "vcgadd",
        "vsubs",
        "mte_load",
        "mte_store",
        "mem_bar",
        "BarrierType",
        "Pipe",
        "pipe_barrier",
        "mte_l1_l0a",
        "mte_l1_l0b",
        "mte_l0c_ub",
        "mad",
        "empty_like",
    ]
    for name in expected_public_exports:
        expect(hasattr(pto, name), f"pto.{name} should be exported from the public namespace")

    fake_tensor = _FakeTensor((2, 3, 4))
    fake_empty = pto.empty_like(fake_tensor)
    expect(isinstance(fake_empty, _FakeTensor), "pto.empty_like(...) should preserve host tensor factory type")
    expect(fake_empty.shape == fake_tensor.shape, "pto.empty_like(...) should preserve the logical tensor shape")
    expect(not hasattr(pto.scalar, "sts"), "scalar.sts should not remain in the public scalar namespace")

    with make_context() as ctx, Location.unknown(ctx):
        tile_buf_ty = pto.tile_buf_type(
            [16, 32],
            pto.f32,
            [16, 8],
            address_space="mat",
            blayout="ColMajor",
            slayout="RowMajor",
        )
        expect(hasattr(tile_buf_ty, "memory_space"), "TileBufType should expose a memory_space accessor")
        expect(hasattr(tile_buf_ty, "shape"), "TileBufType should expose a shape accessor")
        expect(hasattr(tile_buf_ty, "valid_shape"), "TileBufType should expose a valid_shape accessor")
        expect(hasattr(tile_buf_ty, "element_type"), "TileBufType should expose an element_type accessor")
        expect(tile_buf_ty.memory_space.value == pto.MemorySpace.MAT.value, "TileBufType.memory_space should preserve the authored address space")
        expect(list(tile_buf_ty.shape) == [16, 32], "TileBufType.shape should preserve the authored physical shape")
        expect(list(tile_buf_ty.valid_shape) == [16, 8], "TileBufType.valid_shape should preserve the authored valid shape")
        expect(str(tile_buf_ty.element_type) == "f32", "TileBufType.element_type should preserve the authored element type")

    host_vec_copy.verify()
    runtime_metadata_kernel.verify()
    shared_subkernel_lowering_probe.verify()
    simt_helper_lowering_probe.verify()
    carry_loop_lowering_probe.verify()
    runtime_scalar_operator_probe.verify()
    tile_slice_surface_probe.verify()
    tile_valid_shape_update_probe.verify()
    integer_loop_bound_probe.verify()
    scalar_pointer_offset_probe.verify()
    simt_pointer_offset_probe.verify()
    scalar_store_element_coercion_probe.verify()
    public_surface_exports_probe.verify()

    default_compiled = host_vec_copy.compile()
    explicit_default = host_vec_copy.compile(BLOCK=128)
    block64 = host_vec_copy.compile(BLOCK=64)

    expect(default_compiled is explicit_default, "default constexpr compile should hit specialization cache")
    expect(default_compiled is not block64, "different constexpr values should materialize different specializations")
    expect(len(host_vec_copy.cached_specializations()) == 2, "expected exactly two cached specializations")
    expect(default_compiled.constexpr_bindings == {"BLOCK": 128}, "default constexpr binding mismatch")
    expect(block64.constexpr_bindings == {"BLOCK": 64}, "BLOCK=64 constexpr binding mismatch")
    expect(
        default_compiled.specialization_key.abi_signature == block64.specialization_key.abi_signature,
        "ABI signature should stay stable across constexpr-only specializations",
    )
    expect(
        default_compiled.specialization_key.constexpr_signature
        != block64.specialization_key.constexpr_signature,
        "constexpr specialization key should differ when BLOCK changes",
    )

    default_text = default_compiled.mlir_text()
    block64_text = block64.mlir_text()
    expect("!pto.tile_buf<vec, 1x128xf32>" in default_text, "default specialization MLIR missing BLOCK=128 tile")
    expect("!pto.tile_buf<vec, 1x64xf32>" in block64_text, "BLOCK=64 specialization MLIR missing specialized tile")
    expect("valid=?" not in default_text, "default alloc_tile() should keep full static valid-shape when valid_shape= is omitted")

    runtime_metadata_text = runtime_metadata_kernel.compile().mlir_text()
    expect(
        "pto.make_tensor_view %arg0, shape = [%arg1, %arg2], strides = [%arg3, %arg4]" in runtime_metadata_text,
        "make_tensor_view(A) should materialize runtime shape/stride metadata from the tensor proxy",
    )
    expect(
        "pto.alloc_tile valid_row = %arg1 valid_col = %arg2 : !pto.tile_buf<vec, 1x128xf32, valid=?x?>" in runtime_metadata_text,
        "alloc_tile(valid_shape=[rows, cols]) should lower runtime metadata through valid_row/valid_col operands",
    )
    expect(
        "sizes = [%arg1, %arg2]" in runtime_metadata_text,
        "partition_view sizes derived from tensor metadata should remain runtime MLIR values",
    )

    tile_valid_shape_text = tile_valid_shape_update_probe.compile().mlir_text()
    expect(
        re.search(
            r"pto\.set_validshape %[0-9]+, %arg1, %arg2 : !pto\.tile_buf<vec, 1x128xf32, valid=\?x\?>",
            tile_valid_shape_text,
        ) is not None,
        "tile.valid_shape = [rows, cols] should lower to pto.set_validshape on a dynamic-valid tile",
    )

    SUBKERNEL_OBSERVATIONS.clear()
    shared_subkernel_lowering_probe.compile(TRACE_TOKEN=1)
    expect(
        SUBKERNEL_OBSERVATIONS == [
            ("cube", "top_level_cube_probe", 1),
            ("ukernel", "ukernel_probe", 1),
            ("simd", "nested_simd_probe", 2),
            ("simd", "nested_simd_probe", 1),
        ],
        f"unexpected shared subkernel lowering observations: {SUBKERNEL_OBSERVATIONS!r}",
    )

    simt_text = simt_helper_lowering_probe.compile(TRACE_TOKEN=1).mlir_text()
    expect(
        simt_text.count("pto.store_vfsimt_info") == 2,
        "each @pto.simt callsite should materialize a caller-side store_vfsimt_info",
    )
    expect(
        simt_text.count("call @simt_tid_probe()") == 2,
        "each @pto.simt callsite should lower to a func.call of the helper symbol",
    )
    expect(
        simt_text.count("func.func @simt_tid_probe() attributes {pto.simt_entry}") == 1,
        "@pto.simt helper should materialize exactly one reusable pto.simt_entry function",
    )
    expect("pto.get_tid_x" in simt_text, "SIMT helper body should contain pto.get_tid_x")
    expect("pto.get_tid_y" in simt_text, "SIMT helper body should contain pto.get_tid_y")
    expect("pto.get_tid_z" in simt_text, "SIMT helper body should contain pto.get_tid_z")

    carry_text = carry_loop_lowering_probe.compile(BLOCK=32).mlir_text()
    expect("scf.for" in carry_text, "carry loop should lower to scf.for")
    expect("iter_args(" in carry_text, "carry loop should lower named state through scf.for iter_args")
    expect("scf.yield" in carry_text, "carry loop should lower loop.update(...) to scf.yield")
    expect(
        carry_text.count("!pto.tile_buf<vec, 1x32xf32>") >= 3,
        "carry loop MLIR should materialize the specialized carried tile types",
    )
    expect(
        re.search(r"outs\(%[^\s]+#2 : !pto\.tile_buf<vec, 1x32xf32>\)", carry_text) is not None,
        "loop.final(\"o\") should materialize the third scf.for result as the final carried state",
    )

    runtime_scalar_text = runtime_scalar_operator_probe.compile(BLOCK=8).mlir_text()
    expect("arith.index_cast" in runtime_scalar_text, "mixed i64/index runtime arithmetic should materialize index_cast")
    expect("arith.floordivsi" in runtime_scalar_text, "runtime // should lower to arith.floordivsi")
    expect("arith.remsi" in runtime_scalar_text, "runtime % should lower to arith.remsi")
    expect("arith.addf" in runtime_scalar_text, "runtime float + should lower to arith.addf")
    expect("arith.mulf" in runtime_scalar_text, "runtime float * should lower to arith.mulf")
    expect("arith.subf" in runtime_scalar_text, "runtime float - should lower to arith.subf")
    expect("arith.divf" in runtime_scalar_text, "runtime float / should lower to arith.divf")
    expect("arith.maximumf" in runtime_scalar_text, "scalar.max(float, float) should lower to arith.maximumf")
    expect("math.exp" in runtime_scalar_text, "scalar.exp(...) should lower to math.exp")
    expect("pto.store" in runtime_scalar_text, "scalar.store(...) should lower to pto.store")

    tile_slice_text = tile_slice_surface_probe.compile(BLOCK=128).mlir_text()
    expect("memref.subview" in tile_slice_text, "tile[row, col:] should lower through memref.subview")
    expect("memref.collapse_shape" in tile_slice_text, "2D tile[row, col:] should flatten through memref.collapse_shape")
    expect("pto.tile_buf_addr" in tile_slice_text, "tile[row, col:] should materialize a memref tile address view")
    expect(
        "pto.vlds" in tile_slice_text and "memref<128xf32, strided<[1], offset: ?>, #pto.address_space<vec>>" in tile_slice_text,
        "vlds(tile[row, col:]) should lower against the memref slice view",
    )
    expect(
        "pto.vsts" in tile_slice_text and "memref<128xf32, strided<[1], offset: ?>, #pto.address_space<vec>>" in tile_slice_text,
        "vsts(vec, tile[row, col:], mask) should lower against the memref slice view",
    )

    integer_loop_text = integer_loop_bound_probe.compile(BLOCK=8).mlir_text()
    expect(
        integer_loop_text.count("arith.index_cast") >= 2,
        "integer runtime loop bounds should be normalized to index with arith.index_cast",
    )
    expect(
        integer_loop_text.count("scf.for") == 2,
        "integer loop bound probe should still lower nested authored loops to scf.for",
    )

    scalar_pointer_offset_text = scalar_pointer_offset_probe.compile().mlir_text()
    expect(
        re.search(r"pto\.store %c1_i32, %\d+\[%c1\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.store %c2_i32, %\d+\[%c2\]", scalar_pointer_offset_text) is not None,
        "scalar.store(ptr + 2) should lower as element offset 2",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr, 1) should lower as element offset 1",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c2(?:_\d+)?\]", scalar_pointer_offset_text) is not None,
        "scalar.load(ptr + 2) should lower as element offset 2",
    )

    simt_pointer_offset_text = simt_pointer_offset_probe.compile().mlir_text()
    expect(
        "call @simt_pointer_offset_helper" in simt_pointer_offset_text,
        "@pto.simt pointer helper should lower to a helper func.call",
    )
    expect(
        re.search(r"pto\.store %c9_i32, %(?:arg0|\d+)\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "ptr+offset sugar inside @pto.simt helpers should lower as address offsets, not scalar add",
    )
    expect(
        re.search(r"pto\.load %\d+\[%c1(?:_\d+)?\]", simt_pointer_offset_text) is not None,
        "@pto.simt pointer helper probe should preserve ptr+offset load syntax on the caller side",
    )

    scalar_store_coercion_text = scalar_store_element_coercion_probe.compile().mlir_text()
    expect(
        scalar_store_coercion_text.count("arith.index_cast") >= 2,
        "scalar.store(...) should coerce index runtime values to the destination integer element type",
    )
    expect(
        "arith.trunci" in scalar_store_coercion_text,
        "scalar.store(...) should coerce wider integer runtime values down to the destination element type",
    )
    expect(
        scalar_store_coercion_text.count("pto.store") == 4,
        "scalar.store(...) coercion probe should still lower to four pto.store operations",
    )

    public_surface_text = public_surface_exports_probe.compile().mlir_text()
    expect("pto.mte_gm_ub" in public_surface_text, "mte_load(...) should lower to pto.mte_gm_ub")
    expect("pto.mte_ub_gm" in public_surface_text, "mte_store(...) should lower to pto.mte_ub_gm")
    expect(public_surface_text.count("pto.mem_bar") >= 1, "mem_bar(...) should still lower explicit memory barriers")
    expect("pto.barrier <PIPE_ALL>" in public_surface_text, "pipe_barrier(Pipe.ALL) should lower to pto.barrier")
    expect("pto.vexp" in public_surface_text, "vexp(...) should lower to pto.vexp")
    expect("pto.vcgmax" in public_surface_text, "vcgmax(...) should lower to pto.vcgmax")
    expect("pto.vcgadd" in public_surface_text, "vcgadd(...) should lower to pto.vcgadd")
    expect("pto.vadds" in public_surface_text, "vsubs(...) should lower via scalar negation plus pto.vadds")
    expect("pto.mte_l1_l0a" in public_surface_text, "mte_l1_l0a(...) should lower to pto.mte_l1_l0a")
    expect("pto.mte_l1_l0b" in public_surface_text, "mte_l1_l0b(...) should lower to pto.mte_l1_l0b")
    expect("pto.mte_l0c_ub" in public_surface_text, "mte_l0c_ub(...) should lower to pto.mte_l0c_ub")
    expect("pto.mad" in public_surface_text, "mad(...) should lower to pto.mad")

    try:
        block64[1, None]
    except NotImplementedError as exc:
        expect("compile / inspect / verify / emit" in str(exc), "runtime-launch diagnostic text mismatch")
    else:
        raise AssertionError("compiled handle unexpectedly accepted runtime launch syntax")

    print("ptodsl_jit_compile: PASS")


if __name__ == "__main__":
    main()
