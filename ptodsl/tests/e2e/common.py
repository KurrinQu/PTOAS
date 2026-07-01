# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Shared e2e test harness: kernel builders, launch, and reference functions."""

# NOTE: do NOT add "from __future__ import annotations" here.
# make_binary_kernel uses dynamic annotation expressions (pto.ptr(pto_dtype, "gm"))
# which must be evaluated at definition time, not stored as strings.

import time
from typing import Callable

import numpy as np

from ptodsl import pto

BINARY_OPS = {
    "add": (pto.tile.add, lambda x, y: x + y),
    "sub": (pto.tile.sub, lambda x, y: x - y),
    "mul": (pto.tile.mul, lambda x, y: x * y),
    "div": (pto.tile.div, lambda x, y: x / (y + 1e-8)),
}


def _npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def _torch_dtype(torch, dtype_str: str):
    return getattr(torch, dtype_str)


def make_binary_kernel(
    op_name: str,
    rows: int,
    cols: int,
    dtype_str: str = "float32",
    target: str = "a3",
    backend: str = "vpto",
    kernel_kind: str = "vector",
):
    """Return a ``@pto.jit`` KernelHandle for an elementwise binary op.

    The generated kernel uses the same 5-D tile-buffer pattern as the
    ``tadd_launch_a3.py`` / ``bop_launch_a3.py`` examples.
    """
    tile_op_fn = BINARY_OPS[op_name][0]
    pto_dtype = getattr(pto, dtype_str)
    fn_name = f"bin_{op_name}_{dtype_str}_{rows}x{cols}"

    def kernel_body(
        A_ptr: pto.ptr(pto_dtype, "gm"),
        B_ptr: pto.ptr(pto_dtype, "gm"),
        C_ptr: pto.ptr(pto_dtype, "gm"),
    ) -> None:
        c0 = pto.const(0)
        c1 = pto.const(1)
        c_rows = pto.const(rows)
        c_cols = pto.const(cols)
        c_elems = pto.const(rows * cols)

        shape = [c1, c1, c1, c_rows, c_cols]
        strides = [c_elems, c_elems, c_elems, c_cols, c1]
        off = [c0, c0, c0, c0, c0]

        a_view = pto.make_tensor_view(A_ptr, shape=shape, strides=strides)
        b_view = pto.make_tensor_view(B_ptr, shape=shape, strides=strides)
        c_view = pto.make_tensor_view(C_ptr, shape=shape, strides=strides)

        a_part = pto.partition_view(a_view, offsets=off, sizes=shape)
        b_part = pto.partition_view(b_view, offsets=off, sizes=shape)
        c_part = pto.partition_view(c_view, offsets=off, sizes=shape)

        a_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        b_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)
        c_tile = pto.alloc_tile(shape=[rows, cols], dtype=pto_dtype)

        pto.tile.load(a_part, a_tile)
        pto.tile.load(b_part, b_tile)
        tile_op_fn(a_tile, b_tile, c_tile)
        pto.tile.store(c_tile, c_part)

    kernel_body.__name__ = fn_name
    return pto.jit(
        name=fn_name,
        kernel_kind=kernel_kind,
        target=target,
        backend=backend,
    )(kernel_body)


def make_input(shape, dtype, torch, seed=42):
    """Return an NPU tensor filled with seeded random small integers.

    Small integers ensure exact fp results, avoiding rounding differences
    on different hardware paths.
    """
    rng = np.random.RandomState(seed)
    x = rng.randint(1, 10, size=shape).astype(np.float32)
    return torch.from_numpy(x).to(device="npu:0", dtype=dtype)


def launch_and_check(
    *,
    kernel_handle,
    ref_fn: Callable,
    shape: tuple[int, int],
    dtype_str: str,
    torch,
    rtol: float = 1e-6,
    atol: float = 1e-6,
    seed: int = 42,
):
    """Compile, launch, and numerical-check one kernel specialization."""
    torch_dt = _torch_dtype(torch, dtype_str)

    x = make_input(shape, torch_dt, torch, seed=seed)
    y = make_input(shape, torch_dt, torch, seed=seed + 1)
    z = torch.empty(shape, dtype=torch_dt, device="npu:0")
    ref = ref_fn(x.cpu().numpy(), y.cpu().numpy())
    stream = _npu_stream(torch)

    t0 = time.perf_counter()
    compiled = kernel_handle.compile()
    compile_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    compiled[1, stream](x.data_ptr(), y.data_ptr(), z.data_ptr())
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    actual = z.cpu().numpy()
    np.testing.assert_allclose(actual, ref, rtol=rtol, atol=atol)
    return compile_s, launch_s
