# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""
Fast Hadamard transform launch demo.

This ports the old DSL in-place fp16 Hadamard idea to the current PTODSL launch
surface.  The old example used gather mask patterns for arbitrary butterfly
halves.  This launchable version demonstrates the aligned vector route for
2-point Hadamard rows: DINTLV/INTLV B16 performs the aligned butterfly without
using the old gather mask-pattern helper.

Run under the CPU simulator:

    scripts/sim_dsl.sh ptodsl/examples/fast_hadamard_launch.py
"""

import argparse
import math
import time
from pathlib import Path
import sys

import numpy as np

if __package__ in {None, ""}:
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "ptodsl" / "__init__.py").exists():
            sys.path.insert(0, str(candidate))
            break
    else:
        raise RuntimeError(
            "Unable to locate the PTODSL Python package root from fast_hadamard_launch.py"
        )

from ptodsl import pto


_DEVICE = "npu:0"
MAX_BATCH = 8
PHYSICAL_N = 256


@pto.jit(
    name="fast_hadamard_f16",
    kernel_kind="vector",
    target="a5",
    mode="explicit",
    insert_sync=False,
)
def fast_hadamard_f16(
    x_ptr: pto.ptr(pto.f16, "gm"),
    batch_i32: pto.i32,
    n_i32: pto.i32,
    log2_n_i32: pto.i32,
):
    total_elems = batch_i32 * n_i32
    x_view = pto.make_tensor_view(
        x_ptr,
        shape=[1, 1, 1, batch_i32, n_i32],
        strides=[total_elems, total_elems, total_elems, n_i32, 1],
    )
    row_tile = pto.alloc_tile(
        shape=[MAX_BATCH, PHYSICAL_N],
        dtype=pto.f16,
        addr=0,
        valid_shape=[batch_i32, n_i32],
        blayout="RowMajor",
    )
    x_part = pto.partition_view(
        x_view,
        offsets=[0, 0, 0, 0, 0],
        sizes=[1, 1, 1, batch_i32, n_i32],
    )

    pto.tile.load(x_part, row_tile)
    pto.set_flag("MTE2", "V", event_id=0)
    pto.wait_flag("MTE2", "V", event_id=0)

    row_ptr = row_tile.as_ptr()
    active_pairs = n_i32 // pto.const(2, dtype=pto.i32)

    with pto.simd():
        with pto.if_(pto.const(0, dtype=pto.i32) < log2_n_i32) as br0:
            with br0.then_:
                active_mask, _ = pto.make_mask(pto.f16, active_pairs)
                with pto.for_(0, batch_i32, step=1) as row:
                    row_offset = row * PHYSICAL_N
                    even, odd = pto.vldsx2(
                        row_ptr,
                        row_offset,
                        pto.DeinterleaveDist.DINTLV_B16,
                    )
                    plus = pto.vadd(even, odd, active_mask)
                    minus = pto.vsub(even, odd, active_mask)
                    pto.vstsx2(
                        plus,
                        minus,
                        row_ptr,
                        row_offset,
                        pto.InterleaveDist.INTLV_B16,
                        active_mask,
                    )
                pto.pipe_barrier(pto.Pipe.ALL)

    pto.set_flag("V", "MTE3", event_id=0)
    pto.wait_flag("V", "MTE3", event_id=0)
    pto.tile.store(row_tile, x_part)

    pto.pipe_barrier(pto.Pipe.ALL)


CASES = [
    {"name": "batch2_n2", "batch": 2, "n": 2, "eps": 1e-3},
    {"name": "batch3_n3", "batch": 3, "n": 3, "eps": 1e-3},
    {"name": "batch4_n5", "batch": 4, "n": 5, "eps": 1e-3},
]


def emit_mlir():
    return fast_hadamard_f16.mlir_module()


def reference_hadamard(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.float32).copy()
    for base in range(0, y.shape[1] - 1, 2):
        lhs = y[:, base].copy()
        rhs = y[:, base + 1].copy()
        y[:, base] = lhs + rhs
        y[:, base + 1] = lhs - rhs
    if y.shape[1] % 2:
        y[:, -1] = 0.0
    return y.astype(np.float16)


def init_runtime():
    import torch
    import torch_npu  # noqa: F401

    torch.npu.config.allow_internal_format = False
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.npu.set_device(_DEVICE)
    return torch


def npu_stream(torch):
    return torch.npu.current_stream()._as_parameter_  # noqa: SLF001


def make_case_inputs(case: dict[str, object]) -> np.ndarray:
    batch = int(case["batch"])
    n = int(case["n"])
    rng = np.random.RandomState(hash(case["name"]) & 0xFFFFFFFF)
    return rng.uniform(-1.0, 1.0, size=(batch, n)).astype(np.float16)


def run_case(case: dict[str, object], compiled, torch) -> None:
    x = make_case_inputs(case)
    ref = reference_hadamard(x)
    x_dev = torch.from_numpy(x).to(_DEVICE)
    stream = npu_stream(torch)

    t0 = time.perf_counter()
    compiled[1, stream](
        x_dev.data_ptr(),
        case["batch"],
        case["n"],
        int(math.log2(int(case["n"]))),
    )
    torch.npu.synchronize()
    launch_s = time.perf_counter() - t0

    np.testing.assert_allclose(
        x_dev.cpu().numpy().astype(np.float32),
        ref.astype(np.float32),
        rtol=case["eps"],
        atol=case["eps"],
    )
    print(f"PASS {case['name']}  launch={launch_s:.3f}s")


def test_fast_hadamard() -> None:
    torch = init_runtime()

    t0 = time.perf_counter()
    compiled = fast_hadamard_f16.compile()
    compile_s = time.perf_counter() - t0
    print(f"compiled fast_hadamard_f16 in {compile_s:.3f}s")

    for case in CASES:
        run_case(case, compiled, torch)
    print("All cases passed.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print MLIR module and exit",
    )
    args = parser.parse_args(argv)

    if args.emit_mlir:
        print(emit_mlir())
        return 0

    test_fast_hadamard()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
