#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ptodsl"))

from ptodsl import pto
from ptodsl._host_tensors import inspect_host_tensor_metadata


def expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def expect_raises(callback, exc_type, *message_fragments: str) -> None:
    try:
        callback()
    except exc_type as exc:
        text = str(exc)
        for fragment in message_fragments:
            expect(fragment in text, f"expected diagnostic fragment {fragment!r} in {text!r}")
    else:
        raise AssertionError(f"expected {exc_type.__name__} to be raised")


@pto.jit(target="a5")
def native_python_if_runtime_const_probe():
    if pto.const(1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def native_python_range_runtime_metadata_probe(A: pto.tensor_spec(rank=2, dtype=pto.f32)):
    for _ in range(A.shape[0]):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def float_loop_bound_probe():
    with pto.for_(0, pto.const(1.5, dtype=pto.f32), step=1):
        pto.pipe_barrier(pto.Pipe.ALL)


@pto.jit(target="a5")
def carry_update_mismatch_probe(*, BLOCK: pto.constexpr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(other=acc)


@pto.jit(target="a5")
def carry_final_mismatch_probe(*, BLOCK: pto.constexpr = 8):
    acc = pto.alloc_tile(shape=[1, BLOCK], dtype=pto.f32)
    loop = pto.for_(0, 1, step=1).carry(acc=acc)
    with loop:
        loop.update(acc=acc)
    loop.final("missing")


@pto.jit(target="a5")
def misaligned_row_major_tile_probe():
    pto.alloc_tile(shape=[128, 1], dtype=pto.f32, valid_shape=[128, 1])


class MissingDTypeTensor:
    shape = (4, 8)
    strides = (8, 1)

    def data_ptr(self):
        return 1024


class BadDataHandleTensor:
    shape = (4, 8)
    strides = (8, 1)
    dtype = "float32"

    def data_ptr(self):
        return "not-an-int"


def define_missing_constexpr_default_probe():
    @pto.jit(target="a5")
    def bad_probe(*, BLOCK: pto.constexpr):
        pto.pipe_barrier(pto.Pipe.ALL)

    return bad_probe


def main() -> None:
    expect_raises(
        native_python_if_runtime_const_probe.compile,
        TypeError,
        "native Python if/while condition",
        "pto.if_(...)",
        "pto.constexpr",
    )
    expect_raises(
        native_python_range_runtime_metadata_probe.compile,
        TypeError,
        "native Python range()/loop bound",
        "pto.for_(...)",
        "runtime value",
    )
    expect_raises(
        float_loop_bound_probe.compile,
        TypeError,
        "pto.for_(...) loop bound",
        "expects an index or integer runtime scalar",
        "f32",
    )
    expect_raises(
        carry_update_mismatch_probe.compile,
        RuntimeError,
        "loop.update(...) must match carry names exactly",
        "missing: acc",
        "unexpected: other",
    )
    expect_raises(
        carry_final_mismatch_probe.compile,
        RuntimeError,
        "loop.final(...) requested unknown carry state 'missing'",
        "expected one of: acc",
    )
    expect_raises(
        misaligned_row_major_tile_probe.compile,
        TypeError,
        "alloc_tile(shape=...) physical row layout is invalid",
        "shape=[128, 1]",
        "row byte size of 4",
        "32-byte aligned",
        "prefer blayout='ColMajor'",
    )
    expect_raises(
        define_missing_constexpr_default_probe,
        TypeError,
        "@pto.jit constexpr parameter 'BLOCK' must declare a default value",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(MissingDTypeTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "missing .dtype",
    )
    expect_raises(
        lambda: inspect_host_tensor_metadata(BadDataHandleTensor()),
        TypeError,
        "host tensor metadata is incomplete or unsupported",
        "data_ptr must return an integer-like data handle",
    )
    print("ptodsl_jit_diagnostics: PASS")


if __name__ == "__main__":
    main()
