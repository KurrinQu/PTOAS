# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementations for row-wise reductions."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES, element_store_dist


def _single_output_col(dst_valid_shape=(), **_):
    return len(dst_valid_shape) == 2 and dst_valid_shape[1] == 1


def register_row_extreme(*, op, name, reduce_op, combine_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
        constraints=[
            tilelib.check_memory_space("ub"),
            tilelib.check_layout("row_major"),
            tilelib.check_s_layout("none_box"),
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)

        for row in range(0, valid_rows, 1):
            first_mask, remained = pto.make_mask(dtype, valid_cols)
            first = pto.vlds(src[row, 0:])
            acc = reduce_op(first, first_mask)
            for col in range(lanes, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                reduced = reduce_op(value, mask)
                acc = combine_op(acc, reduced, one_mask)
            pto.vsts(acc, dst[row, 0:], one_mask, dist=element_store_dist(dtype))

    return template


def register_rowsum():
    @tilelib.tile_template(
        op="pto.trowsum",
        target="a5",
        name="template_trowsum",
        dtypes=[("f16", "f16", "f16"), ("f32", "f32", "f32"), ("i32", "i32", "i32")],
        constraints=[
            tilelib.check_memory_space("ub"),
            tilelib.check_layout("row_major"),
            tilelib.check_s_layout("none_box"),
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row", "sum"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)

        for row in range(0, valid_rows, 1):
            first_mask, remained = pto.make_mask(dtype, valid_cols)
            first = pto.vlds(src[row, 0:])
            acc = pto.vcadd(first, first_mask)
            for col in range(lanes, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                reduced = pto.vcadd(value, mask)
                acc = pto.vadd(acc, reduced, one_mask)
            pto.vsts(acc, dst[row, 0:], one_mask, dist=element_store_dist(dtype))

    return template


def register_rowprod():
    @tilelib.tile_template(
        op="pto.trowprod",
        target="a5",
        name="template_trowprod",
        dtypes=[("f16", "f16", "f16"), ("f32", "f32", "f32"), ("i16", "i16", "i16"), ("i32", "i32", "i32")],
        constraints=[
            tilelib.check_memory_space("ub"),
            tilelib.check_layout("row_major"),
            tilelib.check_s_layout("none_box"),
            _single_output_col,
        ],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "row", "prod"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        dtype = dst.dtype
        valid_rows, valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(dtype)
        one_mask, _ = pto.make_mask(dtype, 1)
        full_mask, _ = pto.make_mask(dtype, lanes)

        for row in range(0, valid_rows, 1):
            first_mask, remained = pto.make_mask(dtype, valid_cols)
            acc = pto.vlds(src[row, 0:])
            fill = pto.vadds(pto.vmuls(acc, 0, first_mask), 1, first_mask)
            for col in range(lanes, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                value = pto.vlds(src[row, col:])
                prod = pto.vmul(acc, value, mask)
                acc = pto.vsel(prod, acc, mask)

            low, _ = pto.vintlv(acc, fill)
            acc = pto.vmul(low, fill, full_mask)
            pto.vsts(acc, dst[row, 0:], one_mask, dist=element_store_dist(dtype))

    return template


__all__ = ["register_row_extreme", "register_rowsum", "register_rowprod"]
