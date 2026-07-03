# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared column arg-reduction templates."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _constraints(src_config, tmp_config, dst_config, src_dtype, tmp_dtype, dst_dtype, dst_valid_shape, **_):
    return (
        src_config.b_layout == "row_major"
        and tmp_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and tmp_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
        and src_dtype == tmp_dtype
        and dst_dtype == "i32"
        and dst_valid_shape[0] == 1
    )


def register_col_arg_template(*, op, name, cmp_mode, reduce_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[("f32", "f32", "i32")],
        constraints=[_constraints],
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("reduction", "column", "arg"),
    )
    def template(src: pto.Tile, tmp: pto.Tile, dst: pto.Tile):
        _ = tmp
        src_valid_rows, src_valid_cols = src.valid_shape
        lanes = pto.elements_per_vreg(src.dtype)
        full_mask = pto.make_mask(src.dtype, pto.PAT.ALL)

        for col in range(0, src_valid_cols, lanes):
            remained = src_valid_cols - col
            mask, _ = pto.make_mask(src.dtype, remained)
            index_old = pto.vdup(pto.i32(0), mask)
            index_new = pto.vdup(pto.i32(0), mask)
            best_vals = pto.vlds(src[0, col:])

            for row in range(1, src_valid_rows, 1):
                index_new = pto.vadds(index_new, pto.i32(1), mask)
                new_vals = pto.vlds(src[row, col:])
                select = pto.vcmp(new_vals, best_vals, full_mask, cmp_mode)
                index_old = pto.vsel(index_new, index_old, select)
                best_vals = reduce_op(best_vals, new_vals, mask)

            pto.vsts(index_old, dst[0, col:], mask)

    return template


__all__ = ["register_col_arg_template"]
