# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for a basic row-wise ``pto.tcvt`` path."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


def _rowwise(src_shape, src_valid_shape, dst_shape, dst_valid_shape, src_config, dst_config, **_):
    return (
        tuple(src_shape) == tuple(dst_shape)
        and tuple(src_valid_shape) == tuple(dst_valid_shape)
        and src_config.b_layout == "row_major"
        and dst_config.b_layout == "row_major"
        and src_config.s_layout == "none_box"
        and dst_config.s_layout == "none_box"
    )


def _round_mode():
    round_mode = pto.get_op_attr("round_mode", "RINT")
    if round_mode == "ROUND":
        return pto.VcvtRoundMode.A
    if round_mode == "FLOOR":
        return pto.VcvtRoundMode.F
    if round_mode == "CEIL":
        return pto.VcvtRoundMode.C
    if round_mode == "TRUNC":
        return pto.VcvtRoundMode.Z
    if round_mode == "ODD":
        return pto.VcvtRoundMode.O
    return pto.VcvtRoundMode.R


@tilelib.tile_template(
    op="pto.tcvt",
    target="a5",
    name="template_tcvt_f32_to_i32",
    dtypes=[("f32", "i32")],
    constraints=[_rowwise],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("convert", "rowwise"),
)
def template_tcvt_f32_to_i32(src: pto.Tile, dst: pto.Tile):
    valid_rows, valid_cols = dst.valid_shape
    dtype = dst.dtype
    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.elements_per_vreg(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vec = pto.vlds(src[row, col:])
            converted = pto.vcvt(vec, dtype, mask, rnd=_round_mode(), sat=pto.VcvtSatMode.SAT)
            pto.vsts(converted, dst[row, col:], mask)
