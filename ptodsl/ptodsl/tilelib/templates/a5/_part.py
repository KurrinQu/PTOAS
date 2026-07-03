# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Shared PTODSL implementations for partition-combine TileOps."""

from ptodsl import pto
import ptodsl.tilelib as tilelib

from ._common import NUMERIC_DTYPES, ub_row_major_constraints


def register_part_binary(*, op, name, vector_op):
    @tilelib.tile_template(
        op=op,
        target="a5",
        name=name,
        dtypes=[(dtype, dtype, dtype) for dtype in NUMERIC_DTYPES],
        constraints=ub_row_major_constraints("src0", "src1", "dst"),
        id=0,
        loop_depth=2,
        is_post_update=False,
        tags=("partition", "binary"),
    )
    def template(src0: pto.Tile, src1: pto.Tile, dst: pto.Tile):
        dtype = dst.dtype
        valid_rows, valid_cols = dst.valid_shape
        lanes = pto.elements_per_vreg(dtype)

        for row in range(0, valid_rows, 1):
            remained = valid_cols
            for col in range(0, valid_cols, lanes):
                mask, remained = pto.make_mask(dtype, remained)
                lhs = pto.vlds(src0[row, col:])
                rhs = pto.vlds(src1[row, col:])
                result = vector_op(lhs, rhs, mask)
                pto.vsts(result, dst[row, col:], mask)

    return template


__all__ = ["register_part_binary"]
