# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for the aligned ``pto.tsort32`` path."""

from ptodsl import pto
import ptodsl.tilelib as tilelib


BLOCK_SIZE = 32
FLOAT_DST_STRIDE_COEF = 2
HALF_DST_STRIDE_COEF = 4
REPEAT_MAX = 255


def _aligned(src_valid_shape, **_):
    return len(src_valid_shape) == 2 and src_valid_shape[1] % BLOCK_SIZE == 0


@tilelib.tile_template(
    op="pto.tsort32",
    target="a5",
    name="template_tsort32",
    dtypes=[("f16", "i32", "f16"), ("bf16", "i32", "bf16"), ("f32", "i32", "f32")],
    constraints=[_aligned],
    id=0,
    loop_depth=2,
    is_post_update=False,
    tags=("sort", "aligned"),
)
def template_tsort32(src: pto.Tile, idx: pto.Tile, dst: pto.Tile):
    dtype = dst.dtype
    valid_rows = dst.valid_shape[0]
    valid_cols = src.shape[1]

    dst_ptr = dst.as_ptr()
    src_ptr = src.as_ptr()
    idx_ptr = idx.as_ptr()

    elem_bytes = pto.bytewidth(dtype)
    dst_stride = ((dst.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    src_stride = ((src.shape[1] * elem_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // elem_bytes
    idx_stride = ((idx.shape[1] * 4 + BLOCK_SIZE - 1) // BLOCK_SIZE * BLOCK_SIZE) // 4
    if idx.shape[0] == 1:
        idx_stride = 0

    type_coef = HALF_DST_STRIDE_COEF
    if str(dtype) == "f32":
        type_coef = FLOAT_DST_STRIDE_COEF

    repeat_num_per_row = (valid_cols + BLOCK_SIZE - 1) // BLOCK_SIZE

    if repeat_num_per_row <= REPEAT_MAX:
        for row in range(0, valid_rows, 1):
            pto.vbitsort(
                pto.addptr(dst_ptr, row * dst_stride),
                pto.addptr(src_ptr, row * src_stride),
                pto.addptr(idx_ptr, row * idx_stride),
                repeat_num_per_row,
            )
    else:
        loop_num = (repeat_num_per_row + REPEAT_MAX - 1) // REPEAT_MAX
        tail_repeat_num = repeat_num_per_row % REPEAT_MAX
        for row in range(0, valid_rows, 1):
            for chunk in range(0, loop_num, 1):
                repeat_num = REPEAT_MAX
                if chunk == loop_num - 1:
                    repeat_num = tail_repeat_num
                pto.vbitsort(
                    pto.addptr(dst_ptr, row * dst_stride + chunk * REPEAT_MAX * BLOCK_SIZE * type_coef),
                    pto.addptr(src_ptr, row * src_stride + chunk * REPEAT_MAX * BLOCK_SIZE),
                    pto.addptr(idx_ptr, row * idx_stride + chunk * REPEAT_MAX * BLOCK_SIZE),
                    repeat_num,
                )
