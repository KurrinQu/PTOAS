"""TileLang DSL template for pto.tadd — used by ExpandTileOp tests."""

import sys
from pathlib import Path
import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tadd",
    dtypes=[(pto.f32, pto.f32, pto.f32)]
)
def template_tadd(dst: pto.Tile, src0: pto.Tile, src1: pto.Tile):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            lhs = pto.vlds(src0[row, col:])
            rhs = pto.vlds(src1[row, col:])
            summed = pto.vadd(lhs, rhs, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return
