"""TileLang DSL template for pto.tadds"""

import tilelang_dsl as pto

@pto.vkernel(
    target="a5",
    op="pto.tadds",
)
def template_tadds(dst: pto.Tile, src: pto.Tile, scalar: pto.AnyType):
    dtype = dst.element_type
    valid_rows, valid_cols = dst.valid_shape

    for row in range(0, valid_rows, 1):
        remained = valid_cols
        for col in range(0, valid_cols, pto.get_lanes(dtype)):
            mask, remained = pto.make_mask(dtype, remained)
            vec = pto.vlds(src[row, col:])
            summed = pto.vadds(vec, scalar, mask)
            pto.vsts(summed, dst[row, col:], mask)
    return
