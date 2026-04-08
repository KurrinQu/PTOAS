"""`pto.tstore` 的 TileLang DSL 草案模板。

这个文件刻意写成目标 authoring surface 草案，不要求当前独立版
`tilelang_dsl` frontend 可以编译通过。对应的语义分析见
`docs/isa/tload-tstore-vec-nd-analysis.md`，缺失特性的清单和
论证见 `docs/plans/2026-04-08-tload-tstore-tilelang-template-design.md`。

这里假定存在、但当前前端尚未补齐的能力包括：
- rank-5 `TensorView.shape` / `TensorView.strides`
- rank-5 `partition_tensor_view` 的 authoring/lowering 支持
- `TensorView.as_ptr()`
- `pto.bytewidth(dtype)`
- `pto.copy_ubuf_to_gm_v2(...)`：暴露 `n_burst / len_burst /
  gm_stride / ub_stride` 的 burst-mode 重载，对应 C++
  `copy_ubuf_to_gm_align_v2`。当前稳定版 `pto.copy_ubuf_to_gm`
  的参数表里没有这几项。
- `pto.set_loop*_stride_ubtoout(src_stride, dst_stride)` 显式命名
  的 src/dst 参数，`pto.set_loop_size_ubtoout(loop1, loop2)` 显式
  的 loop1/loop2 参数。
- `pto.assert_rank / pto.assert_eq / pto.assert_le` 实例化期断言。
"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tstore",
    advanced=True,
)
def template_tstore(src: pto.Tile, dst: pto.TensorView):
    dtype = src.element_type
    elem_bytes = pto.bytewidth(dtype)

    g0 = dst.shape[0]
    g1 = dst.shape[1]
    g2 = dst.shape[2]
    g3 = dst.shape[3]
    g4 = dst.shape[4]

    s0 = dst.strides[0]
    s1 = dst.strides[1]
    s2 = dst.strides[2]
    s3 = dst.strides[3]
    s4 = dst.strides[4]

    valid_rows, valid_cols = src.valid_shape
    ub_rows, ub_cols = src.shape

    # Store 侧的断言比 load 严格：DMA 不支持对 GM 的 read-modify-write，
    # 因此 valid_shape 必须精确等于要落地的 ND 子块大小（见分析文档 §2.1）。
    pto.assert_rank(dst, 5)
    pto.assert_eq(s4, 1)
    pto.assert_eq(valid_rows, g0 * g1 * g2 * g3)
    pto.assert_eq(valid_cols, g4)
    pto.assert_le(valid_rows, ub_rows)
    pto.assert_le(valid_cols, ub_cols)

    n_burst = g3
    len_burst = valid_cols * elem_bytes
    ub_stride = ub_cols * elem_bytes
    gm_stride = s3 * elem_bytes

    # UB 源 tile 是行宽 `ub_cols` 的紧凑 row-major 布局，
    # 递推出三层阶梯 stride；与 tload 对称，只是方向相反。
    src_stride2 = g3 * ub_cols
    src_stride1 = g2 * src_stride2
    src_stride0 = g1 * src_stride1

    loop1 = g2
    loop2 = g1
    loop1_src_stride = src_stride2 * elem_bytes
    loop1_dst_stride = s2 * elem_bytes
    loop2_src_stride = src_stride1 * elem_bytes
    loop2_dst_stride = s1 * elem_bytes

    ub_ptr = src.as_ptr()         # 已存在特性：Tile.as_ptr() -> UBPtr
    gm_ptr = dst.as_ptr()         # 假设特性：TensorView.as_ptr() -> GMPtr

    # 原始 TileLib 的 TStoreVecND 无 loop 守卫且函数退出时不复位，
    # 这里主动补齐这两点，避免寄存器状态污染（见分析文档 §2.3 和附录）。
    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_ubtoout(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_ubtoout(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_ubtoout(loop1=loop1, loop2=loop2)

    for i in range(0, g0, 1):
        src_i = pto.addptr(ub_ptr, i * src_stride0 * elem_bytes)
        dst_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        pto.copy_ubuf_to_gm_v2(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
        )

    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_ubtoout(loop1=1, loop2=1)
    return
