"""`pto.tload` 的 TileLang DSL 草案模板。

这个文件刻意写成目标 authoring surface 草案，不要求当前独立版
`tilelang_dsl` frontend 可以编译通过。对应的语义分析见
`docs/isa/tload-tstore-vec-nd-analysis.md`，缺失特性的清单和
论证见 `docs/plans/2026-04-08-tload-tstore-tilelang-template-design.md`。

这里假定存在、但当前前端尚未补齐的能力包括：
- rank-5 `TensorView.shape` / `TensorView.strides`
- rank-5 `partition_tensor_view` 的 authoring/lowering 支持
- `TensorView.as_ptr()`（目前只有 `Tile.as_ptr()`）
- `pto.bytewidth(dtype)`（元素 → 字节换算）
- `pto.copy_gm_to_ubuf_v2(...)`：暴露 `n_burst / len_burst /
  gm_stride / ub_stride / enable_ub_pad` 的 burst-mode 重载，对应
  C++ `copy_gm_to_ubuf_align_v2`。当前稳定版 `pto.copy_gm_to_ubuf`
  的参数表里没有这几项，无法表达 ND tile 搬运。
- `pto.set_loop*_stride_outtoub(src_stride, dst_stride)` 显式命名的
  src/dst stride 参数，以及 `pto.set_loop_size_outtoub(loop1, loop2)`
  显式的 loop1/loop2 次数参数（避免当前 `(stride0, stride1)` /
  `(size0, size1)` 的位置歧义）。
- `pto.assert_rank / pto.assert_eq / pto.assert_le` 实例化期断言。
"""

import tilelang_dsl as pto


@pto.vkernel(
    target="a5",
    op="pto.tload",
    advanced=True,
)
def template_tload(src: pto.TensorView, dst: pto.Tile):
    dtype = dst.element_type
    elem_bytes = pto.bytewidth(dtype)

    # rank-5 partition view 元信息。
    g0 = src.shape[0]
    g1 = src.shape[1]
    g2 = src.shape[2]
    g3 = src.shape[3]
    g4 = src.shape[4]

    s0 = src.strides[0]
    s1 = src.strides[1]
    s2 = src.strides[2]
    s3 = src.strides[3]
    s4 = src.strides[4]

    valid_rows, valid_cols = dst.valid_shape
    ub_rows, ub_cols = dst.shape

    # Load 侧的断言比 store 宽松：允许 valid_shape < physical_shape，
    # 未覆盖区域由 `enable_ub_pad` 决定是否清零（见分析文档 §1.1 和 §2.3）。
    pto.assert_rank(src, 5)
    pto.assert_eq(s4, 1)
    pto.assert_le(valid_rows, g0 * g1 * g2 * g3)
    pto.assert_le(valid_cols, g4)
    pto.assert_le(g0 * g1 * g2 * g3, ub_rows)
    pto.assert_le(g4, ub_cols)

    n_burst = g3
    len_burst = g4 * elem_bytes
    gm_stride = s3 * elem_bytes
    ub_stride = ub_cols * elem_bytes

    # UB 目标 tile 是行宽为 `ub_cols` 的紧凑 row-major 布局，
    # 从最内层 `g3 × ub_cols` 块递推出三层阶梯 stride。
    dst_stride2 = g3 * ub_cols
    dst_stride1 = g2 * dst_stride2
    dst_stride0 = g1 * dst_stride1

    # loop1 ↔ g2（内层），loop2 ↔ g1（外层），软件 for ↔ g0。
    loop1 = g2
    loop2 = g1
    loop1_src_stride = s2 * elem_bytes
    loop1_dst_stride = dst_stride2 * elem_bytes
    loop2_src_stride = s1 * elem_bytes
    loop2_dst_stride = dst_stride1 * elem_bytes

    # TensorView / Tile → 类型化基指针。
    gm_ptr = src.as_ptr()         # 假设特性：TensorView.as_ptr() -> GMPtr
    ub_ptr = dst.as_ptr()         # 已存在特性：Tile.as_ptr() -> UBPtr

    # 配置硬件 loop 寄存器：显式命名 src/dst 和 loop1/loop2，
    # 避免当前 DSL `(stride0, stride1)` / `(size0, size1)` 的位置歧义。
    # 退化情形 loop1 == loop2 == 1 时跳过寄存器写入，避免污染。
    if loop1 != 1 or loop2 != 1:
        pto.set_loop2_stride_outtoub(
            src_stride=loop2_src_stride, dst_stride=loop2_dst_stride
        )
        pto.set_loop1_stride_outtoub(
            src_stride=loop1_src_stride, dst_stride=loop1_dst_stride
        )
        pto.set_loop_size_outtoub(loop1=loop1, loop2=loop2)

    # 最外层 ND 轴由软件循环承担；内部三层分别映射到
    # hardware loop2 / loop1 / 单次 copy 的 nBurst x lenBurst。
    # 每次 `copy_gm_to_ubuf_v2` 实际被硬件重放 `loop1 * loop2` 次。
    for i in range(0, g0, 1):
        src_i = pto.addptr(gm_ptr, i * s0 * elem_bytes)
        dst_i = pto.addptr(ub_ptr, i * dst_stride0 * elem_bytes)
        pto.copy_gm_to_ubuf_v2(
            dst=dst_i,
            src=src_i,
            n_burst=n_burst,
            len_burst=len_burst,
            gm_stride=gm_stride,
            ub_stride=ub_stride,
            enable_ub_pad=False,
        )

    # 显式恢复 normal mode，避免寄存器状态污染后续无关的 copy。
    if loop1 != 1 or loop2 != 1:
        pto.set_loop_size_outtoub(loop1=1, loop2=1)
    return
