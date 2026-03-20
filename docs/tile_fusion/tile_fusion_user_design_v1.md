# PTOAS Tile Fusion 用户文档（A5 v1）

- 状态：Current
- 适用范围：`ptoas --pto-arch=a5` 的当前源码实现
- 目标读者：上层框架的 PTO IR 生成端维护者、手写或程序化生成 PTO IR 的内核作者、评估 tile fusion 收益与边界的性能工程人员

本文描述当前版本 PTOAS tile fusion 的用户可见行为和 PTO IR 组织约束，重点回答两个问题：

1. 哪些 PTO IR 形态更利于当前版本的 tile fusion
2. 哪些结构会削弱、切断或显著降低当前版本的融合机会

本文是面向外部使用者的特性文档，不展开编译器内部 pass 顺序、helper 生成细节或低层 loop fusion 实现。如果需要维护者视角的实现说明，请阅读
[`docs/tile_fusion/oplib_lowering_tile_fusion_design_v1.md`](./oplib_lowering_tile_fusion_design_v1.md)。

## 1. 为什么需要这份文档

当前版本的 tile fusion 对 PTO IR 的组织方式敏感。即使两段 PTO IR 在数学意义上等价，只要数据流组织方式不同，进入当前融合路径的机会和后续收益也可能明显不同。

对上层框架和 PTO IR 生成端来说，缺少一份稳定的用户侧说明，会直接带来这些问题：

1. 连续的 elementwise 热链被切碎后，编译器可利用的融合空间下降。
2. 中间 tile 过早暴露给链外后，链内中间往返难以继续压缩。
3. 同一类 mixed chain 在不同生成器中呈现出不一致的 IR 形态，导致行为预期不稳定。
4. 使用者不得不依赖编译器内部实现细节，才能判断某类 PTO IR 是否适合当前 tile fusion 路径。

本文的目标不是承诺“所有语义等价 IR 都会融合”，而是给出当前版本中更稳定、更推荐的 PTO IR 组织方式。

## 2. 当前版本如何启用

当前 tile fusion 是 A5 路径上的显式开启能力，不是默认启用特性。

### 2.1 入口条件

当前用户可见入口条件如下：

1. 使用 `--pto-arch=a5`
2. 显式传入 `--enable-op-fusion`
3. 使用默认 OpLib 模板目录，或显式传入 `--op-lib-dir=<path>` 覆盖默认值

典型命令形态如下：

```bash
build/tools/ptoas/ptoas input.pto \
  --pto-arch=a5 \
  --enable-op-fusion \
  -o output.cpp
```

说明：

1. `--enable-op-fusion` 在 `--pto-arch!=a5` 时不会生效。
2. 当前 OpLib lowering 路径会先尝试安装目录默认位置，再回退到仓内 `oplib/level3`。
3. 只有自定义模板目录、负向测试或局部覆写时才需要显式传 `--op-lib-dir`。
4. 本文所有约束和例子都以当前 A5 路径为前提。

## 3. 当前用户可见范围

### 3.1 当前 in-scope op 范围

当前版本对外承认的 tile fusion 范围限定为 12 个 elementwise op：

| 类别        | Op                                                        |
| ----------- | --------------------------------------------------------- |
| tile-tile   | `tmul` / `tdiv` / `tadd` / `tsub` / `tmax` / `tmin`       |
| tile-scalar | `tmuls` / `tdivs` / `tadds` / `tsubs` / `tmaxs` / `tmins` |

本文不把其他 family 写成当前 tile fusion 的用户承诺范围，包括但不限于：

1. unary
2. ternary
3. reduction
4. broadcast
5. compare-select
6. bitwise

### 3.2 当前关注的数据流范围

当前版本面向上层的核心约束是同一 block 内的线性连续 elementwise chain，而不是通用图级融合模型。

当前推荐的数据流形态是：

```text
tload -> elementwise chain -> tstore
```

其中，链内的 tile producer-consumer 关系应尽量保持直接、连续、稳定。

### 3.3 Mixed chain 属于当前目标范围

当前版本的目标输入不只包括 pure tile-tile chain，也包括 tile-tile 和 tile-scalar 混合出现的 mixed chain。

对 mixed chain，推荐遵循以下做法：

1. scalar 按普通外部输入参与链，不需要为了“看起来统一”而先包装成额外中间 tile。
2. tile producer-consumer 关系保持连续，scalar 作为附加输入进入相应阶段即可。
3. mixed chain 的价值在于保留完整链条，而不是强行把所有操作改写成同一种 operand 形态。

## 4. 面向生成端的组织原则

### 4.1 优先生成同一 block 内的线性连续链

当前版本最推荐的 PTO IR 形态，是同一 block 内按程序顺序排列的 elementwise 直线链。

建议：

1. 让当前阶段的 tile 输入直接消费前一阶段的 tile 输出。
2. 保持 producer 和 consumer 在 IR 中相邻或近邻出现。
3. 避免在链中插入无关 op、同步点或非当前 in-scope op。

不建议把当前版本理解为“任意 DAG 里只要存在局部依赖就能自动找出并融合”。当前路径主要针对连续线性链。

### 4.2 把 mixed chain 当作一等输入形态

如果计算本身天然是 mixed chain，建议直接用 mixed chain 表达，不要为了规避 tile-scalar op 而额外引入中间 tile。

推荐模式：

1. `tmuls -> tmaxs -> tmins`
2. `tmul -> tadd`
3. `tadds -> tdivs`

这些阶段可以出现在同一条链上。关键不是所有 op 长得一样，而是链条中的 tile 数据流保持连续。

### 4.3 中间结果应优先保持为链内局部值

当前版本更适合处理“链内局部中间值”。如果中间 tile 在链中途被写回外部视图、提前物化为链外可观察结果，或者被无关路径消费，通常会削弱当前版本的融合机会，也会降低后续压缩链内往返的空间。

因此更推荐：

1. 让中间 tile 只服务于链内后继阶段。
2. 把真正需要对外可见的结果留在链尾统一导出。
3. 将链中临时结果视为内部工作值，而不是中间发布点。

需要注意的是，“中间值链外可见”在当前实现里不总是等价于“绝不发生任何融合”，但它通常会引入额外边界、额外接口或收益折损，因此不属于推荐的用户侧组织方式。

### 4.4 保持链内结构稳定

当前版本建议在链内尽量保持以下属性稳定：

1. `shape`
2. `dtype`
3. `layout`
4. 访问形态

这条建议的重点是：不要依赖编译器内部符号名或内部改写细节去猜测什么会被接受，而应在 PTO IR 层面尽量保持链内结构清晰、连续、稳定。

### 4.5 维持清晰的 `tload -> chain -> tstore` 边界

对上层框架生成端来说，最容易稳定复用的组织方式是：

1. 在链前完成 `tload`
2. 在链内保留连续的 elementwise producer-consumer 关系
3. 在链尾统一 `tstore`

这类边界清晰的结构，比“边算边频繁对外写回”更符合当前版本的 tile fusion 入口条件。

## 5. 会削弱或打断机会的常见结构

### 5.1 链中插入无关操作

当前分组模型是严格连续的。链中插入无关 op，会把原本的直线型数据流切开。

典型包括：

1. 普通无关计算
2. 额外调用
3. 显式同步或其他边界类 op

### 5.2 中间结果过早对外可见

如果链中间结果被提前 `tstore` 到外部视图，或者在链中途作为链外可观察值发布出去，当前版本的融合空间会明显变差。

用户侧应优先避免：

1. 链中段 `tstore`
2. 把中间 tile 设计成阶段性对外接口
3. 先导出中间值，再在当前链下半段继续消费它

### 5.3 block、region 和控制流边界

当前版本不以跨 block、跨 region 或复杂控制流边界的融合为主要目标范围。

因此以下形态都不属于当前版本最稳定的输入：

1. `scf.if` 两个分支各自产生 tile，随后在分支外继续 elementwise chain
2. chain 被 block 边界切开
3. chain 被 region 边界切开
4. chain 跨越复杂控制流结构

### 5.4 数学等价不等于 IR 组织等价

对用户来说，最重要的认识是：

1. 数学表达式等价，不代表 PTO IR 组织方式等价。
2. PTO IR 组织方式等价，也不代表当前版本的 profitability 完全相同。

当前约束主要来自两个方面：

1. legality：当前路径是否承认这种输入形态
2. profitability：即便可处理，这种形态是否仍保留足够的链内收益

因此，推荐把“连续链”“中间值局部化”“清晰边界”理解为用户侧稳定实践，而不是理解为任意实现细节。

## 6. 正例

### 6.1 Mixed chain 正例

```mlir
func.func @softmax_like(%in: !pto.ptr<f32>, %out: !pto.ptr<f32>) {
  // 省略 make_tensor_view / partition_view / alloc_tile / constant 定义

  pto.tload ins(%in_pt : !pto.partition_tensor_view<...>)
            outs(%x : !pto.tile_buf<...>)

  pto.tmuls ins(%x, %scale : !pto.tile_buf<...>, f32)
            outs(%x : !pto.tile_buf<...>)
  pto.tmaxs ins(%x, %neg4 : !pto.tile_buf<...>, f32)
            outs(%x : !pto.tile_buf<...>)
  pto.tmins ins(%x, %pos4 : !pto.tile_buf<...>, f32)
            outs(%x : !pto.tile_buf<...>)

  pto.tmul  ins(%x, %x : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%tmp0 : !pto.tile_buf<...>)
  pto.tadd  ins(%x, %tmp0 : !pto.tile_buf<...>, !pto.tile_buf<...>)
            outs(%y : !pto.tile_buf<...>)

  pto.tadds ins(%y, %bias : !pto.tile_buf<...>, f32)
            outs(%y : !pto.tile_buf<...>)
  pto.tdivs ins(%y, %divisor : !pto.tile_buf<...>, f32)
            outs(%y : !pto.tile_buf<...>)

  pto.tstore ins(%y : !pto.tile_buf<...>)
             outs(%out_pt : !pto.partition_tensor_view<...>)
  return
}
```

这个例子体现了当前版本最推荐的 mixed chain 组织方式：

1. scalar 作为普通外部输入直接参与，不额外包装成中间 tile。
2. `%x` 和 `%y` 在链内被后续阶段直接消费。
3. 中间 tile `%tmp0` 只服务于链内后继阶段。
4. `tload` 和 `tstore` 形成清晰边界。

### 6.2 Pure tile-tile chain 正例

```mlir
pto.tmul ins(%a, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp0 : !pto.tile_buf<...>)
pto.tdiv ins(%tmp0, %b : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp1 : !pto.tile_buf<...>)
pto.tadd ins(%tmp1, %tmp0 : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp2 : !pto.tile_buf<...>)
pto.tsub ins(%tmp2, %a : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%out : !pto.tile_buf<...>)
```

这个例子说明当前版本关注的是连续的 tile producer-consumer 关系，而不是“临时 tile 数量必须最少”。只要链条是清晰的线性连续关系，存在若干链内临时值本身不是问题。

## 7. 反例

### 7.1 反例一：链中插入无关操作

```mlir
pto.tmul ins(%x, %y : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp0 : !pto.tile_buf<...>)

func.call @do_side_work() : () -> ()

pto.tadd ins(%tmp0, %z : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp1 : !pto.tile_buf<...>)
```

该结构显式引入了链中边界，不再表现为连续直线型数据流。

### 7.2 反例二：中间结果链外可见

```mlir
pto.tmul ins(%x, %y : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp0 : !pto.tile_buf<...>)

pto.tstore ins(%tmp0 : !pto.tile_buf<...>)
           outs(%mid_pt : !pto.partition_tensor_view<...>)

pto.tadd ins(%tmp0, %z : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%tmp1 : !pto.tile_buf<...>)
```

该结构把 `%tmp0` 从链内局部结果升级为链外可观察结果，会形成明确边界并削弱当前版本的融合收益。

### 7.3 反例三：控制流切分

```mlir
%tmp_after_if = scf.if %cond -> (!pto.tile_buf<...>) {
  pto.tmul ins(%x, %y : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%tmp0 : !pto.tile_buf<...>)
  scf.yield %tmp0 : !pto.tile_buf<...>
} else {
  pto.tmul ins(%x, %z : !pto.tile_buf<...>, !pto.tile_buf<...>)
           outs(%tmp0_alt : !pto.tile_buf<...>)
  scf.yield %tmp0_alt : !pto.tile_buf<...>
}

pto.tadd ins(%tmp_after_if, %bias_tile : !pto.tile_buf<...>, !pto.tile_buf<...>)
         outs(%out : !pto.tile_buf<...>)
```

该结构已经不再是单条 block 内的线性连续链，而是被控制流边界切开。

## 8. 生成端快速自检清单

在生成 PTO IR 时，可以用下面的 checklist 做快速自检：

- [ ] 当前编译入口是否是 `--pto-arch=a5`
- [ ] 是否显式开启了 `--enable-op-fusion`
- [ ] 若使用自定义模板目录，是否显式提供了 `--op-lib-dir=<path>`
- [ ] 链中 op 是否全部落在当前 12 个 in-scope op 范围内
- [ ] 是否优先组织为同一 block 内的线性连续 chain
- [ ] mixed chain 中的 scalar 是否直接作为普通外部输入参与
- [ ] 中间 tile 是否优先保持为链内局部结果
- [ ] 是否避免在链中途 `tstore` 或把中间结果设计成链外可观察值
- [ ] 是否避免在链中插入无关 op、同步点或其他边界类操作
- [ ] 是否尽量保持链内 `shape`、`dtype`、`layout` 和访问形态稳定
- [ ] 是否尽量形成清晰的 `tload -> elementwise chain -> tstore` 边界

## 9. 使用边界说明

本文明确不承诺以下内容：

1. 所有语义等价 PTO IR 都必然发生 tile fusion。
2. 当前 12 个 op 之外的 broader family 也享有同等用户合同。
3. 跨 block、跨 region、跨 sync 或复杂控制流边界下存在通用融合模型。
4. 当前实现边界会作为长期 ABI 或永久规范稳定不变。

同时，本文也不要求使用者理解以下内部细节：

1. 编译器内部 pass 顺序
2. 内部 helper 命名
3. 内部 clone 策略
4. low-level fusion pattern 细节

用户侧只需要关心：当前版本更偏好什么样的 PTO IR 组织方式，以及什么样的结构会削弱当前版本的融合空间。

## Appendix A. 当前有效范围

### A.1 当前入口条件

1. 当前 tile fusion 讨论以 `--pto-arch=a5` 为前提。
2. `--enable-op-fusion` 是当前版本打开 fusion pass 的显式入口。
3. 当前路径仍依赖 OpLib 输入，但默认会自动查找安装目录或仓内 `oplib/level3`。

### A.2 当前 in-scope op 范围

| 类别        | Op                                                        |
| ----------- | --------------------------------------------------------- |
| tile-tile   | `tmul` / `tdiv` / `tadd` / `tsub` / `tmax` / `tmin`       |
| tile-scalar | `tmuls` / `tdivs` / `tadds` / `tsubs` / `tmaxs` / `tmins` |

### A.3 当前关注的数据流范围

当前版本主要关注：

1. 同一 block 内的线性连续 chain
2. 当前阶段 tile 输入直接消费前一阶段 tile 输出的 producer-consumer 关系
3. 清晰的 `tload -> elementwise chain -> tstore` 边界
4. pure tile-tile chain 与 mixed chain 两类输入

### A.4 当前不覆盖的内容

以下内容不属于当前版本的主要目标范围：

1. 跨 block 的链
2. 跨 region 的链
3. 跨 sync 的链
4. 复杂控制流边界下的一般融合模型
5. 超出当前 12 个 op 范围的 broader family 混合链

### A.5 相关文档

如果需要编译器内部实现、当前流水线结构或 OpLib lowering 维护细节，请阅读：

[`docs/tile_fusion/oplib_lowering_tile_fusion_design_v1.md`](./oplib_lowering_tile_fusion_design_v1.md)
