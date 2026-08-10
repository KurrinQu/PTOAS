# RMSNorm Persistent-SIMT 连续启动异常定位复盘

## 1. 文档目的

本文复盘一次只在同一进程连续启动多个 device program 时出现的
RMSNorm persistent-SIMT 异常。目标不是记录某一次命令的输出，而是总结一套
可复用的定位方法：如何建立稳定复现，如何逐轮裁剪变量，如何区分数据通路和
指令取指通路，最后如何用最小二进制 A/B 把硬件现象追溯到编译器编码问题。

本文中的结论分为两层：

- 编译器产物层的根因已经通过源码分析和真机 A/B 确认；
- runtime、firmware 和 IFU 内部究竟在哪个时刻建立或失效 cache/fetch 状态，
  仍需要 BIU/IFU 深层日志进一步确认。复盘中会明确区分事实与推断。

### 1.1 常用术语

| 术语 | 含义 |
| --- | --- |
| device program | 加载到 device 的独立可执行单元；独立 program 对应独立的 text allocation |
| fatbin | 多个 device object 的打包容器；并入同一 fatbin 的 kernel 属于同一 device program |
| SIMT VF / simt_entry | 一次 SIMT 向量函数调用，及其被 outline 后的入口函数 |
| `VF_SIMT` | scalar kernel 中发起 SIMT VF dispatch 的机器指令；本文的问题字段所在 |
| dispatch | caller 发起一次 SIMT VF 调用的动作 |
| PB / `PUSH_PB` | dispatch 的参数传递协议 |
| RVEC | SIMT 取指/执行路径上的向量指令流 |
| IFU | 取指单元（instruction fetch unit） |
| CA model / Camodel | cycle-approximate 仿真模型 |
| MTE2 / MTE3 | GM -> UB / UB -> GM 的搬运引擎 |
| D2H / H2D | device -> host / host -> device 数据拷贝 |
| TPER | keep/resume 绑定的物理寄存器约束名；slot N -> TPER(4+N) -> R{4+N} |

## 2. 一页结论

### 2.1 问题场景

RMSNorm 使用 persistent SIMT fragment，在同一个 scalar kernel 中发起两个
SIMT VF 调用，并通过 `pto.keep` / `pto.resume` 在两个 SIMT VF 之间保存线程
寄存器。三个 shape 的单 kernel 结果都正确，但在同一进程、同一 device、同一
stream 中连续加载并执行独立 device program 时，后一个 kernel 可能在首次
`VF_SIMT` dispatch 时失败。

最初的典型现象是：

```text
d=4096, batch=4096       PASS
d=5120, batch=4096       PASS
d=7168, batch=4096       aclrtSynchronizeStream -> 507035
                           vector core exception
                           errcode 355: VEC read GM address out of bounds
```

其他缩减实验中还出现过 errcode 341（UB access out of bounds）。这些 GM/UB
越界是错误指令流继续执行后的下游表现，不是最初的地址计算根因。

### 2.2 最终因果链

```text
pto.keep / pto.resume
        |
        v
空 inline asm + fixed TPER constraint
        |
        v
BiSheng 的 SIMT function-size 统计返回 -1
        |       （失败状态没有阻止后续 VF_SIMT 更新）
        v
-1 -> unsigned 0xffffffff -> 16-bit code-size 字段 0xffff
        |
        v
前序 kernel 声明约 256 KiB 的 SIMT text 范围
        |
        v
范围覆盖后续尚未加载的独立 device program 地址
        |
        v
后续 program 加载后，SIMT/RVEC 取指路径仍看到此前的全零内容
        |       （取指/cache 机制细节待 BIU/IFU 日志确认，见 2.3 与 15.2）
        v
错误 SIMT 指令流 -> vector core exception / GM 或 UB 越界
```

### 2.3 已确认与尚未完全确认

| 结论 | 依据 | 说明 |
| --- | --- | --- |
| 单独运行 d4096/d5120/d7168 可以通过 | 直接观测 | 不是固定 shape 的静态正确性问题 |
| 独立 device program 连续切换时后一个 SIMT 可能失败 | 真机矩阵 | 同一 device program/fatbin 组合可通过 |
| SIMT body 和 RMSNorm 主循环不是必要条件 | 保留 dispatch 的 IR 二分 | 空 body 仍失败，删除 dispatch 后异常消失 |
| 数据已经正确写入 GM | Host D2H、device MTE 探针 | 同一地址读出的 64 字节逐字节匹配 |
| 连续场景的 RVEC 取指看到全零 | CA trace | 正确 target PC 上出现连续 `0x0`，`VF_SIMT` 不 retire |
| d5120 的 `VF_SIMT` code size 为 `0xffff` 是直接原因 | 最小二进制 A/B | 只改两个字段后连续运行通过 |
| BiSheng 由空 inline asm 传播 `-1` 并编码成 `0xffff` | 源码分析 | function-size 与更新链路确认 |
| 具体 IFU invalidate/refill 时序 | 尚未直接观测 | 需要 BIU/IFU 深层日志，不影响编译器根因结论 |

## 3. 复现基线和实验对象

### 3.1 三个 kernel

最终真机矩阵使用的关键参数如下：

| shape | grid | SIMT threads | dynamic UB |
| ---: | ---: | ---: | ---: |
| 4096 | 64 | 128 | 82432 bytes |
| 5120 | 64 | 256 | 152576 bytes |
| 7168 | 64 | 256 | 160768 bytes |

每个 shape 都能从保存的 LLVM IR 重新构建 device object 和 host launcher，避免
把 TileLang cache、旧 object 或手工汇编残留误认为当前源码行为。

工具链基线为：

```text
CANN:    cann-9.1.T530
BiSheng: clang 15.0.5, commit 5c68a1cb1231
Device:  0
```

### 3.2 复现器同时保留单独启动与连续启动两种模式

问题最初只在 `example_rmsnorm_persistent_simtvf.py` 的同一进程连续运行中暴露。
因此复现器明确区分两种模式：

```text
single:   新进程 -> 加载一个 device program -> launch -> synchronize -> 退出
sequence: 一个进程 -> 依次加载多个 device program -> 每次 launch 后同步
```

后续还保存了不依赖 TileLang 的 PTODSL/PTO IR/LLVM IR/object/SO/host runner，
这样问题可以直接交给没有 TileLang 环境的同事复现。

### 3.3 初始硬件日志

典型错误的关键片段为：

```text
aclrtSynchronizeStream(kernel) failed: 507035
The vector core execution is abnormal.
errcode:(355) The address for VEC to read GM is out of bounds
fault kernel_name=rmsnorm_d7168_kernel, current=entry+0x1dc
```

需要注意，`current=entry+0x1dc` 或 `entry+0x1e8` 是异步异常报告时的 scalar
caller PC，不应直接解读为 SIMT entry 或 `VF_SIMT` 指令地址本身。

## 4. 定位方法：每轮都做“裁剪—复现—改写问题”

这次定位不是一次性猜中字段，而是反复执行以下闭环：

```text
记录现象
   |
   v
建立单变量 A/B 或最小控制
   |
   v
保留可比较的产物（IR/object/trace）
   |
   v
运行并记录“谁变了、谁没变”
   |
   v
缩小故障边界，淘汰一类假设
   |
   v
根据新边界设计下一轮实验
```

每轮实验至少保留四项信息：

1. 被改变的唯一变量；
2. 为保证控制结构合法而保留的内容；
3. 结果是通过、失败、超时还是无效实验；
4. 该结果能排除什么，不能排除什么。

整个定位过程可以压缩成八轮：

| 轮次 | 当时的问题定义或怀疑 | 关键实验 | 实验后如何改写问题 |
| ---: | --- | --- | --- |
| 1 | d7168 kernel 本身是否错误 | 单 shape、交换 shape 顺序 | 失败与后续 program 位置相关，不属于固定 shape |
| 2 | UB、thread bound、block ID、寄存器是否错误 | 固定/放大 UB、同线程数序列、入口状态检查 | 常规资源配置不是必要条件 |
| 3 | SIMT body 哪条指令错误 | 保留 dispatch，逐步清空 body 和主循环 | body 不必要，边界收敛到 SIMT dispatch |
| 4 | 是否为任意 SIMT program 切换问题 | 最小 SIMT、scalar-only、同 fatbin/独立 program | 完整前序 kernel 与独立 program 装载组合才触发 |
| 5 | caller 是否推错 SIMT PC | 单跑/连续 CA issue/retire 对照 | target 正确，但连续场景 RVEC 在该地址取零 |
| 6 | 第二个 program text 是否没有加载 | Host D2H 与 device GM -> UB -> GM | GM 中 text 正确，异常只在指令取指视图 |
| 7 | 哪个取指描述信息异常 | 检查 `VF_SIMT` 字段和 ELF symbol size | `0xffff` 声明范围远超真实 function |
| 8 | `0xffff` 是相关还是因果 | 仅修补两条字段的 6-byte A/B | 连续运行恢复，再回溯 BiSheng 的 `-1` 传播链 |

## 5. 第一轮裁剪：从“d7168 有问题”到“前序 program 影响后续 program”

### 5.1 交换 shape 顺序

真机顺序矩阵如下：

| 顺序或构造 | 结果 | 新结论 |
| --- | --- | --- |
| d4096 单独 | 通过 | d4096 具备独立正确性 |
| d5120 单独 | 通过 | d5120 具备独立正确性 |
| d7168 单独 | 通过且数值正确 | d7168 不是固定的坏 kernel |
| d5120 -> d7168，独立 program | d7168 `507035/355` | 前序 program 可能污染后续 SIMT |
| d7168 -> d5120，独立 program | d5120 `507035/355` | 失败者随顺序变化 |
| d4096 -> d5120，独立 program | d5120 `507035/355` | 128->256 与 256->256 均失败，线程数切换不是必要条件 |
| d5120 -> d5120 | 通过 | 同一 program/entry 重用不是同一问题 |

这里最重要的改写是：

```text
原问题：为什么连续运行时 d7168 会失败？
新问题：为什么一个完整 persistent-SIMT program 执行后，
        后续独立 program 的首次 SIMT dispatch 会失败？
```

### 5.2 早期代码差异和 block ID 假设

曾对比 TileLang 生成的 PTODSL、手写 PTODSL、LLVM IR 和反汇编，注意到部分
寄存器初始化指令存在差异，例如 `MOVI X15`、`MOVK X18` 等，也怀疑
`VF_SIMT` 后的 `BLOCKID` 读取和 `pto.get_block_idx()` 修复可能改变地址。

这些方向值得检查，但都不能解释以下组合事实：

- 每个 shape 单独运行正确；
- 改变 host program 的打包方式会改变结果；
- SIMT body 为空仍可失败；
- 只改前序 `VF_SIMT` 的 code-size 字段即可修复。

因此，汇编差异和 block ID 不是最终根因。后续 trace 也显示 scalar caller
推入的 SIMT target PC 正确。

## 6. 第二轮裁剪：排除资源配置和普通 kernel 状态

### 6.1 Dynamic UB

分别使用每个 shape 的正确 UB 值，以及统一设置约 192 KiB 的 common-UB 变体。
异常仍存在，某些变体只是改变了报错 kernel 或下游 errcode。

结论：UB 大小可能改变错误指令流撞到的地址，但不是触发跨 program 失败的
必要条件。

### 6.2 Launch bound

`d4096(128) -> d5120(256)` 和 `d5120(256) -> d7168(256)` 都失败，因此
线程数切换既不是充分条件，也不是必要条件。

### 6.3 `InitSocState` 和寄存器驻留

不同 kernel 的 scalar 入口会重新初始化运行状态；persistent register 的设计
目标是在同一 kernel 的两个 SIMT VF 之间保存 TPER 值，而不是在 kernel 之间保留
普通寄存器值。

后续真正相关的是：persistent keep/resume 的 lowering 使用了空 inline asm，
它触发了 BiSheng 的 code-size 统计缺陷；并不是 TPER 值跨 kernel 泄漏本身。

## 7. 第三轮裁剪：保留 SIMT 协议，只缩减 body

### 7.1 作废的实验：删除整个 `VF_SIMT`

早期直接注释整个 `VF_SIMT` 调用，破坏了 `PUSH_PB`、dispatch 参数和后续
等待之间的协议。Camodel 出现了如下 assertion：

```text
PemPB::push_pb(...)
((slot_pos % PUSH_PB_SREGS) == 0) &&
((slot_pos / PUSH_PB_SREGS) < PUSH_PB_MAX_SLOT_NUM)
```

这不是原始问题的证据，而是非法修改 caller 控制结构的结果。后续所有二分都
改为保留 caller、PB、metadata 和 drain wait，只修改 SIMT body 或主循环。

二分也从直接编辑汇编切换到修改 LLVM IR 后重新编译。直接替换汇编指令容易
改变指令布局、SIMT target offset 或 PB 协议，难以保证实验只改变 SIMT body。

### 7.2 合法的 LLVM IR 二分

| d7168 变体 | d5120 -> d7168 | 解释 |
| --- | --- | --- |
| 两个 SIMT body 为空，主循环保留 | `507035`，errcode 330 | body 内容不是必要条件 |
| 仅 SIMT1 body 为空 | `507035`，errcode 341 | SIMT0 路径足以触发 |
| 仅 SIMT0 body 为空 | `507035`，errcode 341/334 | 空 SIMT0 dispatch 仍失败 |
| 跳过主循环，保留 SIMT0、metadata、drain wait | `507035`，errcode 355/341 | 数学主循环不是必要条件 |
| 移除 SIMT call 与 metadata | 无 device exception，但结果错误 | dispatch 消失后不再触发 |
| 只保留 metadata | 无 device exception，但结果错误 | metadata 单独不足以触发 |
| metadata 保留，SIMT0 body 为 `ret void` | `507035`，errcode 334/355 | 空 body 仍不能规避 |

这轮实验把故障边界压缩为：

```text
SIMT 算术和访存 body 不是必要条件；
真正发起 VF_SIMT dispatch 才是必要条件。
```

如果只保留 SIMT call 而删除 `store.vfsimt.info`，release BiSheng 会在
`Identify HiTPE SIMT VF Info Config` 阶段崩溃。这种形态不是有效的运行对照，
不能拿来判断硬件根因。

## 8. 第四轮裁剪：比较 device program 的打包和装载边界

### 8.1 关键矩阵

| 构造 | 结果 | 说明 |
| --- | --- | --- |
| 每个 kernel 单独运行 | 通过 | 独立静态控制 |
| d5120/d7168 合并为一个 fatbin/device module | 通过 | text 一起进入同一 program |
| 两个 fat object 链入同一个 host SO | 通过 | runtime 形成同一 device program allocation |
| 两个 SO、两个独立 device program，d5120 -> d7168 | 失败 | 后续首次 SIMT 受影响 |
| d5120 -> scalar-only program | 通过 | 后续 scalar 路径正常 |
| scalar-only -> d5120 | 通过 | scalar 前序不会触发污染 |
| 最小 SIMT A -> 最小 SIMT B | 通过 | 不是任意 SIMT -> SIMT 都失败 |
| 最小 SIMT -> d5120 | 通过 | 完整 persistent 前序才是关键变量 |
| d5120 -> 最小 SIMT | 失败 | 后续最小 SIMT 也能成为受害者 |

因此关键边界不是 host `dlopen` 本身，而是独立 device program allocation 和
text 装载时序。

### 8.2 合法 preloader 的正反对照

同一目标 SO 中提供 scalar-only preloader：

```text
目标 scalar preloader -> d5120 -> 目标 d7168    通过
d5120 -> 目标 scalar preloader -> 目标 d7168    失败
```

第二种情况下 scalar preloader 已经正常执行，说明目标 binary 确实加载完成；
但 d7168 的首次 SIMT 仍失败。这个结果比“binary 没加载”更精确地指向
program 切换时的 SIMT/RVEC fetch context 或 cache 可见性。

## 9. 第五轮裁剪：Camodel 取指对照

### 9.1 使用正确的 issue/retire 语义

CA 日志中：

- `coreX.veccoreY.instr_popped_log.dump` 表示 issue/popped；
- `coreX.veccoreY.instr_log.dump` 表示 retire；
- 只能在相同 `coreX.veccoreY` 的日志对中按 instruction ID 配对；
- `VF_SIMT` 的 issue 不等于 SIMT body 已经完成，必须检查同 ID 是否 retire 以及
  后续 RVEC 指令。

### 9.2 单独运行 d7168

在 block 57、physical core 28、veccore1 的有效 trace 中，单独运行时可以看到：

```text
VF_SIMT issue  ->  有同 ID 的 retire
RVEC           ->  SIMT_S2R / SIMT_SHFI / SIMT_IADD_I / SIMT_END
```

这与 d7168 反汇编和预期 SIMT body 一致。

### 9.3 连续 d5120 -> d7168

连续场景的关键 issue 行为是：

```text
[00188616] (PC: 0x9000d0f0fc) PUSHQ : ... (ID: 8051975) VF_SIMT
[00188637] (PC: 0x9000d0f120) RVECLD: (Binary: 0x0000000000000000) ... SIMT_LDG
[00188638] (PC: 0x9000d0f120) RVECLD: (Binary: 0x0000000000000000) ... SIMT_LDG
[00188639] (PC: 0x9000d0f120) RVECLD: (Binary: 0x0000000000000000) ... SIMT_LDG
```

对应的 retire 日志只有前面的 MTE3/wait：

```text
[00188613] ... (ID: 8051946) MOV_SRC_TO_DST_ALIGNv2
[00188614] ... (ID: 8051948) WAIT_FLAG ...
```

retire 日志中没有 ID `8051975` 的 `VF_SIMT`。全零 `SIMT_LDG` 事件本身可能
被模型 retire，但它们代表取指路径看到的错误指令，不代表正确 SIMT body 已经
执行。

Camodel 没有稳定复现真机终态 `507035`：完整 64-block 序列过慢，1-block
序列可进入第二个 launch 但在观察窗口内不结束。因此上述 trace 是取指侧的
定位证据，不应表述为“camodel 完整复现了硬件错误”。

## 10. 第六轮裁剪：GM -> UB -> GM 探针

### 10.1 探针地址计算

SIMT entry 位于内嵌 device ELF 中，host 不直接导出该符号。探针通过：

```text
SIMT0 runtime address
  = scalar kernel runtime entry
  + (ELF st_value(SIMT0) - ELF st_value(scalar entry))
```

例如一轮 d7168 探针中：

```text
scalar entry:  0x9000d0f000
SIMT0 offset:  0x120
SIMT0 address: 0x9000d0f120
```

### 10.2 两条数据通路

在 `VF_SIMT` 前同时做：

```text
已加载 SIMT text --MTE2--> UB[32768:32832]
UB[32768:32832] --MTE3--> debug GM
```

Host 再从同一 runtime 地址做 D2H，并与 debug GM 比较。

连续场景中两边都得到非零且完全一致的数据：

```text
host-D2H text
  00: 5c 06 9e 02 00 2e 00 06 9c 00 0e 12 08 0a 06 06
  10: 5d 00 0e 00 00 00 06 04 01 00 00 00 00 00 00 00

kernel MTE GM->UB->GM
  00: 5c 06 9e 02 00 2e 00 06 9c 00 0e 12 08 0a 06 06
  10: 5d 00 0e 00 00 00 06 04 01 00 00 00 00 00 00 00

probe changed=yes exact-match=yes all-zero=no
```

结论是：

```text
GM 中已有正确程序字节       yes
MTE 数据通路可以读到字节     yes
RVEC/SIMT IFU 看到正确字节   no（取到全零）
```

这排除了 H2D 未加载、GM 数据通路读不到 text、ELF text 被截断和 scalar target 计算错误，
将问题边界收窄到 SIMT/RVEC 的取指可见性、指令 cache/映射或 per-program fetch
context。

## 11. 第七轮裁剪：二进制字段定位

### 11.1 `VF_SIMT` code size 字段

对 d5120 caller 反汇编，发现两条指令为：

```text
0x15fffffc15272c00  VF_SIMT X7,  X5, #65535, #0, #1
0x15fffffc15372c00  VF_SIMT X23, X5, #65535, #0, #1
```

本次修补只触碰 code-size 字段 bit `[52:37]`：

```text
63                    53 52                    37 36                     0
+-----------------------+------------------------+--------------------------+
|       其他字段        |      code_size[15:0]   | 其他控制/寄存器字段      |
+-----------------------+------------------------+--------------------------+
                          ^ only patch this field

code_size_words = (instruction >> 37) & 0xffff
code_size_bytes = code_size_words * 4
```

字段以 4-byte instruction word 为单位，`0xffff` 表示：

```text
65535 words = 262140 bytes = 0x3fffc bytes
```

`VF_SIMT` 的其他控制字段（包括低位的 join/exit 相关字段）保持不变；A/B
实验只改 `[52:37]`。

### 11.2 与 ELF symbol size 对照

| callee | ELF `st_size` | 本次修补值 (`st_size / 4`) | 原始 field | 修补 field |
| --- | ---: | ---: | ---: | ---: |
| `rmsnorm_d5120_kernel_simt_0_simt_entry` | 232 (`0xe8`) bytes | 58 (`0x3a`) | `0xffff` | `0x003a` |
| `rmsnorm_d5120_kernel_simt_1_simt_entry` | 2320 (`0x910`) bytes | 580 (`0x244`) | `0xffff` | `0x0244` |

这里的 `st_size / 4` 是本次修补使用的安全替代值；正常有限的 BiSheng code size
不要求在所有情况下都与 symbol size 严格相等。运行时 d5120 与 d7168 scalar
program base 间距约为 `0x2000`：

```text
d5120 scalar base: 0x100040800000
d5120 SIMT0:       base + 0x218
d5120 SIMT1:       base + 0x300
d7168 scalar base: 0x100040802000
d7168 SIMT0:       base + 0x208
```

实际 ELF text 并没有物理重叠，但两个 `0x3fffc` 的声明范围都足以覆盖后续
program 的 SIMT 地址：

```text
d5120 SIMT0:  0x100040800218 ─┬─ 实际 0xe8 字节
                              └─ 声明 0x3fffc ───────────┐
d5120 SIMT1:  0x100040800300 ─┬─ 实际 0x910 字节
                              └─ 声明 0x3fffc ───────────┤
d7168 base:   0x100040802000 <───────────────────────────┘ 两条声明范围均覆盖
d7168 SIMT0:  0x100040802208 <── 取指看到全零的位置
```

这解释了为什么同一 device module 可以通过，而独立 program 的后续 SIMT 会受影响。

## 12. 第八轮裁剪：最小二进制 A/B 闭环

### 12.1 修改前后

只改 d5120 两条 `VF_SIMT` 的 code-size 字段：

```text
SIMT0:
  raw     0x15fffffc15272c00
  patched 0x15e0075c15272c00
  size    0xffff -> 0x003a (232 bytes)

SIMT1:
  raw     0x15fffffc15372c00
  patched 0x15e0489c15372c00
  size    0xffff -> 0x0244 (2320 bytes)
```

除上述字段外，以下内容完全不变：

```text
scalar code      SIMT body      PB 参数      dynamic UB
host launcher    launch 顺序    SO 布局      d7168 binary
```

最终 linked device binary 只有 6 个字节变化（每条指令对应字段跨越 3 个字节）。

### 12.2 真机结果

| 实验 | 结果 |
| --- | --- |
| 原始 d5120 -> 原始 d7168 | d7168 稳定 `507035/355` |
| 修补 d5120 -> 原始 d7168，clean run 1 | 两个 kernel 完成，抽样值正确，退出 0 |
| 修补 d5120 -> 原始 d7168，clean run 2 | 相同结果，退出 0 |
| 原始 d4096 -> 修补 d5120 | 仍失败 |

修补序列日志中的数值校验片段为：

```text
result d=5120 row=0    rstd=3.85358191 expected=3.85358191
result d=5120 row=4095  rstd=3.85358191 expected=3.85358191
result d=7168 row=0    rstd=3.85351181 expected=3.85351181
result d=7168 row=4095  rstd=3.85351181 expected=3.85351181
process exit code: 0
```

中间曾有一次设备计算和数值校验均完成、但 host cleanup 以退出码 139（SIGSEGV）
结束的运行；新
进程立即复跑得到 clean exit 0。该 cleanup 现象不属于 device `507035`，不改变
A/B 结论。

A/B 图可以简化为：

```text
                 前序 d5120                         后续 d7168
原始 object      VF_SIMT size=0xffff  -----------> 507035 / fetch zero

仅改 6 bytes     VF_SIMT size=0x3a/0x244 --------> 正确执行 / 数值匹配
```

只修补后一个 kernel 仍失败，说明污染来自前序未修补 kernel（d4096 的两条 `VF_SIMT`
code-size 字段同为 `0xffff`）；这也是区分“受害者”和“污染源”的关键实验。

### 12.3 A/B 的证明力

前面的顺序矩阵、IR 二分和取指 trace 都是在收缩候选范围。这组 A/B 则满足
直接因果验证的条件：

- 修改对象唯一：两条 `VF_SIMT` 的 code-size 字段；
- 修改方向可解释：从异常最大值改为由对应 callee `st_size` 换算的安全值；
- 受害 kernel 未改动：d7168 仍使用原始 binary；
- 结果可重复：两次 clean run 均通过并退出 0；
- 反向控制成立：前序 d4096 未修补时，后续已修补 d5120 仍失败。

因此不再需要把 code-size 字段是否参与故障作为未知项。

## 13. 回溯 BiSheng 根因

### 13.1 PTOAS 的 lowering 形态

`pto.keep` / `pto.resume` 使用空 inline asm 和 fixed TPER constraint 表达
物理寄存器 def/use。等价 LLVM IR 形态如下：

```llvm
; keep: asm 模板为空，不生成设备指令
%keep = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5},0,1"(i32 %a, i32 %b)

; resume: 从固定 TPER 读取值
%resume = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5}"()
```

空模板没有设备指令大小，但 Machine IR 中仍然保留 inline-asm 节点，供寄存器
约束和 def/use 建模使用。

### 13.2 错误传播路径

BiSheng HiIPU backend 的 function-size 处理可以概括为：

```text
computeBlockSize()
  遇到 inline asm -> return -1
        |
        v
computeFunctionSize() -> 保存 -1
        |
        v
calculateOffsets() -> 记录失败，但没有阻止后续更新
        |
        v
updateVFInstrNum()
  (-1 >> 2) -> unsigned 0xffffffff -> 16-bit field 0xffff
```

问题有两个相互叠加的缺陷：

1. 没有把不产生设备指令的空 inline asm 按 0 字节处理；
2. size 计算失败后仍继续写 `VF_SIMT`，把内部错误值变成了看似合法的机器码。

这解释了为什么 persistent-register 版本触发问题，而普通 SIMT 版本不一定触发：
不是 TPER 值本身跨 kernel 泄漏，而是 keep/resume 的表示方式改变了 BiSheng 的
function-size 统计输入。

## 14. 修复方案和验证策略

### 14.1 BiSheng 方案 A：正确处理空 inline asm

优先要求：

- 严格为空的 asm template 按 0 字节计数；
- 非空且无法确定长度的 inline asm 保持失败；
- size 为负数、未知、未对齐或超出字段范围时，禁止更新 `VF_SIMT`；
- 给出明确编译诊断，不生成 `VF_SIMT #65535`。

这是改动最小的根因修复，不改变 PTOAS IR、TPER slot 规则或 SIMT body。

### 14.2 BiSheng 方案 B：named-register intrinsic

长期可以用 `llvm.read_register` / `llvm.write_register` 表达 TPER 的物理
寄存器读写，移除对空 inline asm 的依赖。但这需要 BiSheng 支持：

- TPER named register 的 `i32`/`i64` 读写；
- 64-bit pair 与 32-bit sub-register alias；
- register allocator 的 reserved/def/use 建模；
- copy lowering、调度和 code-size 统计。

方案 B 更完整，但改动和发布周期大于方案 A。

### 14.3 PTOAS 临时 object patch

在修复版 BiSheng 发布前，PTOAS 在 BiSheng 生成 raw vector object 后：

1. 从 LLVM module 收集 SIMT caller/callee manifest；
2. 从未 strip 的 ELF symbol table 读取 callee `st_size`；
3. 在 caller symbol 范围内识别 `VF_SIMT`，并解码目标地址构造序列；
4. 仅对 manifest、callee symbol 和机器 callsite 三方一致的 `0xffff` 修补；
5. 对 relocation、字段范围、字节差异和修补后 object 做校验；
6. 把 patched object 交给后续 device merge。

该方案只改 code-size 字段，不改变 SIMT body、PB、同步或寄存器语义。

### 14.4 修复后的回归层次

| 层次 | 回归目标 |
| --- | --- |
| BiSheng unit test | 空 inline asm 按 0 字节，失败值不能进入编码 |
| PTOAS lit | manifest/symbol/callsite 三方校验和 object patch |
| 静态 object 检查 | 不再出现本问题产生的 `0xffff`，有限值不越过 callee symbol |
| 真机单 kernel | d4096/d5120/d7168 数值正确 |
| 真机 sequence | 独立 program 的 d4096 -> d5120 -> d7168 连续通过 |
| 新工具链退出策略 | `verify` 模式全部 no-op 后删除临时 patcher |

## 15. 失败实验和解读边界

### 15.1 不能作为证据的结果

- 直接删除整个 `VF_SIMT` 后出现的 `PUSH_PB` assertion：caller 协议已被破坏；
- 删除 `store.vfsimt.info` 后 release BiSheng 崩溃：缩减形态不符合编译器预期；
- 早期 MTE2/MTE3 没有 event 依赖时的全零探针：存在流水线时序错误；
- 观察窗口内 camodel 超时：不等于真机一定通过或一定失败。

### 15.2 不应过度表述的结论

当前可以说：

```text
错误的 VF_SIMT code size 会触发跨独立 program 的 SIMT 取指故障。
```

在未取得 BIU/IFU 深层日志前，不应表述为：

```text
已经精确证明某个 cache line 在某个 cycle 被 prefetch，
并且某个 invalidate 指令一定没有执行。
```

后者是合理的硬件机制假设，但不是本次 compiler A/B 必须证明的内容。

## 16. 可复用的定位清单

遇到类似“单独通过、连续 launch 失败”的异步 device 错误时，建议按以下顺序：

1. 固定 device、stream、grid、UB 和工具链版本；
2. 同时保存单独启动与连续启动控制；
3. 交换顺序，判断失败者是否随位置移动；
4. 区分 host SO、device program allocation 和 fatbin 边界；
5. 对资源参数做单变量 A/B，但不要把 errcode 变化误认为根因变化；
6. 缩减 IR 时保留 dispatch、metadata、PB 和流水线同步；
7. 先查 scalar caller 的目标地址，再查 SIMT/RVEC 实际取到的 binary；
8. 用 host D2H 和 device MTE 探针区分“GM 中没有 text”和“IFU 看不到 text”；
9. 对异常机器指令字段做范围、单位和 symbol size 对照；
10. 最终构造只修改一个字段的最小 binary A/B；
11. A/B 成立后再回溯 compiler pass 的错误值传播链；
12. 把已证明的事实和尚未证明的 runtime/硬件机制分开记录。

## 17. 附录：12 页团队分享大纲

以下内容由正文直接裁剪，供 12 页团队分享使用，可独立取用。每页只讲一个
判断，不把所有尝试堆在同一页。

| 页 | 标题 | 页面核心内容 | 建议放置的证据/图 | 讲述重点 |
| ---: | --- | --- | --- | --- |
| 1 | 问题是什么 | 单独通过，连续启动后续 SIMT 报 `507035` | 一行错误日志 + 三 shape 流程图 | 先讲反常现象，不先猜根因 |
| 2 | 复现条件 | 同一进程、独立 device program、后续首次 `VF_SIMT` | `single`/`sequence` 两条时间线 | 明确同步点和 program 边界 |
| 3 | 第一轮矩阵 | 交换顺序后受害 shape 变化 | d4096/d5120/d7168 顺序表 | 失败不属于固定 shape |
| 4 | 排除资源因素 | UB、thread bound、`InitSocState`、block ID | 参数表和排除表 | 错误码变化不等于根因变化 |
| 5 | IR 二分原则 | 保留 caller/PB/metadata/wait，只删 body | 正确与作废缩减示意图 | 删除整个 `VF_SIMT` 会破坏协议 |
| 6 | SIMT 故障边界 | 空 body、跳过主循环仍失败 | IR 二分结果表 | 必要条件是 dispatch，不是算术 body |
| 7 | 打包/装载对照 | 同 fatbin 通过，独立 program 失败 | program allocation 示意图 | 关键是 device program，不是 `dlopen` |
| 8 | Camodel 取指证据 | 正确 target PC，连续场景 RVEC 取零 | issue/retire 片段 | `VF_SIMT` issue 不等于 retire |
| 9 | 数据通路证据 | D2H 与 GM->UB->GM 都正确 | 两段 64-byte hex 对照 | GM 有 text，问题在 IFU 视图 |
| 10 | 二进制异常字段 | `0xffff` 与真实 `st_size` 对照 | 指令字段位图 + 地址范围图 | 解释为什么范围能跨 program |
| 11 | 6-byte A/B | 只改前序两个字段，失败变通过 | raw/patched 表 + 结果矩阵 | 把相关性变成因果证据 |
| 12 | BiSheng 根因与修复 | `-1 -> 0xffff`，方案 A/B/PTOAS patch | 编译器传播链 + 修复路线 | 结束于可执行的修复和经验 |

### 第 1 页建议讲稿

```text
这不是 d7168 单独运行错误，而是一个跨 device program 的顺序相关错误。
如果只看最后的 GM 越界，会沿着错误指令流走很远；今天的重点是如何把它
从异步硬件症状收敛到一个具体的编译器编码字段。
```

### 第 8 页建议讲稿

```text
CA model 没有稳定复现真机的 507035，但它给出了更有区分度的中间现象：
scalar caller 推入的地址正确，RVEC 在这个地址连续取到 0，VF_SIMT 没有 retire。
所以我们把数据地址问题和指令取指问题分开了。
```

### 第 11 页建议讲稿

```text
这是整个定位的闭环：不改 kernel 数学、不改 launcher、不改 UB、不改 d7168，
只改 d5120 两条指令的 code-size 字段，最终 binary 只差 6 个字节，连续运行
立即恢复。此时再回头读 BiSheng 的 size pass，根因就不再是猜测。
```

## 18. 证据来源

正文中的数据来自以下记录；大体积 object、SO 和完整数 GB trace 不纳入本文：

- [PTOAS object patch 方案](vpto-vfsimt-size-object-patch-plan.md)

以下文件是本地归档，不作为阅读本文的前置条件，也不要求随本文提交：

```text
/home/qukelin/projects/PTOAS/docs/designs/
  bisheng-vfsimt-code-size-inline-asm-issue.md

/home/qukelin/projects/PTOAS/test/vpto/cases/onboard-only/
  rmsnorm-persistent-fragment-sequence-repro/EXPERIMENT_STATUS.md
  rmsnorm-persistent-fragment-sequence-repro/hardware-matrix-20260729/README.md

/home/qukelin/test/persistent/rmsnorm-text-mte-probe/results/
  text-sequence-vfsimt.log
  vfsimt-trace-evidence.log
```

## 19. 最终结论

这次问题的直接根因不是 RMSNorm 的数学逻辑、UB 大小、block ID 或某条 SIMT
算术指令，而是 persistent keep/resume 采用空 inline asm 后，BiSheng function
size 统计失败值未被拦截，最终把 `VF_SIMT` code size 编码为 `0xffff`。

`0xffff` 使前序 kernel 声明了远超真实 SIMT text 的取指范围。在独立 device
program 连续切换时，这个错误范围与后续 text 装载时序组合，造成 SIMT/RVEC
取指路径看到全零指令，随后才表现为 vector core exception 和 GM/UB 越界。

最可靠的定位证据是：只修改前序 d5120 两个 `VF_SIMT` 的 `[52:37]`，将
`0xffff` 改为 `0x3a/0x244`，其他内容保持不变，连续 d5120 -> d7168 立即通过。
