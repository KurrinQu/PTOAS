# [HiIPU][VF_SIMT] 空 inline asm 导致 code size 被编码为 `0xffff`

## 问题概述

BiSheng 在统计 SIMT vector function 的 code size 时，只要 machine basic block
中出现 inline asm，就会将该 function 的大小记为 `-1`。后续流程没有拦截这个
失败值，而是继续把它写入 `VF_SIMT` 的 code size 字段，最终 object 中的值变成
`0xffff`。

本问题中的 inline asm 模板为空，不生成任何设备指令，只用于通过 fixed TPER
constraint 表达跨 SIMT VF 调用的物理寄存器 def/use。因此，SIMT function 的
真实大小是可计算的，空 inline asm 应按 0 字节处理。

错误的 `0xffff` 不只是反汇编显示异常。硬件会把它解释为 65535 条 4-byte
指令，即 262140 字节的有效 SIMT text 范围。当不同 kernel 分别位于不同 `.so`
并在同一进程中依次加载、执行时，前一个 kernel 的错误范围会影响后一个 kernel
的 IFU 取指，最终导致 vector core exception。

## 问题场景

PTOAS 使用 `pto.keep`/`pto.resume` 在同一个 scalar kernel 发起的连续 SIMT VF
调用之间保留每个线程的寄存器值。LLVM IR 使用空 inline asm 和 fixed TPER
constraint 表达固定物理寄存器的写入和读取。下面是简化后的等价形式：

```llvm
; Keep: 将两个值写入 TPER4/TPER5。
%keep = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5},0,1"(i32 %a, i32 %b)

; Resume: 从 TPER4/TPER5 读回两个值。
%resume = call { i32, i32 } asm sideeffect "",
  "={TPER4},={TPER5}"()
```

asm 模板严格为空，不会生成设备指令，但 LLVM Machine IR 中仍存在 inline-asm
节点。d5120 RMSNorm 的两个 `simt_entry` 都包含这种节点：第一个 SIMT VF 在退出
前 keep，第二个 SIMT VF 在入口 resume，并在退出前再次 keep。

静态复现只需编译一个包含上述空 inline asm 的 SIMT function。问题产物使用的
主要编译参数如下：

```bash
bisheng d5120.ll \
  --cce-aicore-arch=dav-c310-vec \
  --cce-aicore-only \
  --cce-generic-addrspace=off \
  -O2 \
  -cce-bitcode-is-aicore \
  -Wno-override-module \
  -mllvm -cce-dyn-kernel-stack-size=true \
  -mllvm -cce-vf-auto-sync=global \
  -c --save-temps \
  -o d5120.o
```

已确认问题的 BiSheng 版本为 clang 15.0.5、commit `5c68a1cb1231`，目标为
`hiipu64-hisilicon-cce` / `dav-c310-vec`。真机复现使用 CANN
`cann-9.1.T530`。

硬件复现场景如下：

1. `kernel1` 和 `kernel2` 分别打包在 `libkernel1.so` 和
   `libkernel2.so` 中；
2. 同一进程先加载并执行 `kernel1`，其中依次调用 `simt_vf0` 和
   `simt_vf1`；
3. 再加载并执行 `kernel2`，其中依次调用 `simt_vf2` 和 `simt_vf3`；
4. 每个 kernel 单独运行均可通过，但上述连续运行方式会在 `kernel2` 失败。

## 产物现象

### `VF_SIMT` code size 异常

d5120 真机 A/B 产物的两个 callsite 都被编码为 `#65535`：

```asm
15fffffc15272c00  VF_SIMT X7,  X5, #65535, #0, #1
15fffffc15372c00  VF_SIMT X23, X5, #65535, #0, #1
```

不同构建中的寄存器分配可能不同，但 code size 均为 `0xffff`，不影响问题。

`VF_SIMT` 是 64-bit scalar 指令，code size 位于 bit `[52:37]`，字段宽度为
16 bit，单位为 4 bytes：

```text
code_size = (instruction >> 37) & 0xffff
```

因此，`0xffff` 表示的范围为：

```text
65535 * 4 = 262140 bytes (0x3fffc)
```

最终 ELF 中 SIMT function 的 `st_size` 已经反映真实代码大小。d5120 两个
callee 的实际值如下：

| SIMT function | ELF `st_size` | 正确 code size | 当前编码 |
| --- | ---: | ---: | ---: |
| `rmsnorm_d5120_kernel_simt_0_simt_entry` | 232 bytes (`0xe8`) | 58 (`0x003a`) | 65535 (`0xffff`) |
| `rmsnorm_d5120_kernel_simt_1_simt_entry` | 2320 bytes (`0x910`) | 580 (`0x0244`) | 65535 (`0xffff`) |

### 真机运行异常

原始 d5120 `.so` 执行完成后，再加载并执行原始 d7168 `.so`，d7168 在 stream
同步时报错：

```text
aclrtSynchronizeStream(kernel) failed: 507035
The vector core execution is abnormal.
errcode:(355) The address for VEC to read GM is out of bounds(exceeding 48 bits).
```

部分实验中也出现过 `errcode:(341)` 的 UB access out-of-bound。GM/UB 越界是
错误指令流继续执行后的下游症状，不是 RMSNorm 的 GM 地址或 dynamic UB size
配置错误。

CA trace 还提供了取指侧的直接现象：d7168 单独运行时，第一个 `VF_SIMT`
可以 retire，RVEC 从目标地址执行与反汇编一致的 `SIMT_S2R`、`SIMT_SHFI`、
`SIMT_IADD_I` 和 `SIMT_END`；d5120 -> d7168 连续运行时，scalar caller 仍计算
出同一个正确 SIMT target，但该地址及后续地址取到的 64-bit binary 均为
`0x0000000000000000`，解码成 `SIMT_LDG`，`VF_SIMT` 也不再 retire。例如：

```text
(PC: 0x9000d0f100) RVECLD :
  (Binary: 0x0000000000000000) SIMT_LDG
```

同时，host D2H 和 kernel 内 `GM -> UB -> GM` 探针都能从同一个 SIMT text
地址读到正确的非零字节，并且与 linked `.aicore_binary` 逐字节一致。因此，
全零内容只出现在 RVEC/SIMT 取指路径，不是第二个 program 的 text 没有加载到
GM。

关键运行结果如下：

| 运行方式 | 结果 |
| --- | --- |
| d5120 单独运行 | 通过 |
| d7168 单独运行 | 通过 |
| d5120、d7168 位于同一个 fatbin | 通过 |
| 两个 binary 位于同一个 `.so` | 通过 |
| 两个 kernel 分别位于不同 `.so`，d5120 -> d7168 | d7168 报 `507035/355` |
| 只修正前序 d5120 的两个 code size，再运行 d5120 -> d7168 | 通过，结果与 CPU 参考一致 |

## 根因

问题位于 HiIPU backend 的 `HiIPUVFLabelOpt.cpp`，涉及 function-size 统计和
`VF_SIMT` 更新流程；相关声明位于 `HiIPUVFLabelOpt.h`。

### 1. 遇到 inline asm 后直接返回 `-1`

`computeBlockSize()` 没有区分空模板和真正包含汇编指令的 inline asm：

```cpp
int64_t HiIPUVFLabelOpt::computeBlockSize(MachineBasicBlock &MBB) {
  int64_t Size = 0;
  for (MachineInstr &MI : MBB) {
    if (MI.isInlineAsm()) {
      LLVM_DEBUG(dbgs() << "Inline assembly instruction detected: ";
                 MI.dump();
                 dbgs() << "Skip optimization of VF Label MOVK instructions\n");
      return -1;
    }

    Size += TII->getInstSizeInBytes(MI);
    // ...
  }
  return Size;
}
```

`computeFunctionSize()` 随后把该值作为整个 SIMT function 的大小返回：

```cpp
BlocksInfo[&MBB].Size = computeBlockSize(MBB);
if (BlocksInfo[&MBB].Size < 0)
  return BlocksInfo[&MBB].Size;
```

### 2. 失败状态没有阻止 code size 更新

`calculateOffsets()` 发现 size 小于 0 后只把返回值设为 `false`，仍将 `-1`
保存在 `MachineFunctionsInfo[MF].Size` 中：

```cpp
MachineFunctionsInfo[MF].Size =
    computeFunctionSize(*MF, MachineFunctionsInfo[MF].BlocksInfo);
if (MachineFunctionsInfo[MF].Size < 0)
  Changed = false;
```

`runOnModule()` 不检查 `calculateOffsets()` 是否成功，无条件继续执行
`updateVFInstrNum()`：

```cpp
bool HiIPUVFLabelOpt::runOnModule(Module &Module) {
  // ...
  bool Changed = calculateOffsets();
  updateVFInstrNum();
  return Changed;
}
```

### 3. `-1` 被转换并截断为 `0xffff`

处理 `VF_SIMT` 时，代码没有校验 callee size：

```cpp
if (MI.getOpcode() == HiIPU::VF_SIMT && Callee != nullptr) {
  unsigned AcVFInstrNum =
      static_cast<unsigned>((MachineFunctionsInfo[Callee].Size >> 2));
  MI.getOperand(2).setImm(AcVFInstrNum);
  Callee = nullptr;
}
```

在本次工具链构建中，失败值按以下路径传播：

```text
MachineFunctionsInfo[Callee].Size = -1
             -1 >> 2              = -1
static_cast<unsigned>(-1)          = 0xffffffff
写入 16-bit code size 字段          = 0xffff
```

也就是说，`0xffff` 不是 SIMT function 的真实大小，而是内部失败值未经检查后
转换、截断得到的结果。

## `0xffff` 导致跨 `.so` 失败的过程

当两个 kernel 分别位于不同 `.so` 时，运行故障按以下顺序发生：

1. runtime 先加载 `libkernel1.so` 并执行 `kernel1`；
2. `kernel1` 的 `simt_vf0`、`simt_vf1` 对应的 `VF_SIMT` code size 都是
   `0xffff`，声明的 text 范围远大于两个 SIMT function 的实际大小；
3. 该范围越过 `libkernel1.so` 的实际 SIMT text，覆盖后续
   `libkernel2.so` 将使用的 `simt_vf2`、`simt_vf3` 地址；
4. 此时 `libkernel2.so` 尚未加载，现有 A/B 和取指 trace 表明 IFU 根据错误
   范围提前读取了这些地址，得到全零内容并形成缓存状态；
5. runtime 随后加载 `libkernel2.so`，但执行 `kernel2` 时 IFU 仍使用此前读取
   的内容；
6. `simt_vf2` 取出的指令 binary 持续为全零，错误指令流破坏地址计算或同步
   状态，真机最终报告 vector core exception 以及 GM/UB 越界。

本次运行中，d5120 和 d7168 的 scalar kernel entry 分别为
`0x100040800000` 和 `0x100040802000`。`0xffff` 对应约 256 KiB 的 SIMT text
范围，足以跨过两个 program 之间的地址间隔。

多个 kernel 位于同一个 fatbin 或同一个 `.so` 时，相关 text 在第一次执行前
已经完成加载，因此不会读取到尚未加载区域的全零内容。这解释了同一份 device
代码在不同打包、加载方式下表现不同的原因。

目前没有打开 BIU 深层日志直接观察首次 read 的时刻，因此“提前取指后缺少
invalidate/refill”是对运行时传播机制的解释；空 inline asm 导致 code size
变为 `0xffff`，以及只修复该字段即可消除故障，则已经由源码和 A/B 实验确认。

## A/B 验证

A/B 实验只修改 d5120 两个 `VF_SIMT` 指令的 bit `[52:37]`，其他指令、SIMT
body、PB 参数、dynamic UB、host launcher 和 launch 顺序均保持不变：

```text
SIMT0: 0x15fffffc15272c00 -> 0x15e0075c15272c00
       0xffff -> 0x003a words = 232 bytes

SIMT1: 0x15fffffc15372c00 -> 0x15e0489c15372c00
       0xffff -> 0x0244 words = 2320 bytes
```

修补后的最终 device executable 与原始产物仅有 6 个 byte 不同。结果为：

- 原始 d5120 -> 原始 d7168：d7168 稳定报 `507035/355`；
- 修补 d5120 -> 原始 d7168：两次 clean run 均通过，d5120/d7168 数值与 CPU
  参考一致；
- 原始 d4096 -> 修补 d5120：d5120 仍报 `507035/355`，说明只修补后一个
  kernel 无法清除前序未修补 kernel 已经留下的 IFU 状态。

这组实验确认，前序 kernel 中错误的 `VF_SIMT` code size 是跨 `.so` 连续运行
失败的直接原因。

## 修复需求

建议同时修复 size 统计和失败值传播，避免同类问题再次生成看似合法的机器码。

1. 对 asm 模板严格为空的 inline asm 按 0 字节处理，并继续统计 basic block
   中的其他机器指令；
2. 对模板非空且无法准确计算大小的 inline asm 保持失败，不估算其指令数；
3. 将 `calculateOffsets()` 的成功/失败状态与 pass 的 `Changed` 返回语义分开，
   size 计算失败时禁止调用 `updateVFInstrNum()`；
4. 写入 `VF_SIMT` 前校验 callee size 非负、4-byte 对齐，并且换算结果能够由
   code size 字段合法表达；
5. 任一校验失败时给出明确编译错误，禁止把负数、未知值或溢出值写入 object。

建议增加以下回归测试：

- SIMT callee 含空 inline asm：编译成功，最终 code size 等于 callee ELF
  `st_size / 4`；
- SIMT callee 含无法确定长度的非空 inline asm：明确报错，不生成
  `VF_SIMT #65535`；
- caller 含多个 SIMT callee：每个 callsite 分别写入对应 callee 的真实大小；
- callee size 未对齐或超出字段范围：明确报错；
- 两个 kernel 位于不同 `.so` 并在同一进程中连续加载、执行：结果正确，不再
  出现第二个 kernel 的 vector core exception。
