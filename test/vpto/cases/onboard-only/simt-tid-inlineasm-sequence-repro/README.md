# SIMT tid inline-asm 连续启动复现（最小 device 裁剪用例）

本用例复现 `docs/designs/bisheng-vfsimt-code-size-inline-asm-issue.md` 中记录的
BiSheng `VF_SIMT` code-size 问题：

```text
kernel1 的 SIMT body 含空 inline asm（keep/resume 形态，fixed TPER 约束）
  -> BiSheng function-size 统计返回 -1
  -> kernel1 的 VF_SIMT code-size 字段 = 0xffff（65535 words ≈ 256 KiB）
  -> kernel1 声明的取指范围覆盖尚未加载的 kernel2 的 SIMT text 地址
  -> 同进程连续 launch 时，kernel2 的首次 SIMT dispatch 取指异常
```

## 用例内容

- `kernels/k1_inlineasm/tid_asm_kernel.ll`：手写 VPTO LLVM IR。scalar 侧循环
  dispatch 64 次（足够的 dispatch 次数是污染触发的必要条件）；SIMT body 先
  `llvm.hivm.get.TID.X()`，再经一对空 inline asm（`"={TPER4},0"` keep /
  `"={TPER4}"` resume）传递 tid，最后 `store i32` 到 GM。
- `kernels/k2_plain/tid_plain_kernel.ll`：同样的 SIMT body，但直接读 tid 写
  GM，不含 inline asm（受害者）。
- 两个 kernel 各自链接成独立 `libkernel.so`（独立 device program）。
- `host/sequence_main.cpp`：同一进程 dlopen 两个 SO，同一 device、同一 stream
  连续 launch，同步后 D2H 校验 128 个 lane 的 tid。

## 构建与运行

```bash
./build.sh

# 连续启动 k1 -> k2（预期 k1 PASS，k2 FAULT 507035）
ASCEND_RT_VISIBLE_DEVICES=0 ACL_DEVICE_ID=0 ./run.sh

# 单 kernel 对照（预期都 PASS）
TID_REPRO_ONLY=k1 ./run.sh
TID_REPRO_ONLY=k2 ./run.sh

# 反序对照 k2 -> k1（预期都 PASS）
TID_REPRO_REVERSE=1 ./run.sh
```

默认 `CANN_HOME=/home/qukelin/tools/CANN_9.1/cann-9.1.T530`。在其他环境运行时，
通过 `CANN_HOME` 指定 CANN 安装目录；也可单独覆盖 `BISHENG_BIN`、
`BISHENG_CC1`、`LD_LLD` 和 `CLANG_RES`。host stub 的 `-triple` 与 `-target-cpu`
默认按 `uname -m` 自动选择（`x86_64` → `x86_64-unknown-linux-gnu`/`x86-64`，
`aarch64` → `aarch64-unknown-linux-gnu`/`generic`），可通过 `HOST_TRIPLE`、
`HOST_TARGET_CPU` 覆盖。

## 编码字段检查

```bash
llvm-objdump -s -j .text build/k1_inlineasm/kernel.device.o | grep fcffff
llvm-objdump -s -j .text build/k2_plain/kernel.device.o | grep fcffff
```

k1 的 `VF_SIMT` 机器码含 `fcffff15`（code-size 字段 `0xffff`，根因特征值）；
k2 无该模式（code-size `0x0012`，与实际 SIMT body 大小一致）。

## 真机验证结果（2026-08-02，CANN 9.1.T530 + BiSheng clang 15.0.5）

| 序列 | k1_inlineasm | k2_plain | 结论 |
| --- | --- | --- | --- |
| k2 单独 | — | PASS（128/128） | 受害 kernel 自身正确 |
| k1 单独 | PASS（128/128） | — | 污染源自身正确 |
| k1 -> k2 | PASS | **FAIL：507035 vector core exception**（fault kernel=tid_plain_kernel） | 复现跨 program 取指故障 |
| k2 -> k1 | PASS | PASS | k2 的 code-size 正常，不污染后续 |

注：污染触发要求污染源具备足够的 SIMT dispatch 次数（本用例为 64 次/核）；
单次 dispatch 的极简前序不触发。camodel 不保序跨 SIMT VF 的寄存器状态，本用例
需要真机运行。
