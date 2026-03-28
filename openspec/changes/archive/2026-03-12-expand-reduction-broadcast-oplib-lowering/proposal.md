# Proposal: 扩展 4.5-4.6 Reduction 与 Broadcast 的 OP-Lib Lowering

## 概述

本变更覆盖 `docs/PTO_IR_manual.md` 第 4.5 节和第 4.6 节，共 12 个 op：

1. Reduction：`trowsum`、`trowmax`、`trowmin`、`tcolsum`、`tcolmax`、`tcolmin`
2. Broadcast：`trowexpand`、`tcolexpand`、`trowexpandmul`、`trowexpanddiv`、`trowexpandsub`、`texpands`

本 change 依赖 `generalize-oplib-template-capabilities` 先提供多 family ABI、模板体 reduction 能力和 directory 统一。

## 背景与动机

4.5-4.6 与 4.4 的差异不是“多几个 elementwise op”，而是：

1. 输出形状不再总是与输入同形
2. reduction family 中存在 `tmp`
3. `tcolsum` 带 `isBinary`
4. `texpands` 是 `(scalar, dst)` 签名

这些都不能继续依赖当前 binary-only OP-Lib 模型。

## 目标

1. 让 4.5-4.6 的 12 个 op 全部进入 OP-Lib lowering。
2. 对共享 skeleton 的 reduction / broadcast op 进行 family 归并。
3. 修正文档中与实际 IR 不一致的 reduction `tmp` 描述。

## 非目标

1. 不覆盖 4.4 的 vector arithmetic。
2. 不覆盖 4.7-4.8 的 compare / bitwise。
3. 不在本 change 内引入特化 reduction kernel。

## 成功标准

1. 12 个 op 全部具备静态 valid shape 与动态 valid shape 的 IR 用例。
2. `tcolsum` 覆盖 `isBinary=false/true`。
3. `trowsum`、`trowmax`、`trowmin` 的 manual 文档与实际 IR 参数对齐。
