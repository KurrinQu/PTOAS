# Proposal: 泛化 Level-3 OP-Lib 模板能力以支持多 Family Lowering

## 概述

本变更负责把当前只面向 binary float elementwise 的 Level-3 OP-Lib 基础设施，泛化成可承载 4.4-4.8 多 family lowering 的底座。

这是后续三个 feature change 的共同前置：

1. `expand-vector-op-oplib-lowering`
2. `expand-reduction-broadcast-oplib-lowering`
3. `expand-compare-bitwise-oplib-lowering`

## 背景与动机

当前 OP-Lib 体系存在四个硬限制：

1. `pto.oplib.kind` 实际上只支持 `l3_binary_elementwise_template`
2. 模板签名被固定为 `(!pto.tile_buf, !pto.tile_buf, !pto.tile_buf) -> ()`
3. 匹配元数据只支持单套 `rows/cols/blayout/slayout/fractal`
4. A5 OP-Lib vector EmitC 只支持 float load/store + float binary arith

这使得下列 family 无法接入：

1. `(tile, scalar, dst)`、`(tile, dst)`、`(scalar, dst)` 等非 3-tile ABI
2. `src/tmp/dst` 的 reduction family
3. `mask/src0/src1/dst` 的 compare/select family
4. integer vector 和 vector reduction / vector compare / transcendental family

## 目标

1. 将 OP-Lib 注册、匹配、实例化和 inline 机制推广到多 family 模型。
2. 引入按参数编号建模的匹配元数据，摆脱 binary-only 假设。
3. 扩展模板体 IR 白名单和 A5 OP-Lib vector EmitC 能力，使 4.4-4.8 所需 family 可表达、可导入、可 codegen。
4. 统一模板源码目录，停止维护 `test/tile_fusion/oplib/` 的重复模板。

## 非目标

1. 本 change 不直接补齐 4.4-4.8 的业务 family 模板和全量用例。
2. 不为其他架构引入通用 OP-Lib 支持。
3. 不引入任意 `math.*` 或任意 dialect；只放开后续 change 所需的最小集合。

## 成功标准

1. OP-Lib 可以按 `kind` 校验并导入多种固定签名模板。
2. `argN.*`、`scalar_pos`、`cmp_mode`、`is_binary` 等元数据可被 matcher 使用。
3. A5 OP-Lib vector lowering 能接受后续 change 需要的 float unary、math unary、vector compare/select、vector reduction、integer vector family。
4. lit 测试切到 `oplib/level3/` 作为唯一模板源。
