# tilelang-dsl-vpto-lowering Specification

## Purpose
TBD - created by archiving change add-tilelang-dsl-authoring-vpto-lowering. Update Purpose after archive.
## Requirements
### Requirement: TileLang DSL v1 MUST infer dedicated `pto.vecscope` for stable vector chains and reserve `pto.strict_vecscope` for advanced mode

TileLang DSL v1 中，stable vector surface 在用户省略显式 scope 时 MUST 通过 frontend 自动推断 dedicated `pto.vecscope`。  
当用户启用 `advanced=True` 并显式书写 `with pto.strict_vecscope(...) as (...):` 时，frontend MUST 保留该 surface 并直接 lower 为 dedicated `pto.strict_vecscope` authoring-form VPTO carrier。  
`strict_vecscope` MUST NOT 在非 advanced kernel 中出现。

#### Scenario: stable vector chain lowers through inferred `pto.vecscope`

- **WHEN** 用户在非 advanced kernel 中连续书写 `make_mask`、`vlds`、vector ALU、`vsts` 等 stable vector chain
- **THEN** frontend MUST 为该 chain 生成 dedicated `pto.vecscope`
- **AND** 生成结果 MUST 满足当前 authoring-form VPTO legality contract

#### Scenario: explicit `strict_vecscope` is preserved only in advanced mode

- **WHEN** 用户在 `advanced=True` kernel 中显式书写 `with pto.strict_vecscope(...) as (...):`
- **THEN** lowering 结果 MUST 生成对应的 `pto.strict_vecscope`
- **AND** region argument、capture operand 和 block argument 类型 MUST 与 DSL surface 中的显式 capture 一一对应

#### Scenario: non-advanced kernel rejects explicit `strict_vecscope`

- **WHEN** 用户在未启用 `advanced=True` 的 kernel 中书写 `with pto.strict_vecscope(...) as (...):`
- **THEN** frontend MUST 在生成 VPTO IR 之前报错
- **AND** 诊断 MUST 明确指出 `strict_vecscope` requires `advanced=True`

### Requirement: TileLang DSL v1 MUST support the fixed elementwise lowering profile

TileLang DSL v1 lowering MUST 支持以下固定 support matrix：

- 2D `TensorView`
- 1D/2D `Tile`
- `dma_load`
- `dma_store`
- `make_mask(dtype, PAT.*)` / `make_mask(dtype, remaining)`
- `vlds`
- `vsts`
- unary：`vabs`, `vrelu`, `vexp`, `vnot`
- binary：`vadd`, `vsub`, `vmul`, `vdiv`, `vmax`, `vmin`, `vand`, `vor`, `vxor`
- vector-scalar：`vadds`, `vsubs`, `vmuls`, `vdivs`, `vmaxs`, `vmins`
- `for range(lb, ub, step)`
- `if/else`
- `set_flag`, `wait_flag`, `pipe_barrier`

support matrix 外的 surface MUST 在 frontend reject。

#### Scenario: representative elementwise kernel lowers to authoring-form VPTO

- **WHEN** 用户编写由 `TensorView`、`Tile`、高层 DMA、typed mask、elementwise vector op、`for`、`if` 和基础 sync 组成的 kernel
- **THEN** frontend MUST 产出只包含 `func.func`、`arith`、`scf` 和合法 `pto.*` authoring surface 的 VPTO IR
- **AND** 该 IR MUST 不依赖 matcher、implicit vecscope inference 或 advanced family 才能成立

#### Scenario: unsupported advanced family is rejected in v1

- **WHEN** 用户在 v1 kernel 中使用 compare/select/reduction/rearrangement、UB-to-UB copy 或其他不在 support matrix 内的 family
- **THEN** frontend MUST 直接报错
- **AND** MUST NOT 静默降级为其他 family 或生成半合法 VPTO IR

### Requirement: TileLang DSL v1 MUST support static physical Tile shape with dynamic TensorView views and loop bounds

TileLang DSL v1 中，Tile physical shape MUST 是静态编译期常量。  
TensorView shape、slice 边界、loop bound 和 tail 相关 remaining value MAY 包含 runtime value。  
`valid_shape` 仅可使用静态值或由 TensorView partition 直接推导。

#### Scenario: dynamic TensorView slice and tail mask lower successfully

- **WHEN** 用户使用 dynamic TensorView slice、dynamic loop bound，并在 loop 中通过 `make_mask(dtype, remaining)` 处理尾块
- **THEN** frontend MUST 生成合法的 authoring-form VPTO IR
- **AND** tail mask MUST lower 为与元素类型匹配的 typed predicate family
- **AND** Tile physical shape MUST 继续保持静态契约

### Requirement: `dma_load` and `dma_store` MUST lower to VPTO DMA programming plus copy ops

TileLang DSL 的高层 `dma_load` / `dma_store` MUST 在 frontend lower 到当前合法 VPTO authoring surface：

- GM -> UB：必要的 `set_loop*_stride_outtoub` / `set_loop_size_outtoub` + `copy_gm_to_ubuf`
- UB -> GM：必要的 `set_loop*_stride_ubtoout` / `set_loop_size_ubtoout` + `copy_ubuf_to_gm`

参数 MUST 由 TensorView slice、Tile shape/config 和 padding mode 推导。

#### Scenario: high-level DMA becomes legal VPTO copy programming

- **WHEN** 用户在 DSL 中编写 `dma_load(input_tensor[slice], ub_tile)` 或 `dma_store(ub_tile, output_tensor[slice])`
- **THEN** lowering MUST 显式生成对应的 DMA programming op 和 copy op
- **AND** 生成结果 MUST 符合当前 VPTO copy-family 的 authoring contract

### Requirement: `verify()` MUST validate generated IR through the repo VPTO authoring-stage legality path

TileLang DSL descriptor 的 `verify()` MUST 以 repo 当前 `ptoas` legality 路径验证生成结果。  
当可用的 `ptoas` binary 缺失、不可执行或环境不完整时，`verify()` MUST 返回结构化 `verifier unavailable` 结果，而不是静默通过。

#### Scenario: generated VPTO module is checked by `ptoas`

- **WHEN** 用户对一个已 specialization 的 kernel 调用 `verify()`
- **THEN** implementation MUST 使用与 `ptoas --pto-backend=vpto` 一致的 authoring-stage legality contract 对生成 module 进行校验
- **AND** 成功结果 MUST 代表 generated IR 已通过当前 repo 的 VPTO authoring legality

#### Scenario: verifier-unavailable is reported explicitly

- **WHEN** `verify()` 无法找到或执行 `ptoas` binary
- **THEN** implementation MUST 返回结构化 `verifier unavailable` 结果
- **AND** MUST NOT 把“未验证”误报成“验证通过”
