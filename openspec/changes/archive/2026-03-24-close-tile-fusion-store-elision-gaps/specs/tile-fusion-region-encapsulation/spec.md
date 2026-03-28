# tile-fusion-region-encapsulation Specification

## ADDED Requirements

### Requirement: pto.yield frontier MUST preserve stable yielded order without auxiliary frontier-class metadata

`PTOFusionRegionGenPass` 生成 `pto.yield` / `pto.fusion_region` results 时，MUST 仅通过 yielded value 列表表达 region 的对外可见 frontier。该列表 MUST 保持稳定顺序，且 MUST NOT 再并行生成额外的 frontier-class metadata。

#### Scenario: Yielded frontier follows stable span order

- **WHEN** 一个 `pto.fusion_region` 需要为多个仍对 region 外可见的 tile 生成 result / yield
- **THEN** `PTOFusionRegionGenPass` MUST 按 fused span 中稳定的枚举顺序生成 `pto.yield` operand 与 `pto.fusion_region` result
- **AND** 下游 store-elision MUST 仅通过该显式 yielded frontier 判断“哪些值仍对 region 外可见”

#### Scenario: Internal-dead values never receive frontier slots

- **WHEN** 某个 region 内定义的 tile 在封装后不再对 region 外可见
- **THEN** 它 MUST NOT 出现在 `pto.yield` 中
- **AND** MUST NOT 在 yielded frontier 中占位或生成伪 result
