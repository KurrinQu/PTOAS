## 1. OpenSpec 契约落定

- [ ] 1.1 新增 `openspec/changes/extend-a5-oplib-fusion-pipeline/specs/oplib-transforms/spec.md`，定义 mixed chain 分组、vec-scope-aware low-level fusion 和链内中间访存消除契约。
- [ ] 1.2 新增 `openspec/changes/extend-a5-oplib-fusion-pipeline/specs/oplib-lowering/spec.md`，补充 grouped lowering 与 single-op lowering 的能力一致性要求。
- [ ] 1.3 在 `proposal.md` 和 `design.md` 中明确本 change 只覆盖 6 个 tile-tile binary op 与 6 个 tile-scalar op，不扩展到 ternary/unary/reduction/broadcast。

## 2. 分组与 outline 扩展

- [x] 2.1 修改 `lib/PTO/Transforms/PTOCreateFusionGroups.cpp`，把 binary-only 分组规则改为基于 `OpLibOpInterface` / `OpLibMatchDescriptor` 的 descriptor-driven grouping。
- [x] 2.2 让 mixed chain 支持 tile-tile 与 tile-scalar 混合分组，并把 scalar external input 纳入 group interface。
- [x] 2.3 修改 `lib/PTO/Transforms/PTOOutlineFusionGroups.cpp`，从固定 `(src0, src1, dst)` helper ABI 升级为“唯一 external operands + destination tiles”模型。
- [x] 2.4 保证 outline 后 helper 克隆保留 `tdivs` 的 `operandOrder` 与其他 lowering 相关 attrs。

## 3. Grouped lowering 与 low-level fusion

- [x] 3.1 调整 grouped lowering path，使其对上述 12 个 op 的支持范围与 single-op path 保持一致。
- [x] 3.2 保证 mixed chain grouped lowering 继续复用现有 `OpLibOpInterface`，不引入第二套 matcher 协议。
- [x] 3.3 修改 `lib/PTO/Transforms/TileFusion/PTOLowLevelLoopFusion.cpp`，把匹配入口从裸相邻 `scf.for` 升级为相邻 `pto.simd.vec_scope` stage。
- [x] 3.4 在 low-level fusion 重写过程中完成 store-to-load forwarding，并删除仅供后继 stage 消费的中间 `vector.maskedstore`。
- [x] 3.5 对不满足 canonical shape 的 helper 保持保守 no-op，不做错误合并。

## 4. 回归与验证

- [x] 4.1 更新 `test/tile_fusion/create_fusion_groups.mlir`，覆盖 tile-scalar 和 mixed chain 标组。
- [x] 4.2 更新 `test/tile_fusion/materialize_fusion_groups.mlir`，覆盖 mixed chain helper 参数包含 scalar。
- [x] 4.3 重写 `test/tile_fusion/low_level_loop_fusion.mlir`，显式检查单一 `pto.simd.vec_scope`、无链内中间 `vector.maskedstore` / 回读 `vector.maskedload`。
- [x] 4.4 新增 1 个 `test/tile_fusion/` negative 用例，验证非法 case 保守不融合。
- [x] 4.5 更新或新增 `test/oplib/` 用例，使用 `softmax_chain.pto` 验证 mixed chain，使用 `binary_max_min_chain.pto` 验证纯 tile-tile 对照。
- [x] 4.6 新增 1 个 EmitC/codegen smoke，检查输出中保留 `__VEC_SCOPE__` 且没有链内中间 tile 的多余 round-trip 访存模式。

## 5. 验证命令

- [x] 5.1 运行 `../llvm-project/build-shared/bin/llvm-lit -sv test/tile_fusion/...`
- [x] 5.2 运行 `../llvm-project/build-shared/bin/llvm-lit -sv test/oplib/<相关用例>`
- [x] 5.3 运行至少 1 个 `ptoas ... --enable-op-fusion -o %t.cpp` 的直出 C++ smoke，并记录结果
