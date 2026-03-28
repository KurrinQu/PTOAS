## 1. OpenSpec 契约落定

- [x] 1.1 新增 `openspec/changes/add-a5vm-memref-vld-vst/specs/a5vm-vld-vst-addressing/spec.md`，定义 `vld*/vst*` 双地址形态、memref-first 主线和 stateful 指针边界。
- [x] 1.2 新增 `openspec/changes/add-a5vm-memref-vld-vst/specs/a5vm-backend-pipeline/spec.md`，补充 A5 backend 主线地址模型与发射边界职责。
- [x] 1.3 在 `proposal.md` 与 `design.md` 中明确非目标：不扩展 `copy_* / gather / scatter`，不定义 stateful memref `base_out` 语义。

## 2. IR 接口与 verifier 收口

- [x] 2.1 在 `include/PTO/IR/A5VMOps.td` 为 `vld*/vst*`（含 predicate 无状态变体）引入 buffer-like 地址类型约束（`memref` 或 `!llvm.ptr`）。
- [x] 2.2 保持 `pstu/vstu/vstus/vstur` 的 `base/base_out` 为 pointer-only 约束，不放宽为 memref。
- [x] 2.3 在 `lib/PTO/IR/A5VM.cpp` 对应 verifier 中对 stateful `base/base_out` 增加显式 pointer-only 校验，避免语义漂移。
- [x] 2.4 校准相关错误信息，确保能区分“双形态允许”与“stateful 指针限定”两类失败路径。

## 3. Backend 与发射边界

- [x] 3.1 校验 `PTOToA5VM -> PTOLowLevelLoopFusion -> PTOFlattenFusionRegion` 阶段地址模型默认行为，确保主线保留 memref 语义。
- [x] 3.2 在 A5VM text 发射路径确认 memref 地址可以稳定映射为 pointer ABI，不引入行为分叉。
- [x] 3.3 在 A5VM LLVM 发射路径确认 memref 地址同样完成 pointer ABI 对接，且与 ptr 形态等价。
- [x] 3.4 对发射边界增加必要的 guard/diagnostic，避免未收口的 memref 直接泄漏到 intrinsic ABI。

## 4. 测试与文档

- [x] 4.1 补充 `test/phase1` 或对应 A5VM verifier 用例：`vld*/vst*` memref 正例。
- [x] 4.2 补充对应 ptr 正例，验证向后兼容。
- [x] 4.3 新增 stateful 负例：`base/base_out` 使用 memref 时按契约失败。
- [x] 4.4 增加至少 1 组 memref/ptr 对照发射用例，验证 A5VM text/LLVM 发射语义一致。
- [x] 4.5 更新 `docs/vpto-spec.md` 中 `vld/vst` 相关语法与语义说明，明确“编译器推荐 memref，低层编程可用 ptr”。

## 5. 验证命令

- [x] 5.1 运行最小相关 lit：`python3 -m lit -sv test/phase1/<a5vm相关用例>`。
- [x] 5.2 运行 A5 backend 流水线相关用例：`python3 -m lit -sv test/phase2/<a5vm相关用例>`。
- [x] 5.3 运行至少一组 A5VM 发射验证（text/llvm 之一）并记录 memref/ptr 对照结果。

### 5.3 验证记录（2026-03-26）

- 输入：`test/phase2/a5vm_vld_vst_emit_memref_ptr_equivalence.mlir`
- 命令：
  `ptoas --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-text ...`
  `ptoas --pto-arch=a5 --pto-backend=a5vm --a5vm-emit-hivm-llvm ...`
- 结果：
  `text` 输出中 `memref_form/ptr_form` 均调用 `llvm.hivm.vldsx1` 与 `llvm.hivm.vstsx1`，地址均为 `ptr addrspace(6)`。
  `llvm` 输出中 `ptr_form` 直接调用相同 intrinsic；`memref_form` 经 `ptrtoint/inttoptr` 桥接后调用同一 intrinsic，语义对齐。
