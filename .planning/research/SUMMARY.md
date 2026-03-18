# Research Summary

## Stack

Use PTO IR as the semantic source, introduce a new PTO-to-A5VM lowering layer at the current `emitc` backend boundary, model hardware-facing ops in an `a5vm` dialect, and emit textual LLVM HIVM intrinsic IR as the final artifact for v1.

## Table Stakes

- Replace the current `emitc` path without changing overall pass-pipeline behavior.
- Preserve PTO interface semantics and template-driven behavior for the `Abs` path.
- Define legal 256-byte `a5vm` vector types and the minimum ops needed for `TLOAD`, `TABS`, and `TSTORE`.
- Emit textual LLVM HIVM intrinsic IR and derive the required intrinsic inventory from the implemented sample path.

## Suggested Architecture

`PTO -> PTO semantic lowering helpers -> A5VM dialect -> textual LLVM HIVM emission`

The PTO lowering layer is where template-like dispatch behavior must be preserved. `a5vm` should stay hardware-facing and explicit about vector legality. The final textual emitter should be isolated so it can later be replaced by real LLVM intrinsic ops.

## Watch Out For

- Do not collapse PTO template behavior into a hardcoded one-off `Abs` emission path.
- Do not broaden v1 beyond the `Abs` acceptance chain.
- Do not allow arbitrary vector widths; `a5vm` must enforce fixed 256-byte vectors.
- Do not guess a large HIVM intrinsic catalog before the implemented sample path reveals what is actually needed.

## Local Evidence Used

- `test/samples/Abs/abs.py`
- `lib/PTO/Transforms/PTOToEmitC.cpp`
- `docs/PTO_IR_manual.md`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/common/pto_instr.hpp`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TUnaryOp.hpp`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TLoad.hpp`
- `/usr/local/Ascend/cann-8.5.0/aarch64-linux/include/pto/npu/a2a3/TStore.hpp`
- `/usr/local/Ascend/cann-8.5.0/tools/bisheng_compiler/lib/clang/15.0.5/include/__clang_cce_vector_intrinsics.h`
