# CV Separation Pass 修正设计

日期: 2026-03-05
基于: `信息补充.md` 反馈

## 背景

首版 CV 分离 pass 实现后，根据硬件数据通路补充信息，发现以下问题需要修正。

## 修正项

### 1. classifyOp() 补充 Cube 侧地址空间

**问题**: `cubeSpaces` 缺少 MAT(L1)、BIAS、SCALING，导致访问这些地址空间的 op 被错误归类为 SHARED。

**修复**:
```cpp
static const pto::AddressSpace cubeSpaces[] = {
    pto::AddressSpace::LEFT, pto::AddressSpace::RIGHT,
    pto::AddressSpace::ACC, pto::AddressSpace::MAT,
    pto::AddressSpace::BIAS, pto::AddressSpace::SCALING};
```

### 2. TMovOp 改为地址空间分类

**问题**: TMOV 是灵活指令，MAT→LEFT/RIGHT 属于 CUBE 侧，VEC→VEC 属于 VECTOR 侧，不能按 op 类型硬编码。

**修复**: 从 VECTOR op 类型列表中移除 `TMovOp`，让它走地址空间匹配逻辑：
- 操作数含 MAT/LEFT/RIGHT → CUBE
- 操作数含 VEC → VECTOR

### 3. Bridge sync pipe 按方向选择

**问题**: `insertBridge()` 固定用 `PIPE_MTE3`，但 ACC→GM 应使用 `PIPE_FIX`。

**修复**:
- Producer 是 SectionCubeOp → sync.set 用 `PIPE_FIX` (ACC→GM)
- Producer 是 SectionVectorOp → sync.set 用 `PIPE_MTE3` (UB→GM)
- sync.wait 统一用 `PIPE_MTE2`

### 4. ptoas.cpp pipeline 集成

**问题**: 当前 CV 模式跳过下游 pass 并 dump IR，不符合预期。

**修复**: `--enable-cv-separation` 开启时，仅在原有 pipeline 前插入 CV 两个 pass，下游 pass 继续执行。

### 5. 测试用例更新

更新测试用例反映真实数据通路：
- Cube 完整路径: GM → MAT → LEFT/RIGHT (via TMOV) → matmul → ACC
- TMOV 分类: MAT→LEFT 归 CUBE, VEC→VEC 归 VECTOR
- Bridge 方向: Cube→Vec 用 PIPE_FIX, Vec→Cube 用 MTE3

## 数据通路参考

| 方向 | 路径 | Pipeline |
|------|------|----------|
| GM → UB (VEC) | MTE2 | PIPE_MTE2 |
| GM → MAT (L1) | MTE2 | PIPE_MTE2 |
| MAT → LEFT/RIGHT | TMOV/MTE1 | - |
| CUBE 计算 | matmul | PIPE_M |
| ACC → GM | - | PIPE_FIX |
| UB → GM | - | PIPE_MTE3 |
| VECTOR 计算 | add/trans 等 | PIPE_V |

## 涉及文件

- `lib/PTO/Transforms/CVClassifyAndSplit.cpp`
- `lib/PTO/Transforms/CVInsertBridge.cpp`
- `tools/ptoas/ptoas.cpp`
- `test/basic/cv_*.mlir`
