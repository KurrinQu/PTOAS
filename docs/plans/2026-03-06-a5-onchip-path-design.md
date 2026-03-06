# A5 片上通路 CV 桥接设计

**日期**: 2026-03-06
**状态**: 待实现
**前置**: CV 分离 V2 设计 (2026-03-05)

---

## 1. 概述

### 1.1 目标

在 A5 架构下，使用片上通路（`tmov(ACC→VEC)`, `tmov(UB→MAT)`）替代 GM workspace 中转，减少跨域数据搬运开销。同时处理 A5 的 1 Cube + 2 Vector 核算力配比：Cube 写出数据后，两个 Vector 核按行方向各读一半。

### 1.2 核心变更

在 `CVInsertBridge::insertBridge()` 中根据 `PTOArch` 分支：

| 架构 | Cube→Vector 桥接 | Vector→Cube 桥接 |
|------|-----------------|-----------------|
| A3 | `tstore(ACC→GM)` + `tload(GM→UB)` | `tstore(UB→GM)` + `tload(GM→MAT)` |
| A5 | `tmov(ACC→VEC)` + 行切分 subview | `tmov(VEC→MAT)` |

### 1.3 硬件基础

- `TMovOp` 是通用数据搬运 intrinsic，pto-isa 底层根据 src/dst 地址空间自动选择硬件指令
- A5 硬件支持 ACC(L0C) → VEC(UB) 和 VEC(UB) → MAT(L1) 的片上直连通路
- A5 每个 Block 有 1 个 Cube 核 + 2 个 Vector 核（Core0/Core1）

---

## 2. 数据流

### 2.1 Cube→Vector（ACC→VEC + 行切分）

**A3 路径（GM 中转）**：
```
Cube section:                          Vector section:
  tmatmul → ACC [H, W]                  sync.wait(PIPE_MTE2, id)
  tstore(ACC→GM, PIPE_FIX)              tload(GM→UB) → 全量 [H, W]
  sync.set(PIPE_FIX, id)                vector 计算(全量)
```

**A5 路径（片上直连 + 行切分）**：
```
Cube section:                          Vector section (每个核):
  tmatmul → ACC [H, W]                  sync.wait(PIPE_V, id)
  tmov(ACC→VEC) → UB [H, W]             sub_id = get_subblock_idx    // 0 or 1
  sync.set(PIPE_FIX, id)                half = subview(UB, [sub_id*H/2, 0], [H/2, W])
  sync.set(PIPE_FIX, id+16)             vector 计算(half)
```

### 2.2 Vector→Cube（VEC→MAT）

**A3**：`tstore(UB→GM, PIPE_MTE3)` → `tload(GM→MAT)`

**A5**：`tmov(VEC→MAT)` → Cube 直接从 MAT 读取

### 2.3 行切分规则

- 按行方向（dim 0）均分为 2 份
- Vector Core0（sub_id=0）取 `[0, 0] ~ [H/2-1, W-1]`
- Vector Core1（sub_id=1）取 `[H/2, 0] ~ [H-1, W-1]`
- **约束：行数 H 必须为偶数**，奇数行时 emit error 并中止 pass

---

## 3. 实现细节

### 3.1 CVInsertBridge::insertBridge() 分支

```cpp
void insertBridge(BridgePoint &bp, Value workspace, unsigned flagId,
                  OpBuilder &builder) {
  bool isCubeProducer = isa<SectionCubeOp>(bp.producerSection);
  auto loc = bp.producerValue.getLoc();

  if (targetArch == PTOArch::A5) {
    insertBridgeA5(bp, flagId, builder);
  } else {
    insertBridgeA3(bp, workspace, flagId, builder);
  }
}
```

### 3.2 A5 桥接实现（Cube→Vector 方向）

```cpp
void insertBridgeA5(BridgePoint &bp, unsigned flagId, OpBuilder &builder) {
  Block &prodBody = bp.producerSection->getRegion(0).front();
  builder.setInsertionPoint(&prodBody, prodBody.end());
  auto loc = bp.producerValue.getLoc();

  // 1. tmov(ACC→VEC): 片上直连，数据写入 UB
  Value dst = bp.consumerUses[0]->get();
  builder.create<TMovOp>(loc, bp.producerValue, dst);

  // 2. sync.set: 通知两个 Vector 核
  bool isCubeProducer = isa<SectionCubeOp>(bp.producerSection);
  auto storePipe = isCubeProducer ? pto::PIPE::PIPE_FIX
                                  : pto::PIPE::PIPE_MTE3;
  auto pipeAttr = PipeAttr::get(builder.getContext(), storePipe);
  builder.create<SyncSetOp>(loc, pipeAttr, flagId);
  if (isCubeProducer) {
    builder.create<SyncSetOp>(loc, pipeAttr, flagId + 16);
  }

  // 3. Consumer section 开头: sync.wait + 行切分
  Block &consBody = bp.consumerSection->getRegion(0).front();
  builder.setInsertionPointToStart(&consBody);

  auto waitPipe = PipeAttr::get(builder.getContext(), pto::PIPE::PIPE_V);
  builder.create<SyncWaitOp>(loc, waitPipe, flagId);

  // 4. 行切分: 奇数行报错
  auto dstType = cast<MemRefType>(dst.getType());
  int64_t totalRows = dstType.getShape()[0];
  if (totalRows % 2 != 0) {
    bp.producerSection->emitError("A5 on-chip path requires even row count, got ")
        << totalRows;
    signalPassFailure();
    return;
  }
  int64_t halfRows = totalRows / 2;
  int64_t totalCols = dstType.getShape()[1];

  // sub_id = get_subblock_idx() → index
  auto subId = builder.create<GetSubBlockIdxOp>(loc, builder.getI64Type());
  auto subIdIdx = builder.create<arith::IndexCastOp>(
      loc, builder.getIndexType(), subId);

  // rowOffset = sub_id * halfRows
  auto halfRowsVal = builder.create<arith::ConstantIndexOp>(loc, halfRows);
  auto rowOffset = builder.create<arith::MulIOp>(loc, subIdIdx, halfRowsVal);

  // halfView = subview dst[rowOffset, 0] [halfRows, totalCols] [1, 1]
  auto zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
  auto colsVal = builder.create<arith::ConstantIndexOp>(loc, totalCols);
  auto oneIdx = builder.create<arith::ConstantIndexOp>(loc, 1);

  auto halfView = builder.create<memref::SubViewOp>(loc, dst,
      /*offsets=*/ValueRange{rowOffset, zeroIdx},
      /*sizes=*/ValueRange{halfRowsVal, colsVal},
      /*strides=*/ValueRange{oneIdx, oneIdx});

  // 5. 替换 consumer 中对 dst 的引用为 halfView
  for (auto *use : bp.consumerUses) {
    use->set(halfView.getResult());
  }
}
```

### 3.3 A3 桥接实现（现有逻辑提取）

```cpp
void insertBridgeA3(BridgePoint &bp, Value workspace, unsigned flagId,
                    OpBuilder &builder) {
  // 现有 GM 中转逻辑不变
  Block &prodBody = bp.producerSection->getRegion(0).front();
  builder.setInsertionPoint(&prodBody, prodBody.end());
  auto loc = bp.producerValue.getLoc();

  builder.create<TStoreOp>(loc, TypeRange{}, bp.producerValue, workspace);

  auto storePipe = isa<SectionCubeOp>(bp.producerSection)
                       ? pto::PIPE::PIPE_FIX : pto::PIPE::PIPE_MTE3;
  auto pipeAttr = PipeAttr::get(builder.getContext(), storePipe);
  builder.create<SyncSetOp>(loc, pipeAttr, flagId);

  Block &consBody = bp.consumerSection->getRegion(0).front();
  builder.setInsertionPointToStart(&consBody);

  auto waitPipe = PipeAttr::get(builder.getContext(), pto::PIPE::PIPE_MTE2);
  builder.create<SyncWaitOp>(loc, waitPipe, flagId);

  Value dst = bp.consumerUses[0]->get();
  builder.create<TLoadOp>(loc, TypeRange{}, workspace, dst);
}
```

### 3.4 TMovOp 分类说明

`tmov(ACC→VEC)` 的 `getPipe()` 返回 `PIPE_V`（fallback），`classifyOp` 会将其分类为 VECTOR。但这不影响正确性，因为：

- `CVClassifyAndSplit` 在 `CVInsertBridge` **之前**运行
- Bridge pass 插入的 `tmov(ACC→VEC)` 直接放在 producer (Cube) section 内部
- 不会再被 classify 处理

### 3.5 Workspace 参数

- A5 模式下不再需要 GM workspace 参数（片上直连）
- `findWorkspaceArg()` 仅在 A3 路径中调用
- 若 A3 路径需要 workspace 但找不到，仍然报错

### 3.6 runOnOperation() 调整

```cpp
void runOnOperation() override {
  func::FuncOp func = getOperation();
  auto bridges = findBridgePoints(func);
  if (bridges.empty())
    return;

  // A3 需要 workspace，A5 不需要
  Value workspace = nullptr;
  if (targetArch != PTOArch::A5) {
    workspace = findWorkspaceArg(func);
    if (!workspace) {
      func.emitError("cross-domain dependency found but no GM workspace");
      return signalPassFailure();
    }
  }

  OpBuilder builder(func.getContext());
  unsigned flagId = 0;
  for (auto &bp : bridges)
    insertBridge(bp, workspace, flagId++, builder);
}
```

---

## 4. 生成 IR 示例

### 4.1 A5 Cube→Vector（ACC→VEC + 行切分）

**输入**：
```mlir
func.func @matmul_then_reduce(
    %left: memref<16x256xf16, #pto.address_space<left>>,
    %right: memref<256x16xf16, #pto.address_space<right>>,
    %acc: memref<16x16xf32, #pto.address_space<acc>>,
    %ub: memref<16x16xf32, #pto.address_space<vec>>,
    %gm_out: memref<16x16xf32, #pto.address_space<gm>>) {
  pto.section.cube {
    pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)
  }
  pto.section.vector {
    // 使用 %acc 的结果（跨域依赖）
    pto.tstore ins(%ub : ...) outs(%gm_out : ...)
  }
  return
}
```

**A5 输出 IR**：
```mlir
pto.section.cube {
  pto.tmatmul ins(%left, %right : ...) outs(%acc : ...)
  pto.tmov ins(%acc : memref<16x16xf32, acc>) outs(%ub : memref<16x16xf32, vec>)
  pto.sync.set #pto.pipe<PIPE_FIX>, 0
  pto.sync.set #pto.pipe<PIPE_FIX>, 16
}
pto.section.vector {
  pto.sync.wait #pto.pipe<PIPE_V>, 0
  %sub_id = pto.get_subblock_idx
  %idx = arith.index_cast %sub_id : i64 to index
  %c8 = arith.constant 8 : index
  %offset = arith.muli %idx, %c8 : index
  %half = memref.subview %ub[%offset, 0] [8, 16] [1, 1]
      : memref<16x16xf32, vec> to memref<8x16xf32, strided<[16,1], offset:?>, vec>
  pto.tstore ins(%half : ...) outs(%gm_out : ...)
}
```

**A5 最终 C++ 输出**：
```cpp
#if defined(__DAV_CUBE__)
  TMATMUL(acc, left, right);
  TMOV(ub, acc);                    // ACC→VEC 片上直连
  set_intra_block(PIPE_FIX, 0);    // → Core0
  set_intra_block(PIPE_FIX, 16);   // → Core1
#endif

#if defined(__DAV_VEC__)
  wait_intra_block(PIPE_V, 0);
  int sub_id = get_subblockid();
  float* half = ub + sub_id * 8 * 16;   // 行切分
  TSTORE(gm_out, half);
#endif
```

---

## 5. 错误处理

| 场景 | 处理 |
|------|------|
| A5 + 奇数行 | `emitError("A5 on-chip path requires even row count")` + `signalPassFailure()` |
| A3 + 无 workspace | `emitError("no GM workspace argument")` + `signalPassFailure()` |
| A5 + 无跨域依赖 | 正常跳过，无操作 |
| A5 + dynamic shape | 暂不支持，仅处理静态 shape |

---

## 6. 测试用例

| 测试文件 | 验证内容 |
|---------|---------|
| `cv_a5_onchip_cube_to_vec.mlir` | A5 ACC→VEC tmov + 行切分 subview，C++ 输出含 `TMOV` + `get_subblockid` |
| `cv_a5_onchip_odd_rows.mlir` | 奇数行报错 |
| `cv_a3_gm_unchanged.mlir` | A3 仍走 GM 路径，输出含 `TSTORE`/`TLOAD` |

---

## 7. 涉及文件

| 文件 | 改动 |
|------|------|
| `lib/PTO/Transforms/CVInsertBridge.cpp` | 核心：arch 分支、A5 tmov + 行切分、A3 提取 |
| `tools/ptoas/ptoas.cpp` | 无改动（arch 已传入） |
| `test/basic/cv_a5_onchip_*.mlir` | 新增测试 |
| `docs/plans/2026-03-06-a5-onchip-path-design.md` | 本文档 |
