# PTOAS 内存一致性设计

本文说明 PTOAS 当前暴露的内存一致性 IR，以及为什么暂时移除自动
MemoryConsistency 分析 pass。

## 当前状态

PTOAS 目前保留显式内存一致性 IR 和 EmitC lowering：

| PTO IR | 语义 | EmitC lowering |
| --- | --- | --- |
| `pto.cmo.cacheinvalid all #pto.address_space<gm>` | whole-cache GM cache maintenance | `dcci((__gm__ void*)0, cache_line_t::ENTIRE_DATA_CACHE)` |
| `pto.cmo.cacheinvalid %addr single_cache_line : !pto.ptr<T, gm>` | 指定地址所在 cache line 的 GM cache maintenance | `dcci((__gm__ void*)addr, cache_line_t::SINGLE_CACHE_LINE)` |
| `pto.fence.barrier_all #pto.fence_scope<gm>` | GM visibility fence | `dsb(DSB_DDR)` |

`pto.cmo.cacheinvalid` 和 `pto.fence.barrier_all` 由 PyPTO 或用户显式插入。
PTOAS 不再通过独立 `pto-memory-consistency` pass 自动扫描 signal/payload
关系、插入 pipe drain、消除 marker-only CMO，或诊断缺失的 CMO/fence。

## 为什么移除自动分析 pass

issue #950 暴露了 `pto-memory-consistency` 在复杂单 block 控制流中的编译时
间退化。该 pass 的 state `merge()` 直接追加状态向量；处理单块 `scf.if` 时，
子 region 返回值已经包含入口状态，父层又把它合并回原状态，导致每经过一个
`if`，历史状态近似翻倍。

issue #950 的 repro 中包含约 80 个连续 `scf.if`、160 次 GM scalar load、163 次
GM scalar store，因此状态规模接近指数增长，触发 ptoas 0.50 相比 0.48 的确定性
编译超时。

在重新设计 bounded、去重的数据流模型之前，默认 pipeline 不应继续运行这个
自动分析 pass。当前策略是先回到显式 IR 模式，保证编译时间可控。

## 使用要求

PyPTO 或用户需要显式表达内存一致性动作：

```mlir
// cacheable scalar GM store 发布 payload
pto.store_scalar %value, %payload[%idx] : !pto.ptr<i32, gm>, i32
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
pto.fence.barrier_all #pto.fence_scope<gm>
pto.comm.tnotify ...
```

```mlir
// TWait 后读取 cacheable scalar GM payload
pto.comm.twait ...
pto.cmo.cacheinvalid %payload single_cache_line : !pto.ptr<i32, gm>
%value = pto.load_scalar %payload[%idx] : !pto.ptr<i32, gm> -> i32
```

如果涉及 non-cacheable MTE/FIX/comm macro payload，当前 PTOAS 不再自动根据
`cmo.cacheinvalid` marker 补 pipe drain。需要由上游显式生成必要的
`pto.barrier #pto.pipe<...>`，并确保 pipe drain 发生在 `fence.barrier_all` 前。

## 后续重新引入自动分析的要求

后续若恢复 MemoryConsistency pass，需要先满足以下条件：

- state merge 必须 bounded，不能通过简单追加导致指数增长。
- state key 至少应按 payload identity、pipe、cacheability 和 action kind 去重。
- `scf.if` 的 region exit state 不能重复携带父层入口 state。
- loop 和多 block region 需要明确固定点或保守 summary 上界。
- 新 pass 需要包含 issue #950 规模的 compile-time 回归测试。
