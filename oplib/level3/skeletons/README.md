# Level-3 Skeleton Sources

本目录承载 `oplib/level3` 的单一维护源，不直接作为 `ptoas --op-lib-dir`
的 importer 输入。

## 布局

- `../families/a5_oplib_v1_family_dsl.json`
  - A5 OpLib V1 的声明式 Family DSL，覆盖 family、参数角色、dtype 轴、variant 轴、
    metadata、matcher key 和 snippet contract 绑定。
- `../families/a5_oplib_v1_snippet_contracts.json`
  - Mixed-Body MLIR snippet 合同真值层，固定 snippet 可见 SSA 名称、
    结果 SSA 约定，以及由生成器统一负责的外围 scaffolding。
- `../families/snippets/`
  - 作者维护的 Mixed-Body MLIR snippet 源；生成器按合同把它们注入
    skeleton 的 `@@EXTRA_SETUP@@` / `@@COMPUTE@@` 插槽。
- `../family_dsl.py`
  - Family DSL / snippet contract 的 loader / validator，以及到 `catalog.json`
    的投影检查。
- `catalog.json`
  - 定义主要计算模式的 family、参数维度和 concrete 展开矩阵。
- `module.tmpl.mlir`
  - 所有 concrete 输出共用的模块级 wrapper。
- `*.instance.tmpl.mlir`
  - 各计算模式的单实例 skeleton。

## 输出位置

- skeleton source：`oplib/level3/skeletons/`
- importer-active concrete 模板：`oplib/level3/*.mlir`

当前 `oplib/level3` 已切换为由 skeleton source 统一生成 importer-active concrete
模板：

1. skeleton source 只在 `skeletons/` 维护一份。
2. family 维度与 matcher 维度在 `families/a5_oplib_v1_family_dsl.json` 维护。
3. op/variant 的核心 vector 计算体在 `families/snippets/` 维护。
4. `dtype`、condition、core op、variant 等差异通过生成脚本展开到根目录 concrete 模板。
5. lowering/importer 继续只读取根目录 `oplib/level3/*.mlir`。

## 生成入口

使用相邻脚本：

```bash
python3 oplib/level3/family_dsl.py --check-snippet-contracts
python3 oplib/level3/family_dsl.py --check-catalog
python3 oplib/level3/generate_level3_templates.py --write
python3 oplib/level3/generate_level3_templates.py --check
```

`--write` 会刷新根目录 `oplib/level3/*.mlir` concrete 模板，`--check` 会检查
catalog / template 与已落盘 concrete 模板是否漂移，并报告 root 目录下多余的
stale generated `.mlir` 文件。`--write` 会把这类 stale generated 文件一并清理，
使根目录重新收敛到期望输出集合。生成器启动时还会强制校验 Family DSL、
snippet contract、snippet 源与 `catalog.json` 是否保持同步。
