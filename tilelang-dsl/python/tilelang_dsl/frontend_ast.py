"""Frontend AST nodes for TileLang DSL descriptor materialization."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from .support_matrix import (
    ADVANCED_EXPR_PTO_CALLS,
    ADVANCED_TOPLEVEL_PTO_CALLS,
    ADVANCED_VECSCOPE_PTO_CALLS,
    DEFERRED_PTO_SURFACES,
    SUPPORTED_TOPLEVEL_PTO_CALLS,
    SUPPORTED_VECSCOPE_PTO_CALLS,
    advanced_mode_message,
    deferred_surface_message,
)


@dataclass(frozen=True)
class FrontendParameterNode:
    name: str
    kind: str
    annotation: Any
    dtype: Any


@dataclass(frozen=True)
class FrontendTileSpecializationNode:
    name: str
    shape: tuple[int, ...]
    memory_space: str
    config: Any
    valid_shape: tuple[int | None, ...] | None


class FrontendExprNode:
    """Base class for lowered frontend expressions."""


@dataclass(frozen=True)
class FrontendNameExpr(FrontendExprNode):
    name: str


@dataclass(frozen=True)
class FrontendConstantExpr(FrontendExprNode):
    value: Any


@dataclass(frozen=True)
class FrontendSymbolExpr(FrontendExprNode):
    namespace: str
    name: str


@dataclass(frozen=True)
class FrontendSliceExpr(FrontendExprNode):
    start: FrontendExprNode | None
    stop: FrontendExprNode | None
    step: FrontendExprNode | None


@dataclass(frozen=True)
class FrontendTupleExpr(FrontendExprNode):
    elements: tuple[FrontendExprNode, ...]


@dataclass(frozen=True)
class FrontendAttributeExpr(FrontendExprNode):
    base: FrontendExprNode
    attr: str


@dataclass(frozen=True)
class FrontendSubscriptExpr(FrontendExprNode):
    base: FrontendExprNode
    index: FrontendExprNode


@dataclass(frozen=True)
class FrontendBinaryExpr(FrontendExprNode):
    lhs: FrontendExprNode
    op: str
    rhs: FrontendExprNode


@dataclass(frozen=True)
class FrontendCallExpr(FrontendExprNode):
    namespace: str | None
    name: str
    args: tuple[FrontendExprNode, ...]


class FrontendTargetNode:
    """Base class for assignment targets."""


@dataclass(frozen=True)
class FrontendNameTarget(FrontendTargetNode):
    name: str


@dataclass(frozen=True)
class FrontendTupleTarget(FrontendTargetNode):
    elements: tuple[FrontendNameTarget, ...]


class FrontendStmtNode:
    """Base class for lowered frontend statements."""


@dataclass(frozen=True)
class FrontendAssignStmt(FrontendStmtNode):
    target: FrontendTargetNode
    value: FrontendExprNode
    annotation: Any | None = None


@dataclass(frozen=True)
class FrontendExprStmt(FrontendStmtNode):
    expr: FrontendExprNode


@dataclass(frozen=True)
class FrontendReturnStmt(FrontendStmtNode):
    value: FrontendExprNode | None


@dataclass(frozen=True)
class FrontendForStmt(FrontendStmtNode):
    target: str
    lower_bound: FrontendExprNode
    upper_bound: FrontendExprNode
    step: FrontendExprNode
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendIfStmt(FrontendStmtNode):
    condition: FrontendExprNode
    then_body: tuple[FrontendStmtNode, ...]
    else_body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendStrictVecscopeStmt(FrontendStmtNode):
    captures: tuple[FrontendExprNode, ...]
    block_arguments: tuple[str, ...]
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class FrontendKernelNode:
    target: str
    op: str
    name: str
    verify_enabled: bool
    advanced_enabled: bool
    dtype_signature: tuple[Any, ...]
    parameters: tuple[FrontendParameterNode, ...]
    tile_specializations: tuple[FrontendTileSpecializationNode, ...]
    body: tuple[FrontendStmtNode, ...]


@dataclass(frozen=True)
class _FrontendBuildContext:
    source_info: Any
    templates: dict[str, dict[str, str]]
    selected_op: str | None
    advanced_enabled: bool
    vecscope_depth: int = 0

    def error(self, node: ast.AST, message: str) -> Exception:
        if self.source_info is not None:
            return self.source_info.error(node, message)
        return ValueError(message)

    def nested_vecscope(self) -> "_FrontendBuildContext":
        return _FrontendBuildContext(
            source_info=self.source_info,
            templates=self.templates,
            selected_op=self.selected_op,
            advanced_enabled=self.advanced_enabled,
            vecscope_depth=self.vecscope_depth + 1,
        )


_BINARY_OP_NAMES = {
    ast.Add: "add",
    ast.Sub: "sub",
    ast.Mult: "mul",
    ast.FloorDiv: "floordiv",
}


def _attribute_path(node: ast.AST) -> tuple[str, ...] | None:
    if isinstance(node, ast.Name):
        return (node.id,)
    if isinstance(node, ast.Attribute):
        base_path = _attribute_path(node.value)
        if base_path is None:
            return None
        return base_path + (node.attr,)
    return None


def _validate_resolved_template_op_surface(
    op_name: str,
    node: ast.AST,
    context: _FrontendBuildContext,
) -> None:
    if op_name in SUPPORTED_TOPLEVEL_PTO_CALLS:
        return
    if op_name in SUPPORTED_VECSCOPE_PTO_CALLS:
        return
    if op_name in ADVANCED_VECSCOPE_PTO_CALLS:
        if context.advanced_enabled:
            return
        raise context.error(
            node,
            advanced_mode_message(op_name),
        )
    if op_name in ADVANCED_EXPR_PTO_CALLS or op_name in ADVANCED_TOPLEVEL_PTO_CALLS:
        if context.advanced_enabled:
            return
        raise context.error(
            node,
            advanced_mode_message(op_name),
        )
    if op_name in DEFERRED_PTO_SURFACES:
        raise context.error(
            node,
            deferred_surface_message(op_name),
        )
    raise context.error(
        node,
        f"unsupported op surface `pto.{op_name}` in TileLang DSL v1",
    )


def _build_expr(node: ast.AST, context: _FrontendBuildContext) -> FrontendExprNode:
    if isinstance(node, ast.Name):
        return FrontendNameExpr(name=node.id)
    if isinstance(node, ast.Constant):
        return FrontendConstantExpr(value=node.value)
    if isinstance(node, ast.Slice):
        start = None if node.lower is None else _build_expr(node.lower, context)
        stop = None if node.upper is None else _build_expr(node.upper, context)
        step = None if node.step is None else _build_expr(node.step, context)
        return FrontendSliceExpr(start=start, stop=stop, step=step)
    if isinstance(node, ast.Tuple):
        return FrontendTupleExpr(
            elements=tuple(_build_expr(elt, context) for elt in node.elts)
        )
    if isinstance(node, ast.Attribute):
        path = _attribute_path(node)
        if path is not None and path[0] in {"pto", "PAT", "PIPE", "EVENT"} and len(path) >= 2:
            return FrontendSymbolExpr(namespace=".".join(path[:-1]), name=path[-1])
        return FrontendAttributeExpr(base=_build_expr(node.value, context), attr=node.attr)
    if isinstance(node, ast.Subscript):
        return FrontendSubscriptExpr(
            base=_build_expr(node.value, context),
            index=_build_expr(node.slice, context),
        )
    if isinstance(node, ast.BinOp):
        op_name = _BINARY_OP_NAMES.get(type(node.op))
        if op_name is None:
            raise context.error(
                node,
                f"unsupported binary operator `{type(node.op).__name__}` in TileLang DSL v1",
            )
        return FrontendBinaryExpr(
            lhs=_build_expr(node.left, context),
            op=op_name,
            rhs=_build_expr(node.right, context),
        )
    if isinstance(node, ast.Call):
        if (
            isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "pto"
            and node.func.attr == "tpl"
        ):
            if not node.args:
                raise context.error(
                    node,
                    "pto.tpl() requires a non-empty string literal slot name as the first argument",
                )
            slot_expr = node.args[0]
            if not (
                isinstance(slot_expr, ast.Constant)
                and isinstance(slot_expr.value, str)
                and slot_expr.value
            ):
                raise context.error(
                    slot_expr,
                    "pto.tpl() requires a non-empty string literal slot name",
                )
            slot_name = slot_expr.value
            slot_bindings = context.templates.get(slot_name)
            if slot_bindings is None:
                raise context.error(
                    slot_expr,
                    f"unknown template slot {slot_name!r} in TileLang DSL v1",
                )
            if context.selected_op is None:
                raise context.error(
                    node,
                    "pto.tpl() requires pto.select_kernel(...) to bind a concrete op before expansion",
                )
            resolved_op = slot_bindings.get(context.selected_op)
            if resolved_op is None:
                raise context.error(
                    slot_expr,
                    f"template slot {slot_name!r} does not define an implementation for "
                    f"selected op {context.selected_op!r}",
                )
            _validate_resolved_template_op_surface(resolved_op, node, context)
            return FrontendCallExpr(
                namespace="pto",
                name=resolved_op,
                args=tuple(_build_expr(arg, context) for arg in node.args[1:]),
            )
        if isinstance(node.func, ast.Name):
            return FrontendCallExpr(
                namespace=None,
                name=node.func.id,
                args=tuple(_build_expr(arg, context) for arg in node.args),
            )
        if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            return FrontendCallExpr(
                namespace=node.func.value.id,
                name=node.func.attr,
                args=tuple(_build_expr(arg, context) for arg in node.args),
            )
    raise context.error(
        node,
        f"unsupported expression `{type(node).__name__}` in TileLang DSL v1",
    )


def _build_target(node: ast.AST, context: _FrontendBuildContext) -> FrontendTargetNode:
    if isinstance(node, ast.Name):
        return FrontendNameTarget(name=node.id)
    if isinstance(node, ast.Tuple):
        elements = []
        for elt in node.elts:
            if not isinstance(elt, ast.Name):
                raise context.error(elt, "tuple assignment only supports names in TileLang DSL v1")
            elements.append(FrontendNameTarget(name=elt.id))
        return FrontendTupleTarget(elements=tuple(elements))
    raise context.error(
        node,
        f"unsupported assignment target `{type(node).__name__}` in TileLang DSL v1",
    )


def _build_stmt(node: ast.stmt, context: _FrontendBuildContext) -> FrontendStmtNode:
    if isinstance(node, ast.Assign):
        if len(node.targets) != 1:
            raise context.error(node, "multiple assignment targets are not supported in TileLang DSL v1")
        return FrontendAssignStmt(
            target=_build_target(node.targets[0], context),
            value=_build_expr(node.value, context),
        )
    if isinstance(node, ast.AnnAssign):
        if node.value is None:
            raise context.error(node, "annotation-only assignments are not supported in TileLang DSL v1")
        return FrontendAssignStmt(
            target=_build_target(node.target, context),
            value=_build_expr(node.value, context),
            annotation=node.annotation,
        )
    if isinstance(node, ast.Expr):
        return FrontendExprStmt(expr=_build_expr(node.value, context))
    if isinstance(node, ast.Return):
        value = None
        if node.value is not None:
            if not (isinstance(node.value, ast.Constant) and node.value.value is None):
                value = _build_expr(node.value, context)
        return FrontendReturnStmt(value=value)
    if isinstance(node, ast.For):
        if not isinstance(node.target, ast.Name):
            raise context.error(node.target, "for target must be a single name")
        if not isinstance(node.iter, ast.Call) or not isinstance(node.iter.func, ast.Name) or node.iter.func.id != "range":
            raise context.error(node.iter, "only Python range(lb, ub, step) loops are supported")
        if len(node.iter.args) != 3:
            raise context.error(node.iter, "range() expects exactly 3 arguments in TileLang DSL v1")
        return FrontendForStmt(
            target=node.target.id,
            lower_bound=_build_expr(node.iter.args[0], context),
            upper_bound=_build_expr(node.iter.args[1], context),
            step=_build_expr(node.iter.args[2], context),
            body=tuple(_build_stmt(stmt, context) for stmt in node.body),
        )
    if isinstance(node, ast.If):
        return FrontendIfStmt(
            condition=_build_expr(node.test, context),
            then_body=tuple(_build_stmt(stmt, context) for stmt in node.body),
            else_body=tuple(_build_stmt(stmt, context) for stmt in node.orelse),
        )
    if isinstance(node, ast.With):
        if len(node.items) != 1:
            raise context.error(node, "only a single with-item is supported in TileLang DSL v1")
        item = node.items[0]
        if not isinstance(item.context_expr, ast.Call):
            raise context.error(item.context_expr, "with context must be a call in TileLang DSL v1")
        if not (
            isinstance(item.context_expr.func, ast.Attribute)
            and isinstance(item.context_expr.func.value, ast.Name)
            and item.context_expr.func.value.id == "pto"
            and item.context_expr.func.attr == "strict_vecscope"
        ):
            raise context.error(item.context_expr, "only pto.strict_vecscope is supported in TileLang DSL v1")
        if not context.advanced_enabled:
            raise context.error(
                item.context_expr,
                advanced_mode_message("strict_vecscope"),
            )
        if not isinstance(item.optional_vars, ast.Tuple):
            raise context.error(item, "pto.strict_vecscope requires tuple binding in 'as'")
        block_arguments = []
        for elt in item.optional_vars.elts:
            if not isinstance(elt, ast.Name):
                raise context.error(elt, "pto.strict_vecscope bindings must be names")
            block_arguments.append(elt.id)
        return FrontendStrictVecscopeStmt(
            captures=tuple(_build_expr(arg, context) for arg in item.context_expr.args),
            block_arguments=tuple(block_arguments),
            body=tuple(_build_stmt(stmt, context.nested_vecscope()) for stmt in node.body),
        )
    raise context.error(
        node,
        f"unsupported statement `{type(node).__name__}` in TileLang DSL v1",
    )


def build_frontend_kernel_node(descriptor: Any) -> FrontendKernelNode:
    """Project the core-foundation descriptor into a lowering-owned AST."""

    parameters = tuple(
        FrontendParameterNode(
            name=param.name,
            kind=param.kind,
            annotation=param.annotation,
            dtype=param.dtype,
        )
        for param in descriptor.parameters
    )
    tile_specializations = tuple(
        FrontendTileSpecializationNode(
            name=name,
            shape=spec.shape,
            memory_space=spec.memory_space.value,
            config=spec.config,
            valid_shape=spec.valid_shape,
        )
        for name, spec in descriptor.specializations
    )
    source_info = descriptor._source_info
    context = _FrontendBuildContext(
        source_info=source_info,
        templates=descriptor.templates,
        selected_op=descriptor.selected_op,
        advanced_enabled=descriptor.advanced_enabled,
    )
    body = ()
    if source_info is not None:
        body = tuple(_build_stmt(stmt, context) for stmt in source_info.function_def.body)
    return FrontendKernelNode(
        target=descriptor.target,
        op=descriptor.op,
        name=descriptor.name,
        verify_enabled=descriptor.verify_enabled,
        advanced_enabled=descriptor.advanced_enabled,
        dtype_signature=descriptor.dtype_signature,
        parameters=parameters,
        tile_specializations=tile_specializations,
        body=body,
    )


__all__ = [
    "FrontendAssignStmt",
    "FrontendAttributeExpr",
    "FrontendBinaryExpr",
    "FrontendCallExpr",
    "FrontendConstantExpr",
    "FrontendExprNode",
    "FrontendExprStmt",
    "FrontendForStmt",
    "FrontendIfStmt",
    "FrontendKernelNode",
    "FrontendNameExpr",
    "FrontendNameTarget",
    "FrontendParameterNode",
    "FrontendReturnStmt",
    "FrontendSliceExpr",
    "FrontendStrictVecscopeStmt",
    "FrontendStmtNode",
    "FrontendSubscriptExpr",
    "FrontendSymbolExpr",
    "FrontendTargetNode",
    "FrontendTileSpecializationNode",
    "FrontendTupleExpr",
    "FrontendTupleTarget",
    "build_frontend_kernel_node",
]
