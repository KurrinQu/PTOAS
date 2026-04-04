"""Semantic model for TileLang DSL descriptor lowering."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any

from .frontend_ast import (
    FrontendAssignStmt,
    FrontendAttributeExpr,
    FrontendBinaryExpr,
    FrontendCallExpr,
    FrontendConstantExpr,
    FrontendExprNode,
    FrontendExprStmt,
    FrontendForStmt,
    FrontendIfStmt,
    FrontendKernelNode,
    FrontendNameExpr,
    FrontendNameTarget,
    FrontendReturnStmt,
    FrontendSliceExpr,
    FrontendStrictVecscopeStmt,
    FrontendStmtNode,
    FrontendSubscriptExpr,
    FrontendSymbolExpr,
    FrontendTargetNode,
    FrontendTupleExpr,
    FrontendTupleTarget,
)
from .support_matrix import (
    DEFERRED_PTO_SURFACES,
    deferred_surface_message,
    unsupported_feature_message,
)
from .types import Event, MaskPattern, Pipe, ScalarType, bf16, f16, f32, i1, i8, i16, i32


_DTYPE_SYMBOLS = {
    "i1": i1,
    "i8": i8,
    "i16": i16,
    "i32": i32,
    "f16": f16,
    "bf16": bf16,
    "f32": f32,
}
_PATTERN_SYMBOLS = {pattern.name: pattern for pattern in MaskPattern}
_PIPE_SYMBOLS = {pipe.name: pipe for pipe in Pipe}
_EVENT_SYMBOLS = {event.name: event for event in Event}
_UNARY_VECTOR_OPS = {"vabs", "vrelu", "vexp", "vnot"}
_BINARY_VECTOR_OPS = {"vadd", "vsub", "vmul", "vdiv", "vmax", "vmin", "vand", "vor", "vxor"}
_VECTOR_SCALAR_OPS = {"vadds", "vsubs", "vmuls", "vdivs", "vmaxs", "vmins"}


class SemanticType:
    """Base class for semantic value types."""


@dataclass(frozen=True)
class SemanticTensorViewType(SemanticType):
    element_dtype: ScalarType
    rank: int = 2


@dataclass(frozen=True)
class SemanticTensorSliceType(SemanticType):
    element_dtype: ScalarType
    rank: int
    extents: tuple[int | None, ...]


@dataclass(frozen=True)
class SemanticTileType(SemanticType):
    element_dtype: ScalarType
    rank: int
    shape: tuple[int, ...] | None
    memory_space: str | None


@dataclass(frozen=True)
class SemanticScalarType(SemanticType):
    dtype: ScalarType


@dataclass(frozen=True)
class SemanticIndexType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticShapeType(SemanticType):
    rank: int


@dataclass(frozen=True)
class SemanticSliceType(SemanticType):
    pass


@dataclass(frozen=True)
class SemanticTupleType(SemanticType):
    elements: tuple[SemanticType, ...]


@dataclass(frozen=True)
class SemanticMetaType(SemanticType):
    kind: str


@dataclass(frozen=True)
class SemanticMaskType(SemanticType):
    granularity: str


@dataclass(frozen=True)
class SemanticVRegType(SemanticType):
    element_dtype: ScalarType
    lanes: int


_I32_TYPE = SemanticScalarType(dtype=i32)


@dataclass(frozen=True)
class SemanticBinding:
    name: str
    ssa_name: str
    type: SemanticType
    origin: str
    value: Any | None = None


@dataclass(frozen=True)
class SemanticTileBinding:
    name: str
    shape: tuple[int, ...]
    memory_space: str
    config: Any


class SemanticExpr:
    """Base class for typed semantic expressions."""


@dataclass(frozen=True)
class SemanticBindingRef(SemanticExpr):
    binding: SemanticBinding
    type: SemanticType


@dataclass(frozen=True)
class SemanticLiteralExpr(SemanticExpr):
    value: Any
    type: SemanticType


@dataclass(frozen=True)
class SemanticSymbolExpr(SemanticExpr):
    namespace: str
    name: str
    value: Any
    type: SemanticMetaType


@dataclass(frozen=True)
class SemanticSliceExpr(SemanticExpr):
    start: SemanticExpr | None
    stop: SemanticExpr | None
    step: SemanticExpr | None
    type: SemanticSliceType


@dataclass(frozen=True)
class SemanticTupleExpr(SemanticExpr):
    elements: tuple[SemanticExpr, ...]
    type: SemanticTupleType


@dataclass(frozen=True)
class SemanticAttributeAccess(SemanticExpr):
    base: SemanticExpr
    attr: str
    type: SemanticType


@dataclass(frozen=True)
class SemanticSubscriptAccess(SemanticExpr):
    base: SemanticExpr
    index: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticTensorSliceExpr(SemanticExpr):
    base: SemanticExpr
    slices: tuple[SemanticSliceExpr, ...]
    type: SemanticTensorSliceType


@dataclass(frozen=True)
class SemanticBinaryExpr(SemanticExpr):
    lhs: SemanticExpr
    op: str
    rhs: SemanticExpr
    type: SemanticType


@dataclass(frozen=True)
class SemanticCallExpr(SemanticExpr):
    namespace: str | None
    name: str
    args: tuple[SemanticExpr, ...]
    type: SemanticType | None


class SemanticStmt:
    """Base class for semantic statements."""


@dataclass(frozen=True)
class SemanticAssignStmt(SemanticStmt):
    targets: tuple[SemanticBinding, ...]
    value: SemanticExpr
    annotation: Any | None = None


@dataclass(frozen=True)
class SemanticExprStmt(SemanticStmt):
    expr: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaLoadStmt(SemanticStmt):
    src: SemanticTensorSliceExpr
    dst: SemanticExpr


@dataclass(frozen=True)
class SemanticDmaStoreStmt(SemanticStmt):
    src: SemanticExpr
    dst: SemanticTensorSliceExpr


@dataclass(frozen=True)
class SemanticVectorStoreStmt(SemanticStmt):
    value: SemanticExpr
    destination: SemanticExpr
    indices: tuple[SemanticExpr, ...]
    mask: SemanticExpr


@dataclass(frozen=True)
class SemanticVecscopeStmt(SemanticStmt):
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticSetFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticWaitFlagStmt(SemanticStmt):
    src_pipe: str
    dst_pipe: str
    event: str


@dataclass(frozen=True)
class SemanticPipeBarrierStmt(SemanticStmt):
    pipe: str


@dataclass(frozen=True)
class SemanticIfResult:
    result_binding: SemanticBinding
    then_binding: SemanticBinding
    else_binding: SemanticBinding


@dataclass(frozen=True)
class SemanticIfStmt(SemanticStmt):
    condition: SemanticExpr
    then_body: tuple[SemanticStmt, ...]
    else_body: tuple[SemanticStmt, ...]
    results: tuple[SemanticIfResult, ...]


@dataclass(frozen=True)
class SemanticReturnStmt(SemanticStmt):
    value: SemanticExpr | None


@dataclass(frozen=True)
class SemanticForStmt(SemanticStmt):
    induction_variable: SemanticBinding
    lower_bound: SemanticExpr
    upper_bound: SemanticExpr
    step: SemanticExpr
    body: tuple[SemanticStmt, ...]
    loop_carried: tuple[SemanticBinding, ...]


@dataclass(frozen=True)
class SemanticStrictVecscopeStmt(SemanticStmt):
    captures: tuple[SemanticExpr, ...]
    block_arguments: tuple[SemanticBinding, ...]
    body: tuple[SemanticStmt, ...]


@dataclass(frozen=True)
class SemanticParameter:
    binding: SemanticBinding

    @property
    def name(self) -> str:
        return self.binding.name

    @property
    def kind(self) -> str:
        return self.binding.origin

    @property
    def type(self) -> SemanticType:
        return self.binding.type

    @property
    def ssa_name(self) -> str:
        return self.binding.ssa_name


@dataclass(frozen=True)
class SemanticKernel:
    target: str
    op: str
    symbol_name: str
    verify_enabled: bool
    advanced_enabled: bool
    dtype_signature: tuple[Any, ...]
    parameters: tuple[SemanticParameter, ...]
    tile_bindings: tuple[SemanticTileBinding, ...]
    body: tuple[SemanticStmt, ...]


class _SemanticAnalyzer:
    def __init__(self, node: FrontendKernelNode):
        self.node = node
        self._counter = 0
        self._disable_inference_depth = 0
        self._tile_specializations = {
            spec.name: spec for spec in node.tile_specializations
        }
        self._tensor_shape_parameters: list[SemanticParameter] = []

    def analyze(self) -> SemanticKernel:
        env: dict[str, SemanticBinding] = {}
        parameters = []
        for index, param in enumerate(self.node.parameters):
            binding = SemanticBinding(
                name=param.name,
                ssa_name=f"%arg{index}",
                type=self._parameter_type(param),
                origin=param.kind,
            )
            env[param.name] = binding
            parameters.append(SemanticParameter(binding=binding))
        body, _ = self._analyze_kernel_body(env)
        parameters.extend(self._tensor_shape_parameters)
        tile_bindings = tuple(
            SemanticTileBinding(
                name=spec.name,
                shape=spec.shape,
                memory_space=spec.memory_space,
                config=spec.config,
            )
            for spec in self.node.tile_specializations
        )
        return SemanticKernel(
            target=self.node.target,
            op=self.node.op,
            symbol_name=self.node.name,
            verify_enabled=self.node.verify_enabled,
            advanced_enabled=self.node.advanced_enabled,
            dtype_signature=self.node.dtype_signature,
            parameters=tuple(parameters),
            tile_bindings=tile_bindings,
            body=body,
        )

    def _analyze_kernel_body(
        self,
        env: dict[str, SemanticBinding],
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        if not self.node.advanced_enabled:
            return self._analyze_block(self.node.body, env, allow_outer_lookup=True)

        body_without_return = self.node.body
        trailing_return: FrontendReturnStmt | None = None
        if body_without_return and isinstance(body_without_return[-1], FrontendReturnStmt):
            trailing_return = body_without_return[-1]
            body_without_return = body_without_return[:-1]

        if body_without_return and self._can_wrap_whole_kernel_vecscope(body_without_return):
            self._disable_inference_depth += 1
            try:
                scoped_body, scoped_env = self._analyze_block_without_inference(
                    body_without_return,
                    env,
                    allow_outer_lookup=True,
                )
            finally:
                self._disable_inference_depth -= 1
            semantic_body: list[SemanticStmt] = []
            if self._semantic_block_contains_vector_activity(scoped_body):
                semantic_body.append(SemanticVecscopeStmt(body=scoped_body))
            else:
                semantic_body.extend(scoped_body)
            if trailing_return is not None:
                return_stmt, scoped_env = self._analyze_stmt(
                    trailing_return,
                    scoped_env,
                    allow_outer_lookup=True,
                )
                semantic_body.append(return_stmt)
            return tuple(semantic_body), scoped_env

        return self._analyze_block(self.node.body, env, allow_outer_lookup=True)

    def _parameter_type(self, param: Any) -> SemanticType:
        if param.kind == "tensorview":
            return SemanticTensorViewType(element_dtype=param.dtype)
        if param.kind == "tile":
            spec = self._tile_specializations.get(param.name)
            rank = 2 if spec is None else len(spec.shape)
            shape = None if spec is None else spec.shape
            memory_space = None if spec is None else spec.memory_space
            return SemanticTileType(
                element_dtype=param.dtype,
                rank=rank,
                shape=shape,
                memory_space=memory_space,
            )
        if param.kind == "scalar":
            return SemanticScalarType(dtype=param.dtype)
        raise ValueError(f"unsupported parameter kind {param.kind!r}")

    def _new_ssa_name(self, stem: str) -> str:
        name = f"%{stem}_{self._counter}"
        self._counter += 1
        return name

    def _tensor_shape_binding_name(self, tensor_name: str, axis: int) -> str:
        return f"__shape_{tensor_name}_{axis}"

    def _ensure_tensor_shape_parameter(
        self,
        tensor_binding: SemanticBinding,
        axis: int,
    ) -> SemanticBinding:
        hidden_name = self._tensor_shape_binding_name(tensor_binding.name, axis)
        for parameter in self._tensor_shape_parameters:
            if parameter.name == hidden_name:
                return parameter.binding
        binding = SemanticBinding(
            name=hidden_name,
            ssa_name=f"%arg{len(self.node.parameters) + len(self._tensor_shape_parameters)}",
            type=SemanticIndexType(),
            origin="tensorview_shape",
        )
        self._tensor_shape_parameters.append(SemanticParameter(binding=binding))
        return binding

    def _make_binding(
        self,
        name: str,
        ty: SemanticType,
        origin: str,
        *,
        value: Any | None = None,
    ) -> SemanticBinding:
        stem = name if name.isidentifier() else "v"
        return SemanticBinding(
            name=name,
            ssa_name=self._new_ssa_name(stem),
            type=ty,
            origin=origin,
            value=value,
        )

    def _analyze_block(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        index = 0
        while index < len(statements):
            if self._should_infer_vecscope(statements[index], allow_outer_lookup=allow_outer_lookup):
                end = index + 1
                while end < len(statements) and self._should_infer_vecscope(
                    statements[end],
                    allow_outer_lookup=allow_outer_lookup,
                ):
                    end += 1
                run = statements[index:end]
                if self._run_contains_vector_op(run):
                    semantic_statements.append(
                        self._analyze_inferred_vecscope(
                            run,
                            current_env,
                            allow_outer_lookup=allow_outer_lookup,
                        )
                    )
                else:
                    for stmt in run:
                        semantic_stmt, current_env = self._analyze_stmt(
                            stmt,
                            current_env,
                            allow_outer_lookup=allow_outer_lookup,
                        )
                        semantic_statements.append(semantic_stmt)
                index = end
                continue

            semantic_stmt, current_env = self._analyze_stmt(
                statements[index],
                current_env,
                allow_outer_lookup=allow_outer_lookup,
            )
            semantic_statements.append(semantic_stmt)
            index += 1
        return tuple(semantic_statements), current_env

    def _should_infer_vecscope(
        self,
        stmt: FrontendStmtNode,
        *,
        allow_outer_lookup: bool,
    ) -> bool:
        if self._disable_inference_depth > 0:
            return False
        if not self.node.advanced_enabled or not allow_outer_lookup:
            return False
        name = self._frontend_vector_call_name(stmt)
        return name in {"make_mask", "vlds", "vsts"} | _UNARY_VECTOR_OPS | _BINARY_VECTOR_OPS | _VECTOR_SCALAR_OPS

    def _run_contains_vector_op(self, statements: tuple[FrontendStmtNode, ...]) -> bool:
        for stmt in statements:
            name = self._frontend_vector_call_name(stmt)
            if name is None or name == "make_mask":
                continue
            return True
        return False

    def _frontend_vector_call_name(self, stmt: FrontendStmtNode) -> str | None:
        expr: FrontendExprNode | None = None
        if isinstance(stmt, FrontendAssignStmt):
            expr = stmt.value
        elif isinstance(stmt, FrontendExprStmt):
            expr = stmt.expr
        if (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
        ):
            return expr.name
        return None

    def _analyze_inferred_vecscope(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticVecscopeStmt:
        body, _ = self._analyze_block_without_inference(
            statements,
            env,
            allow_outer_lookup=allow_outer_lookup,
        )
        return SemanticVecscopeStmt(body=body)

    def _analyze_block_without_inference(
        self,
        statements: tuple[FrontendStmtNode, ...],
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[tuple[SemanticStmt, ...], dict[str, SemanticBinding]]:
        current_env = dict(env)
        semantic_statements = []
        for stmt in statements:
            semantic_stmt, current_env = self._analyze_stmt(
                stmt,
                current_env,
                allow_outer_lookup=allow_outer_lookup,
            )
            semantic_statements.append(semantic_stmt)
        return tuple(semantic_statements), current_env

    def _can_wrap_whole_kernel_vecscope(
        self,
        statements: tuple[FrontendStmtNode, ...],
    ) -> bool:
        for stmt in statements:
            if isinstance(stmt, FrontendStrictVecscopeStmt):
                return False
            if isinstance(stmt, FrontendExprStmt) and (
                self._is_dma_call(stmt.expr) or self._is_sync_call(stmt.expr)
            ):
                return False
            nested_blocks: tuple[tuple[FrontendStmtNode, ...], ...] = ()
            if isinstance(stmt, FrontendForStmt):
                nested_blocks = (stmt.body,)
            elif isinstance(stmt, FrontendIfStmt):
                nested_blocks = (stmt.then_body, stmt.else_body)
            for block in nested_blocks:
                if not self._can_wrap_whole_kernel_vecscope(block):
                    return False
        return True

    def _semantic_block_contains_vector_activity(
        self,
        statements: tuple[SemanticStmt, ...],
    ) -> bool:
        for stmt in statements:
            if isinstance(stmt, SemanticVecscopeStmt):
                return True
            if isinstance(stmt, SemanticStrictVecscopeStmt):
                return True
            if isinstance(stmt, SemanticVectorStoreStmt):
                return True
            if isinstance(stmt, SemanticAssignStmt) and self._expr_contains_vector_activity(stmt.value):
                return True
            if isinstance(stmt, SemanticExprStmt) and self._expr_contains_vector_activity(stmt.expr):
                return True
            if isinstance(stmt, SemanticForStmt) and self._semantic_block_contains_vector_activity(stmt.body):
                return True
            if isinstance(stmt, SemanticIfStmt) and (
                self._semantic_block_contains_vector_activity(stmt.then_body)
                or self._semantic_block_contains_vector_activity(stmt.else_body)
            ):
                return True
        return False

    def _expr_contains_vector_activity(self, expr: SemanticExpr) -> bool:
        if isinstance(expr, SemanticCallExpr):
            if expr.namespace == "pto" and expr.name in (
                {"make_mask", "vlds"} | _UNARY_VECTOR_OPS | _BINARY_VECTOR_OPS | _VECTOR_SCALAR_OPS
            ):
                return True
            return any(self._expr_contains_vector_activity(arg) for arg in expr.args)
        if isinstance(expr, SemanticBinaryExpr):
            return self._expr_contains_vector_activity(expr.lhs) or self._expr_contains_vector_activity(expr.rhs)
        if isinstance(expr, SemanticTupleExpr):
            return any(self._expr_contains_vector_activity(element) for element in expr.elements)
        if isinstance(expr, SemanticAttributeAccess):
            return self._expr_contains_vector_activity(expr.base)
        if isinstance(expr, SemanticSubscriptAccess):
            return self._expr_contains_vector_activity(expr.base) or self._expr_contains_vector_activity(expr.index)
        if isinstance(expr, SemanticTensorSliceExpr):
            return self._expr_contains_vector_activity(expr.base)
        return False

    def _analyze_stmt(
        self,
        stmt: FrontendStmtNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if isinstance(stmt, FrontendAssignStmt):
            value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            updated_env = dict(env)
            targets = self._bind_assignment_target(
                stmt.target,
                value,
                updated_env,
                stmt.annotation,
            )
            return (
                SemanticAssignStmt(targets=targets, value=value, annotation=stmt.annotation),
                updated_env,
            )
        if isinstance(stmt, FrontendExprStmt):
            if self._is_dma_call(stmt.expr):
                return self._analyze_dma_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_sync_call(stmt.expr):
                return self._analyze_sync_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            if self._is_vector_store_call(stmt.expr):
                return self._analyze_vector_store_stmt(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            expr = self._analyze_expr(stmt.expr, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticExprStmt(expr=expr), dict(env)
        if isinstance(stmt, FrontendReturnStmt):
            value = None
            if stmt.value is not None:
                value = self._analyze_expr(stmt.value, env, allow_outer_lookup=allow_outer_lookup)
            return SemanticReturnStmt(value=value), dict(env)
        if isinstance(stmt, FrontendForStmt):
            return self._analyze_for(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendIfStmt):
            return self._analyze_if(stmt, env, allow_outer_lookup=allow_outer_lookup)
        if isinstance(stmt, FrontendStrictVecscopeStmt):
            return self._analyze_strict_vecscope(stmt, env)
        raise ValueError(f"unsupported frontend statement {type(stmt).__name__}")

    def _is_dma_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"dma_load", "dma_store"}
        )

    def _is_vector_store_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name == "vsts"
        )

    def _is_sync_call(self, expr: FrontendExprNode) -> bool:
        return (
            isinstance(expr, FrontendCallExpr)
            and expr.namespace == "pto"
            and expr.name in {"set_flag", "wait_flag", "pipe_barrier", "barrier"}
        )

    def _analyze_dma_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if expr.name == "dma_load":
            if len(args) != 2:
                raise TypeError("pto.dma_load expects exactly 2 positional arguments in TileLang DSL v1")
            src = self._require_tensor_slice(args[0], "pto.dma_load source")
            dst = self._require_tile_expr(args[1], "pto.dma_load destination")
            self._validate_dma_shape_match(src.type, dst.type, "pto.dma_load")
            return SemanticDmaLoadStmt(src=src, dst=dst), dict(env)
        if expr.name == "dma_store":
            if len(args) != 2:
                raise TypeError("pto.dma_store expects exactly 2 positional arguments in TileLang DSL v1")
            src = self._require_tile_expr(args[0], "pto.dma_store source")
            dst = self._require_tensor_slice(args[1], "pto.dma_store destination")
            self._validate_dma_shape_match(dst.type, src.type, "pto.dma_store")
            return SemanticDmaStoreStmt(src=src, dst=dst), dict(env)
        raise ValueError(f"unsupported DMA stmt pto.{expr.name}")

    def _analyze_vector_store_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if len(expr.args) == 3:
            value = self._analyze_expr(expr.args[0], env, allow_outer_lookup=allow_outer_lookup)
            destination, indices = self._analyze_tile_vector_access(
                expr.args[1],
                env,
                allow_outer_lookup=allow_outer_lookup,
                context="pto.vsts destination",
            )
            mask = self._analyze_expr(expr.args[2], env, allow_outer_lookup=allow_outer_lookup)
        else:
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            if len(args) != 4:
                raise TypeError("pto.vsts expects 3 or 4 positional arguments in TileLang DSL v1")
            value, destination, offset, mask = args
            indices = (offset,)
        self._require_vreg_expr(value, "pto.vsts value")
        self._require_tile_expr(destination, "pto.vsts destination")
        for index in indices:
            self._require_index_typed_expr(index)
        self._require_mask_for_vreg(mask, value.type, "pto.vsts")
        self._require_matching_vector_pointer(value.type, destination.type, "pto.vsts")
        return (
            SemanticVectorStoreStmt(
                value=value,
                destination=destination,
                indices=indices,
                mask=mask,
            ),
            dict(env),
        )

    def _analyze_sync_stmt(
        self,
        expr: FrontendCallExpr,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        args = tuple(
            self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
            for arg in expr.args
        )
        if expr.name in {"set_flag", "wait_flag"}:
            if len(args) != 3:
                raise TypeError(f"pto.{expr.name} expects exactly 3 positional arguments in TileLang DSL v1")
            src_pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} source pipe")
            dst_pipe = self._require_sync_pipe(args[1], f"pto.{expr.name} destination pipe")
            event = self._require_sync_event(args[2], f"pto.{expr.name} event")
            if expr.name == "set_flag":
                return SemanticSetFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
            return SemanticWaitFlagStmt(src_pipe=src_pipe, dst_pipe=dst_pipe, event=event), dict(env)
        if expr.name in {"pipe_barrier", "barrier"}:
            if len(args) != 1:
                raise TypeError(f"pto.{expr.name} expects exactly 1 positional argument in TileLang DSL v1")
            pipe = self._require_sync_pipe(args[0], f"pto.{expr.name} pipe")
            return SemanticPipeBarrierStmt(pipe=pipe), dict(env)
        raise ValueError(f"unsupported sync stmt pto.{expr.name}")

    def _require_tensor_slice(
        self,
        expr: SemanticExpr,
        context: str,
    ) -> SemanticTensorSliceExpr:
        if not isinstance(expr, SemanticTensorSliceExpr):
            raise TypeError(f"{context} must be a TensorView slice in TileLang DSL v1")
        return expr

    def _require_tile_expr(self, expr: SemanticExpr, context: str) -> SemanticExpr:
        if not isinstance(expr.type, SemanticTileType):
            raise TypeError(f"{context} must be a Tile value in TileLang DSL v1")
        if expr.type.rank != 2:
            raise TypeError(f"{context} currently only supports rank-2 Tile values in TileLang DSL v1")
        if expr.type.shape is None:
            raise TypeError(f"{context} requires a statically specialized Tile shape in TileLang DSL v1")
        if expr.type.memory_space != "ub":
            raise TypeError(f"{context} currently only supports MemorySpace.UB Tile values in TileLang DSL v1")
        return expr

    def _validate_dma_shape_match(
        self,
        tensor_slice_type: SemanticTensorSliceType,
        tile_type: SemanticTileType,
        op_name: str,
    ) -> None:
        if tensor_slice_type.rank != 2:
            raise TypeError(f"{op_name} currently only supports rank-2 TensorView slices in TileLang DSL v1")
        if tile_type.rank != 2 or tile_type.shape is None:
            raise TypeError(f"{op_name} requires a statically specialized rank-2 Tile in TileLang DSL v1")
        if tensor_slice_type.element_dtype != tile_type.element_dtype:
            raise TypeError(f"{op_name} requires matching TensorView/Tile element dtypes in TileLang DSL v1")
        for axis, (extent, tile_dim) in enumerate(zip(tensor_slice_type.extents, tile_type.shape)):
            if extent is not None and extent != tile_dim:
                raise TypeError(
                    f"{op_name} requires TensorView slice extent axis {axis}={extent!r} "
                    f"to match Tile shape axis {axis}={tile_dim!r}"
                )

    def _bind_assignment_target(
        self,
        target: FrontendTargetNode,
        value: SemanticExpr,
        env: dict[str, SemanticBinding],
        annotation: Any | None,
    ) -> tuple[SemanticBinding, ...]:
        if isinstance(target, FrontendNameTarget):
            if isinstance(value.type, SemanticTupleType):
                raise ValueError("multi-result call assignment requires tuple binding in TileLang DSL v1")
            annotated_type = self._annotation_type(annotation, value.type)
            binding = self._make_binding(
                target.name,
                annotated_type if annotated_type is not None else value.type,
                "ssa",
                value=self._binding_value_for_expr(value),
            )
            env[target.name] = binding
            return (binding,)
        if isinstance(target, FrontendTupleTarget):
            if not isinstance(value.type, SemanticTupleType):
                raise ValueError("tuple assignment expects a tuple-typed value")
            if annotation is not None:
                raise TypeError("annotated tuple assignment is not supported in TileLang DSL v1")
            if len(target.elements) != len(value.type.elements):
                raise ValueError("tuple assignment arity must match the tuple value")
            tuple_values: tuple[SemanticExpr, ...]
            if isinstance(value, SemanticTupleExpr):
                tuple_values = value.elements
            elif isinstance(value, SemanticCallExpr):
                tuple_values = value.args
            else:
                tuple_values = tuple(SemanticLiteralExpr(value=None, type=element_type) for element_type in value.type.elements)
            bindings = []
            for element, element_type, element_value in zip(target.elements, value.type.elements, tuple_values):
                binding = self._make_binding(
                    element.name,
                    element_type,
                    "ssa",
                    value=self._binding_value_for_expr(element_value),
                )
                env[element.name] = binding
                bindings.append(binding)
            return tuple(bindings)
        raise ValueError(f"unsupported frontend assignment target {type(target).__name__}")

    def _binding_value_for_expr(self, expr: SemanticExpr) -> Any | None:
        if isinstance(expr, SemanticSymbolExpr):
            return expr.value
        if isinstance(expr, SemanticLiteralExpr):
            return expr.value
        if isinstance(expr, SemanticBindingRef):
            return expr.binding.value
        return None

    def _annotation_type(
        self,
        annotation: Any | None,
        inferred_type: SemanticType | None,
    ) -> SemanticType | None:
        if annotation is None:
            return inferred_type
        if isinstance(annotation, ast.Attribute) and isinstance(annotation.value, ast.Name):
            if annotation.value.id == "pto" and isinstance(inferred_type, SemanticScalarType):
                if inferred_type.dtype.name != annotation.attr:
                    raise TypeError(
                        f"annotated scalar type `pto.{annotation.attr}` does not match inferred {inferred_type.dtype!r}"
                    )
                return inferred_type
        raise TypeError("unsupported annotated assignment type in TileLang DSL v1")

    def _analyze_for(
        self,
        stmt: FrontendForStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        lower_bound = self._analyze_expr(stmt.lower_bound, env, allow_outer_lookup=allow_outer_lookup)
        upper_bound = self._analyze_expr(stmt.upper_bound, env, allow_outer_lookup=allow_outer_lookup)
        step = self._analyze_expr(stmt.step, env, allow_outer_lookup=allow_outer_lookup)
        for expr in (lower_bound, upper_bound, step):
            self._require_loop_bound_type(expr.type)

        body_env = dict(env)
        induction_variable = self._make_binding(stmt.target, SemanticIndexType(), "loop_iv")
        body_env[stmt.target] = induction_variable
        body, final_body_env = self._analyze_block(
            stmt.body,
            body_env,
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        loop_carried = []
        for name, outer_binding in env.items():
            final_binding = final_body_env.get(name)
            if final_binding is None or final_binding is outer_binding:
                continue
            merged_type = self._merge_loop_carried_types(outer_binding.type, final_binding.type)
            if merged_type is None:
                raise TypeError(
                    f"loop-carried binding '{name}' changes type from {outer_binding.type!r} to {final_binding.type!r}"
                )
            merged = self._make_binding(name, merged_type, "loop_result")
            updated_env[name] = merged
            loop_carried.append(merged)

        return (
            SemanticForStmt(
                induction_variable=induction_variable,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                step=step,
                body=body,
                loop_carried=tuple(loop_carried),
            ),
            updated_env,
        )

    def _analyze_if(
        self,
        stmt: FrontendIfStmt,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        condition = self._analyze_expr(stmt.condition, env, allow_outer_lookup=allow_outer_lookup)
        self._require_condition_type(condition.type)

        then_body, then_env = self._analyze_block(
            stmt.then_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )
        else_body, else_env = self._analyze_block(
            stmt.else_body,
            dict(env),
            allow_outer_lookup=allow_outer_lookup,
        )

        updated_env = dict(env)
        merged_results: list[SemanticIfResult] = []
        for name, outer_binding in env.items():
            then_binding = then_env.get(name, outer_binding)
            else_binding = else_env.get(name, outer_binding)
            if then_binding is outer_binding and else_binding is outer_binding:
                continue
            if then_binding.type != else_binding.type:
                raise TypeError(
                    f"if/else merge for '{name}' changes type between branches: "
                    f"{then_binding.type!r} vs {else_binding.type!r}"
                )
            merged_binding = self._make_binding(name, then_binding.type, "if_result")
            updated_env[name] = merged_binding
            merged_results.append(
                SemanticIfResult(
                    result_binding=merged_binding,
                    then_binding=then_binding,
                    else_binding=else_binding,
                )
            )

        return (
            SemanticIfStmt(
                condition=condition,
                then_body=then_body,
                else_body=else_body,
                results=tuple(merged_results),
            ),
            updated_env,
        )

    def _analyze_strict_vecscope(
        self,
        stmt: FrontendStrictVecscopeStmt,
        env: dict[str, SemanticBinding],
    ) -> tuple[SemanticStmt, dict[str, SemanticBinding]]:
        if len(stmt.captures) != len(stmt.block_arguments):
            raise ValueError("strict_vecscope capture arity must match block arguments")

        captures = tuple(
            self._analyze_expr(expr, env, allow_outer_lookup=True)
            for expr in stmt.captures
        )
        scope_env: dict[str, SemanticBinding] = {}
        block_arguments = []
        for name, capture in zip(stmt.block_arguments, captures):
            if capture.type is None:
                raise TypeError(
                    f"strict_vecscope block argument '{name}' type could not be inferred"
                )
            block_binding = self._make_binding(name, capture.type, "strict_vecscope_arg")
            scope_env[name] = block_binding
            block_arguments.append(block_binding)
        body, _ = self._analyze_block(
            stmt.body,
            scope_env,
            allow_outer_lookup=False,
        )
        return (
            SemanticStrictVecscopeStmt(
                captures=captures,
                block_arguments=tuple(block_arguments),
                body=body,
            ),
            dict(env),
        )

    def _analyze_expr(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
    ) -> SemanticExpr:
        if isinstance(expr, FrontendNameExpr):
            binding = env.get(expr.name)
            if binding is None:
                if allow_outer_lookup:
                    raise ValueError(f"unknown name '{expr.name}'")
                raise ValueError(
                    f"implicit capture of '{expr.name}' is not allowed in pto.strict_vecscope"
                )
            return SemanticBindingRef(binding=binding, type=binding.type)
        if isinstance(expr, FrontendConstantExpr):
            if isinstance(expr.value, bool):
                raise TypeError("bool constants are not supported in TileLang DSL v1 yet")
            if isinstance(expr.value, int):
                return SemanticLiteralExpr(value=expr.value, type=SemanticIndexType())
            if isinstance(expr.value, str):
                return SemanticLiteralExpr(
                    value=expr.value,
                    type=SemanticMetaType(kind="string"),
                )
            if expr.value is None:
                return SemanticLiteralExpr(value=None, type=SemanticIndexType())
            raise TypeError(f"unsupported constant {expr.value!r} in TileLang DSL v1")
        if isinstance(expr, FrontendSymbolExpr):
            return self._analyze_symbol_expr(expr)
        if isinstance(expr, FrontendSliceExpr):
            start = None if expr.start is None else self._analyze_expr(expr.start, env, allow_outer_lookup=allow_outer_lookup)
            stop = None if expr.stop is None else self._analyze_expr(expr.stop, env, allow_outer_lookup=allow_outer_lookup)
            step = None if expr.step is None else self._analyze_expr(expr.step, env, allow_outer_lookup=allow_outer_lookup)
            for item in (start, stop, step):
                if item is not None:
                    self._require_index_typed_expr(item)
            return SemanticSliceExpr(
                start=start,
                stop=stop,
                step=step,
                type=SemanticSliceType(),
            )
        if isinstance(expr, FrontendTupleExpr):
            elements = tuple(
                self._analyze_expr(element, env, allow_outer_lookup=allow_outer_lookup)
                for element in expr.elements
            )
            return SemanticTupleExpr(
                elements=elements,
                type=SemanticTupleType(elements=tuple(element.type for element in elements)),
            )
        if isinstance(expr, FrontendAttributeExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            if expr.attr == "element_type":
                return self._element_type_expr(base)
            if expr.attr == "valid_shape":
                return self._valid_shape_expr(base)
            attr_type = self._attribute_type(base, expr.attr)
            return SemanticAttributeAccess(base=base, attr=expr.attr, type=attr_type)
        if isinstance(expr, FrontendSubscriptExpr):
            base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
            index = self._analyze_expr(expr.index, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._subscript_type(base, index)
            if isinstance(result_type, SemanticTensorSliceType):
                slices = self._normalize_tensor_slice(index, base.type.rank)
                return SemanticTensorSliceExpr(base=base, slices=slices, type=result_type)
            return SemanticSubscriptAccess(base=base, index=index, type=result_type)
        if isinstance(expr, FrontendBinaryExpr):
            lhs = self._analyze_expr(expr.lhs, env, allow_outer_lookup=allow_outer_lookup)
            rhs = self._analyze_expr(expr.rhs, env, allow_outer_lookup=allow_outer_lookup)
            result_type = self._binary_type(lhs, rhs, expr.op)
            return SemanticBinaryExpr(lhs=lhs, op=expr.op, rhs=rhs, type=result_type)
        if isinstance(expr, FrontendCallExpr):
            if expr.namespace == "pto" and expr.name == "vlds" and len(expr.args) == 1:
                base, indices = self._analyze_tile_vector_access(
                    expr.args[0],
                    env,
                    allow_outer_lookup=allow_outer_lookup,
                    context="pto.vlds source",
                )
                return self._analyze_vlds((base, *indices))
            args = tuple(
                self._analyze_expr(arg, env, allow_outer_lookup=allow_outer_lookup)
                for arg in expr.args
            )
            return self._analyze_call_expr(expr.namespace, expr.name, args)
        raise ValueError(f"unsupported frontend expression {type(expr).__name__}")

    def _analyze_symbol_expr(self, expr: FrontendSymbolExpr) -> SemanticExpr:
        if expr.namespace == "pto":
            dtype = _DTYPE_SYMBOLS.get(expr.name)
            if dtype is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=dtype,
                    type=SemanticMetaType(kind="dtype"),
                )
        if expr.namespace in {"PAT", "pto.PAT", "pto.MaskPattern"}:
            pattern = _PATTERN_SYMBOLS.get(expr.name)
            if pattern is None and expr.name.startswith("PAT_"):
                canonical = expr.name[len("PAT_") :]
                pattern = _PATTERN_SYMBOLS.get(canonical)
            if pattern is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pattern,
                    type=SemanticMetaType(kind="mask_pattern"),
                )
        if expr.namespace in {"PIPE", "pto.PIPE"}:
            pipe = _PIPE_SYMBOLS.get(expr.name)
            if pipe is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=pipe,
                    type=SemanticMetaType(kind="pipe"),
                )
        if expr.namespace in {"EVENT", "pto.EVENT"}:
            event = _EVENT_SYMBOLS.get(expr.name)
            if event is not None:
                return SemanticSymbolExpr(
                    namespace=expr.namespace,
                    name=expr.name,
                    value=event,
                    type=SemanticMetaType(kind="event"),
                )
        raise TypeError(
            f"symbol `{expr.namespace}.{expr.name}` is not supported in TileLang DSL v1"
        )

    def _attribute_type(self, base: SemanticExpr, attr: str) -> SemanticType:
        base_type = base.type
        if isinstance(base_type, SemanticTensorViewType) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        if isinstance(base_type, SemanticTileType) and attr == "shape":
            return SemanticShapeType(rank=base_type.rank)
        raise TypeError(f"unsupported attribute access '{attr}' in TileLang DSL v1")

    def _element_type_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if isinstance(base_type, (SemanticTensorViewType, SemanticTileType)):
            return SemanticSymbolExpr(
                namespace="pto",
                name=base_type.element_dtype.name,
                value=base_type.element_dtype,
                type=SemanticMetaType(kind="dtype"),
            )
        raise TypeError("unsupported attribute access 'element_type' in TileLang DSL v1")

    def _valid_shape_expr(self, base: SemanticExpr) -> SemanticExpr:
        base_type = base.type
        if not isinstance(base_type, (SemanticTensorViewType, SemanticTileType)):
            raise TypeError("unsupported attribute access 'valid_shape' in TileLang DSL v1")
        shape_access = SemanticAttributeAccess(
            base=base,
            attr="shape",
            type=SemanticShapeType(rank=base_type.rank),
        )
        elements = []
        for axis in range(base_type.rank):
            if isinstance(base, SemanticBindingRef) and isinstance(base.type, SemanticTensorViewType):
                self._ensure_tensor_shape_parameter(base.binding, axis)
            elements.append(
                SemanticSubscriptAccess(
                    base=shape_access,
                    index=SemanticLiteralExpr(value=axis, type=SemanticIndexType()),
                    type=SemanticIndexType(),
                )
            )
        return SemanticTupleExpr(
            elements=tuple(elements),
            type=SemanticTupleType(elements=tuple(SemanticIndexType() for _ in elements)),
        )

    def _subscript_type(self, base: SemanticExpr, index: SemanticExpr) -> SemanticType:
        if isinstance(base.type, SemanticShapeType):
            if not isinstance(index.type, SemanticIndexType):
                raise TypeError("shape subscript index must be an index value in TileLang DSL v1")
            if (
                isinstance(base, SemanticAttributeAccess)
                and isinstance(base.base, SemanticBindingRef)
                and isinstance(index, SemanticLiteralExpr)
                and isinstance(index.value, int)
            ):
                if index.value < 0 or index.value >= base.type.rank:
                    raise TypeError(
                        f"shape subscript index {index.value} is out of bounds for rank {base.type.rank}"
                    )
                if isinstance(base.base.type, SemanticTensorViewType):
                    self._ensure_tensor_shape_parameter(base.base.binding, index.value)
            return SemanticIndexType()
        if isinstance(base.type, SemanticTensorViewType):
            if not isinstance(index, SemanticTupleExpr):
                raise TypeError("TensorView slicing expects a tuple of slices in TileLang DSL v1")
            return self._tensor_slice_type(base.type, index)
        raise TypeError("unsupported subscript base in TileLang DSL v1")

    def _analyze_tile_vector_access(
        self,
        expr: FrontendExprNode,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, tuple[SemanticExpr, ...]]:
        if not self.node.advanced_enabled:
            raise TypeError(unsupported_feature_message(f"{context} tile indexing sugar"))
        if not isinstance(expr, FrontendSubscriptExpr):
            raise TypeError(
                f"{context} expects Tile element-indexing syntax in advanced TileLang DSL mode"
            )
        base = self._analyze_expr(expr.base, env, allow_outer_lookup=allow_outer_lookup)
        tile = self._require_tile_expr(base, context)
        indices = self._tile_vector_indices(
            expr.index,
            tile.type,
            env,
            allow_outer_lookup=allow_outer_lookup,
            context=context,
        )
        return base, indices

    def _tile_vector_indices(
        self,
        index_expr: FrontendExprNode,
        tile_type: SemanticTileType,
        env: dict[str, SemanticBinding],
        *,
        allow_outer_lookup: bool,
        context: str,
    ) -> tuple[SemanticExpr, ...]:
        if tile_type.rank == 1:
            if not isinstance(index_expr, FrontendSliceExpr):
                raise TypeError(f"{context} expects Tile[start:] syntax for rank-1 Tile values")
            if index_expr.stop is not None:
                raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
            if index_expr.step is not None:
                raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")
            if index_expr.start is None:
                return (SemanticLiteralExpr(value=0, type=SemanticIndexType()),)
            start = self._analyze_expr(index_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            self._require_index_typed_expr(start)
            return (start,)

        if tile_type.rank != 2 or tile_type.shape is None:
            raise TypeError(f"{context} currently only supports statically specialized rank-1 or rank-2 Tiles")
        if not isinstance(index_expr, FrontendTupleExpr) or len(index_expr.elements) != 2:
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")

        row_expr, col_expr = index_expr.elements
        if not isinstance(col_expr, FrontendSliceExpr):
            raise TypeError(f"{context} expects Tile[row, col:] syntax for rank-2 Tile values")
        if col_expr.stop is not None:
            raise TypeError(f"{context} does not support explicit slice stop in TileLang DSL advanced mode")
        if col_expr.step is not None:
            raise TypeError(f"{context} does not support stepped Tile vector slices in TileLang DSL advanced mode")

        row = self._analyze_expr(row_expr, env, allow_outer_lookup=allow_outer_lookup)
        self._require_index_typed_expr(row)
        if col_expr.start is None:
            col = SemanticLiteralExpr(value=0, type=SemanticIndexType())
        else:
            col = self._analyze_expr(col_expr.start, env, allow_outer_lookup=allow_outer_lookup)
            self._require_index_typed_expr(col)
        return (row, col)

    def _tensor_slice_type(
        self,
        tensor_type: SemanticTensorViewType,
        index: SemanticTupleExpr,
    ) -> SemanticTensorSliceType:
        if len(index.elements) != tensor_type.rank:
            raise TypeError(
                f"TensorView slice rank {len(index.elements)} does not match TensorView rank {tensor_type.rank}"
            )
        extents = []
        for axis, element in enumerate(index.elements):
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError(
                    f"TensorView slicing axis {axis} must use a Python slice in TileLang DSL v1"
                )
            self._require_optional_index_typed_expr(element.start)
            self._require_optional_index_typed_expr(element.stop)
            self._require_optional_index_typed_expr(element.step)

            start = self._static_index_value(element.start, default=0)
            stop = self._static_index_value(element.stop, default=None)
            step = self._static_index_value(element.step, default=1)
            if element.stop is None:
                raise TypeError("TensorView slicing requires explicit stop bounds in TileLang DSL v1")
            if start != 0:
                raise TypeError("TensorView slicing currently only supports zero-based starts in TileLang DSL v1")
            if element.step is not None and step is None:
                raise TypeError("TensorView slicing currently only supports unit stride in TileLang DSL v1")
            if step != 1:
                raise TypeError("TensorView slicing currently only supports unit stride in TileLang DSL v1")
            if stop is None:
                extent = None
            else:
                extent = stop - start
                if extent <= 0:
                    raise TypeError("TensorView slicing requires positive extents in TileLang DSL v1")
            extents.append(extent)
        return SemanticTensorSliceType(
            element_dtype=tensor_type.element_dtype,
            rank=tensor_type.rank,
            extents=tuple(extents),
        )

    def _normalize_tensor_slice(
        self,
        index: SemanticExpr,
        rank: int,
    ) -> tuple[SemanticSliceExpr, ...]:
        if not isinstance(index, SemanticTupleExpr):
            raise TypeError("TensorView slicing expects a tuple index in TileLang DSL v1")
        if len(index.elements) != rank:
            raise TypeError(f"TensorView slicing expects {rank} slice elements in TileLang DSL v1")
        slices = []
        for element in index.elements:
            if not isinstance(element, SemanticSliceExpr):
                raise TypeError("TensorView slicing only supports slice syntax in TileLang DSL v1")
            slices.append(element)
        return tuple(slices)

    def _binary_type(
        self,
        lhs: SemanticExpr,
        rhs: SemanticExpr,
        op: str,
    ) -> SemanticType:
        if op not in {"add", "sub", "mul", "floordiv"}:
            raise TypeError(f"unsupported binary operator '{op}' in TileLang DSL v1")
        if isinstance(lhs.type, SemanticIndexType) and isinstance(rhs.type, SemanticIndexType):
            return SemanticIndexType()
        raise TypeError("binary expressions currently only support index-typed operands")

    def _analyze_call_expr(
        self,
        namespace: str | None,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if namespace is None and name == "range":
            return SemanticCallExpr(namespace=namespace, name=name, args=args, type=None)
        if namespace != "pto":
            raise TypeError(
                f"call surface `{namespace + '.' if namespace else ''}{name}` is not supported in TileLang DSL v1 yet"
            )
        if name in DEFERRED_PTO_SURFACES:
            raise TypeError(deferred_surface_message(name))
        if name == "get_lanes":
            return self._analyze_get_lanes(args)
        if name == "make_mask":
            return self._analyze_make_mask(args)
        if name == "vlds":
            return self._analyze_vlds(args)
        if name in _UNARY_VECTOR_OPS:
            return self._analyze_unary_vector_op(name, args)
        if name in _BINARY_VECTOR_OPS:
            return self._analyze_binary_vector_op(name, args)
        if name in _VECTOR_SCALAR_OPS:
            return self._analyze_vector_scalar_op(name, args)
        raise TypeError(f"call surface `pto.{name}` is not supported in TileLang DSL v1 yet")

    def _analyze_make_mask(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError("pto.make_mask expects exactly 2 positional arguments in TileLang DSL v1")
        dtype_expr, value_expr = args
        dtype = self._require_dtype_symbol(dtype_expr, "pto.make_mask element type")
        if isinstance(value_expr, SemanticSymbolExpr) and value_expr.type.kind == "mask_pattern":
            return SemanticCallExpr(
                namespace="pto",
                name="make_mask",
                args=args,
                type=SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
            )
        self._require_tail_remaining_expr(value_expr, "pto.make_mask tail remaining")
        return SemanticCallExpr(
            namespace="pto",
            name="make_mask",
            args=args,
            type=SemanticTupleType(
                elements=(
                    SemanticMaskType(granularity=self._mask_granularity_for_dtype(dtype)),
                    _I32_TYPE,
                )
            ),
        )

    def _analyze_get_lanes(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) != 1:
            raise TypeError("pto.get_lanes expects exactly 1 positional argument in TileLang DSL v1")
        dtype = self._require_dtype_symbol(args[0], "pto.get_lanes dtype")
        return SemanticLiteralExpr(value=self._vreg_type_for_dtype(dtype).lanes, type=SemanticIndexType())

    def _analyze_vlds(self, args: tuple[SemanticExpr, ...]) -> SemanticExpr:
        if len(args) < 2:
            raise TypeError("pto.vlds expects at least 2 positional arguments in TileLang DSL v1")
        source, *indices = args
        tile = self._require_tile_expr(source, "pto.vlds source")
        for index in indices:
            self._require_index_typed_expr(index)
        return SemanticCallExpr(
            namespace="pto",
            name="vlds",
            args=(source, *indices),
            type=self._vreg_type_for_dtype(tile.type.element_dtype),
        )

    def _analyze_unary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 2:
            raise TypeError(f"pto.{name} expects exactly 2 positional arguments in TileLang DSL v1")
        value, mask = args
        vreg = self._require_vreg_expr(value, f"pto.{name} value")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_unary_dtype(name, vreg.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vreg)

    def _analyze_binary_vector_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        lhs_expr, rhs_expr, mask = args
        lhs = self._require_vreg_expr(lhs_expr, f"pto.{name} lhs")
        rhs = self._require_vreg_expr(rhs_expr, f"pto.{name} rhs")
        if lhs != rhs:
            raise TypeError(f"pto.{name} requires lhs/rhs vector types to match")
        self._require_mask_for_vreg(mask, lhs, f"pto.{name}")
        self._validate_binary_dtype(name, lhs.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=lhs)

    def _analyze_vector_scalar_op(
        self,
        name: str,
        args: tuple[SemanticExpr, ...],
    ) -> SemanticExpr:
        if len(args) != 3:
            raise TypeError(f"pto.{name} expects exactly 3 positional arguments in TileLang DSL v1")
        vector_expr, scalar_expr, mask = args
        vreg = self._require_vreg_expr(vector_expr, f"pto.{name} vector")
        scalar = self._require_scalar_expr(scalar_expr, f"pto.{name} scalar")
        if scalar.dtype != vreg.element_dtype:
            raise TypeError(f"pto.{name} scalar dtype must match vector element dtype")
        self._require_mask_for_vreg(mask, vreg, f"pto.{name}")
        self._validate_vector_scalar_dtype(name, vreg.element_dtype)
        return SemanticCallExpr(namespace="pto", name=name, args=args, type=vreg)

    def _require_dtype_symbol(self, expr: SemanticExpr, context: str) -> ScalarType:
        if not (
            isinstance(expr, SemanticSymbolExpr)
            and expr.type.kind == "dtype"
            and isinstance(expr.value, ScalarType)
        ):
            if (
                isinstance(expr, SemanticBindingRef)
                and isinstance(expr.type, SemanticMetaType)
                and expr.type.kind == "dtype"
                and isinstance(expr.binding.value, ScalarType)
            ):
                return expr.binding.value
            raise TypeError(f"{context} must be a TileLang scalar dtype symbol in TileLang DSL v1")
        return expr.value

    def _require_vreg_expr(self, expr: SemanticExpr, context: str) -> SemanticVRegType:
        if not isinstance(expr.type, SemanticVRegType):
            raise TypeError(f"{context} must be a vector register value in TileLang DSL v1")
        return expr.type

    def _require_scalar_expr(self, expr: SemanticExpr, context: str) -> SemanticScalarType:
        if not isinstance(expr.type, SemanticScalarType):
            raise TypeError(f"{context} must be a scalar value in TileLang DSL v1")
        return expr.type

    def _require_tail_remaining_expr(self, expr: SemanticExpr, context: str) -> None:
        if isinstance(expr.type, SemanticIndexType):
            return
        if isinstance(expr.type, SemanticScalarType) and expr.type.dtype.name == "i32":
            return
        raise TypeError(f"{context} must be an i32 or index value in TileLang DSL v1")

    def _require_mask_for_vreg(
        self,
        mask_expr: SemanticExpr,
        vreg_type: SemanticVRegType,
        context: str,
    ) -> None:
        if not isinstance(mask_expr.type, SemanticMaskType):
            raise TypeError(f"{context} requires a mask operand in TileLang DSL v1")
        expected = self._mask_granularity_for_dtype(vreg_type.element_dtype)
        if mask_expr.type.granularity != expected:
            raise TypeError(
                f"{context} requires mask granularity {expected} for vector dtype {vreg_type.element_dtype!r}"
            )

    def _require_matching_vector_pointer(
        self,
        vreg_type: SemanticVRegType,
        pointer_type: SemanticTileType,
        context: str,
    ) -> None:
        if pointer_type.element_dtype != vreg_type.element_dtype:
            raise TypeError(f"{context} requires destination Tile dtype to match vector dtype")

    def _mask_granularity_for_dtype(self, dtype: ScalarType) -> str:
        if dtype.name in {"f32", "i32"}:
            return "b32"
        if dtype.name in {"f16", "bf16", "i16"}:
            return "b16"
        if dtype.name == "i8":
            return "b8"
        raise TypeError(f"dtype `{dtype.name}` is not supported by make_mask/vector lowering in TileLang DSL v1")

    def _vreg_type_for_dtype(self, dtype: ScalarType) -> SemanticVRegType:
        byte_widths = {
            "i8": 1,
            "i16": 2,
            "i32": 4,
            "f16": 2,
            "bf16": 2,
            "f32": 4,
        }
        width = byte_widths.get(dtype.name)
        if width is None:
            raise TypeError(f"dtype `{dtype.name}` is not supported by vlds/vsts in TileLang DSL v1")
        return SemanticVRegType(element_dtype=dtype, lanes=256 // width)

    def _validate_unary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vexp" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vexp only supports f16/f32 in TileLang DSL v1")
        if name == "vrelu" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vrelu only supports f16/f32 in TileLang DSL v1")
        if name == "vnot" and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError("pto.vnot only supports integer vector dtypes in TileLang DSL v1")
        if name == "vabs" and dtype.name not in {"i8", "i16", "i32", "f16", "f32"}:
            raise TypeError("pto.vabs does not support this dtype in TileLang DSL v1")

    def _validate_binary_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdiv" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdiv only supports f16/f32 in TileLang DSL v1")
        if name in {"vand", "vor", "vxor"} and dtype.name not in {"i8", "i16", "i32"}:
            raise TypeError(f"pto.{name} only supports integer vector dtypes in TileLang DSL v1")
        if name == "vmul" and dtype.name not in {"i16", "i32", "f16", "f32"}:
            raise TypeError("pto.vmul only supports i16/i32/f16/f32 in TileLang DSL v1")
        if name in {"vadd", "vsub", "vmax", "vmin"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _validate_vector_scalar_dtype(self, name: str, dtype: ScalarType) -> None:
        if name == "vdivs" and dtype.name not in {"f16", "f32"}:
            raise TypeError("pto.vdivs only supports f16/f32 in TileLang DSL v1")
        if name in {"vadds", "vsubs", "vmuls", "vmaxs", "vmins"} and dtype.name not in {"i8", "i16", "i32", "f16", "bf16", "f32"}:
            raise TypeError(f"pto.{name} does not support this dtype in TileLang DSL v1")

    def _require_sync_pipe(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "pipe":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be a PIPE symbol or pipe string in TileLang DSL v1")

    def _require_sync_event(self, expr: SemanticExpr, context: str) -> str:
        if isinstance(expr, SemanticSymbolExpr) and expr.type.kind == "event":
            return expr.value.value
        if isinstance(expr, SemanticLiteralExpr) and isinstance(expr.type, SemanticMetaType) and expr.type.kind == "string":
            return expr.value
        raise TypeError(f"{context} must be an EVENT symbol or event string in TileLang DSL v1")

    def _require_loop_bound_type(self, ty: SemanticType) -> None:
        if isinstance(ty, (SemanticIndexType, SemanticScalarType)):
            return
        raise TypeError(f"loop bound must be scalar/index typed, got {ty!r}")

    def _require_condition_type(self, ty: SemanticType) -> None:
        if isinstance(ty, SemanticIndexType):
            return
        if isinstance(ty, SemanticScalarType):
            return
        raise TypeError(f"if condition must be scalar/index typed, got {ty!r}")

    def _merge_loop_carried_types(
        self,
        outer_type: SemanticType,
        final_type: SemanticType,
    ) -> SemanticType | None:
        if final_type == outer_type:
            return outer_type
        if (
            isinstance(outer_type, SemanticIndexType)
            and isinstance(final_type, SemanticScalarType)
            and final_type.dtype == i32
        ):
            return final_type
        if (
            isinstance(final_type, SemanticIndexType)
            and isinstance(outer_type, SemanticScalarType)
            and outer_type.dtype == i32
        ):
            return outer_type
        return None

    def _require_index_typed_expr(self, expr: SemanticExpr) -> None:
        if not isinstance(expr.type, SemanticIndexType):
            raise TypeError("slice bounds and vector offsets must be index-typed in TileLang DSL v1")

    def _static_index_value(self, expr: SemanticExpr | None, *, default: int | None) -> int | None:
        if expr is None:
            return default
        if not isinstance(expr, SemanticLiteralExpr) or not isinstance(expr.value, int):
            return None
        return expr.value

    def _require_optional_index_typed_expr(self, expr: SemanticExpr | None) -> None:
        if expr is None:
            return
        self._require_index_typed_expr(expr)


def analyze_frontend_kernel(node: FrontendKernelNode) -> SemanticKernel:
    """Normalize descriptor-owned AST into a lowering semantic model."""

    return _SemanticAnalyzer(node).analyze()


__all__ = [
    "SemanticAssignStmt",
    "SemanticAttributeAccess",
    "SemanticBinaryExpr",
    "SemanticBinding",
    "SemanticBindingRef",
    "SemanticCallExpr",
    "SemanticDmaLoadStmt",
    "SemanticDmaStoreStmt",
    "SemanticExpr",
    "SemanticExprStmt",
    "SemanticForStmt",
    "SemanticIfResult",
    "SemanticIfStmt",
    "SemanticIndexType",
    "SemanticKernel",
    "SemanticLiteralExpr",
    "SemanticMaskType",
    "SemanticParameter",
    "SemanticPipeBarrierStmt",
    "SemanticReturnStmt",
    "SemanticScalarType",
    "SemanticSetFlagStmt",
    "SemanticShapeType",
    "SemanticSliceExpr",
    "SemanticSliceType",
    "SemanticStmt",
    "SemanticVecscopeStmt",
    "SemanticStrictVecscopeStmt",
    "SemanticSubscriptAccess",
    "SemanticSymbolExpr",
    "SemanticTensorSliceExpr",
    "SemanticTensorSliceType",
    "SemanticTensorViewType",
    "SemanticTileBinding",
    "SemanticTileType",
    "SemanticTupleExpr",
    "SemanticTupleType",
    "SemanticType",
    "SemanticVRegType",
    "SemanticVectorStoreStmt",
    "SemanticWaitFlagStmt",
    "analyze_frontend_kernel",
]
