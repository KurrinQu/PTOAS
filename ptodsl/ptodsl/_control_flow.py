# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Control-flow context managers for PTO kernels.

All CMs work with the current MLIR insertion point; no context threading needed.

Public API
──────────
``vecscope()``            – ``pto.vecscope { … }``
``for_(lo, hi, step, *, iter_args)``
                          – ``scf.for`` with optional iter_args or named carry state
``if_(cond, *, results)`` – ``scf.if`` with optional results + else
``yield_(*vals)``         – ``scf.yield``
"""

from ._bootstrap import make_context  # noqa: F401
from ._runtime_index_ops import coerce_runtime_index
from ._tracing.active import current_session
from ._surface_values import unwrap_surface_value, wrap_like_surface_value, wrap_surface_value
from ._types import _resolve

from mlir.dialects import pto as _pto, scf
from mlir.ir import InsertionPoint


# ── vecscope ──────────────────────────────────────────────────────────────────

class _VecScopeCM:
    """Context manager for ``pto.vecscope { … }``."""

    def __enter__(self):
        self._op = _pto.VecScopeOp()
        self._block = self._op.body.blocks.append()
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()
        return None

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


def vecscope() -> _VecScopeCM:
    """Return a context manager that emits ``pto.vecscope { … }``."""
    return _VecScopeCM()


# ── for_ ──────────────────────────────────────────────────────────────────────

class LoopHandle:
    """
    Handle for a ``scf.for`` loop with iter_args.

    Attributes available *after* the ``with pto.for_(…) as loop:`` block::

        loop.iv         – induction variable
        loop.iter_args  – tuple of inner (mutable) SSA values
        loop.results    – tuple of ForOp results (after loop exit)
    """

    def __init__(self, for_op, *, iter_arg_templates=()):
        self._op = for_op
        self._iter_arg_templates = tuple(iter_arg_templates)

    @property
    def iv(self):
        return wrap_surface_value(self._op.induction_variable)

    @property
    def iter_args(self):
        return tuple(
            wrap_like_surface_value(template, value)
            for template, value in zip(self._iter_arg_templates, self._op.inner_iter_args)
        )

    @property
    def results(self):
        return tuple(
            wrap_like_surface_value(template, value)
            for template, value in zip(self._iter_arg_templates, self._op.results)
        )


class _ForCM:
    def __init__(self, start, stop, step, iter_args):
        self._start = start
        self._stop = stop
        self._step = step
        self._iter_arg_templates = tuple(iter_args) if iter_args is not None else ()
        self._iter_args = [unwrap_surface_value(value) for value in self._iter_arg_templates]
        self._for_op = None
        self._ip = None

    def __enter__(self):
        self._for_op = scf.ForOp(
            _coerce_index(self._start),
            _coerce_index(self._stop),
            _coerce_index(self._step),
            self._iter_args if self._iter_args else None,
        )
        self._ip = InsertionPoint(self._for_op.body)
        self._ip.__enter__()
        if not self._iter_args:
            return wrap_surface_value(self._for_op.induction_variable)
        return LoopHandle(self._for_op, iter_arg_templates=self._iter_arg_templates)

    def __exit__(self, *exc):
        if not self._iter_args:
            scf.YieldOp([])
        self._ip.__exit__(*exc)


def for_(start, stop, *, step, iter_args=None):
    """
    ``scf.for`` context manager.

    Without ``iter_args`` – yields the induction variable; ``scf.yield`` is
    inserted automatically::

        with pto.for_(c0, c16, step=c1) as i:
            ...

    With ``iter_args`` – yields a :class:`LoopHandle`; the caller must emit
    ``pto.yield_(…)`` before the block closes::

        with pto.for_(c0, c128, step=c64, iter_args=(a, b)) as loop:
            x, y = loop.iter_args
            ...
            pto.yield_(nx, ny)
        fa, fb = loop.results

    Named carry state is expressed with ``.carry(...)``::

        loop = pto.for_(c0, c128, step=c64).carry(acc=tile)
        with loop:
            cur = loop.acc
            loop.update(acc=cur)
        out = loop.final("acc")
    """
    return _ForBuilder(start, stop, step, iter_args)


class _CarryLoopStateView:
    def __init__(self, names, values):
        self._names = tuple(names)
        self._values = dict(zip(self._names, values))

    def __getattr__(self, name):
        try:
            return self._values[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class _CarryForCM(_ForCM):
    def __init__(self, start, stop, step, state_items):
        self._state_items = tuple(state_items)
        self._state_names = tuple(name for name, _ in self._state_items)
        self._state_templates = tuple(value for _, value in self._state_items)
        self._session = None
        self._session_frame = None
        super().__init__(start, stop, step, self._state_templates)
        self._yield_values = None
        self._entered = False

    def __enter__(self):
        self._session = current_session()
        if self._session is not None:
            self._session_frame = self._session.begin_carry_loop(
                self._start,
                self._stop,
                self._step,
                self._state_items,
            )
            self._for_op = self._session_frame.for_op
            handle = LoopHandle(self._for_op, iter_arg_templates=self._state_templates)
        else:
            handle = super().__enter__()
        self._entered = True
        self._yield_values = None
        self._loop_handle = handle
        self._state = _CarryLoopStateView(self._state_names, handle.iter_args)
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._session_frame is not None:
                self._session.finish_carry_loop(self._session_frame, exc_type, exc, tb)
                return None
            if exc_type is None:
                if self._yield_values is None:
                    raise RuntimeError(
                        "pto.for_(...).carry(...) requires loop.update(...) before leaving the loop body"
                    )
                scf.YieldOp(self._yield_values)
            return super().__exit__(exc_type, exc, tb)
        finally:
            self._entered = False
            self._session = None
            self._session_frame = None

    @property
    def iv(self):
        if not self._entered:
            raise RuntimeError("loop.iv is only available inside an active carry loop body")
        return self._loop_handle.iv

    def __getattr__(self, name):
        if name in self._state_names:
            if not self._entered:
                raise RuntimeError(f"loop.{name} is only available inside an active carry loop body")
            return getattr(self._state, name)
        raise AttributeError(name)

    def update(self, **kwargs):
        if not self._entered:
            raise RuntimeError("loop.update(...) may only be called inside the loop body")
        if self._session_frame is not None:
            self._session.update_carry_loop(self._session_frame, **kwargs)
            return
        missing = [name for name in self._state_names if name not in kwargs]
        extra = [name for name in kwargs if name not in self._state_names]
        if missing or extra:
            pieces = []
            if missing:
                pieces.append(f"missing: {', '.join(missing)}")
            if extra:
                pieces.append(f"unexpected: {', '.join(extra)}")
            raise RuntimeError("loop.update(...) must match carry names exactly; " + "; ".join(pieces))
        if self._yield_values is not None:
            raise RuntimeError("loop.update(...) may only be called once per loop body")
        self._yield_values = [
            unwrap_surface_value(kwargs[name])
            for name in self._state_names
        ]

    def final(self, name):
        if self._for_op is None:
            raise RuntimeError("loop.final(...) is only available after the loop has been built")
        try:
            index = self._state_names.index(name)
        except ValueError as exc:
            raise RuntimeError(
                f"loop.final(...) requested unknown carry state '{name}'; "
                f"expected one of: {', '.join(self._state_names)}"
            ) from exc
        return wrap_like_surface_value(self._state_templates[index], self._for_op.results[index])


class _ForBuilder:
    def __init__(self, start, stop, step, iter_args=None):
        self._start = start
        self._stop = stop
        self._step = step
        self._iter_args = iter_args

    def __enter__(self):
        self._cm = _ForCM(self._start, self._stop, self._step, self._iter_args)
        return self._cm.__enter__()

    def __exit__(self, *exc):
        return self._cm.__exit__(*exc)

    def carry(self, **kwargs):
        if self._iter_args is not None:
            raise RuntimeError("for_(..., iter_args=...) cannot be combined with .carry(...)")
        if not kwargs:
            raise ValueError("carry(...) requires at least one named loop-carried value")
        for name in kwargs:
            if not isinstance(name, str) or not name:
                raise TypeError("carry(...) names must be non-empty strings")
        return _CarryForCM(self._start, self._stop, self._step, tuple(kwargs.items()))


def _coerce_index(value):
    raw_value = unwrap_surface_value(value)
    return coerce_runtime_index(raw_value, context="pto.for_(...) loop bound")


# ── if_ ───────────────────────────────────────────────────────────────────────

class _BlockCM:
    """Enters the InsertionPoint of a single block for ``with br.then_:`` style."""

    def __init__(self, block):
        self._block = block
        self._ip = None

    def __enter__(self):
        self._ip = InsertionPoint(self._block)
        self._ip.__enter__()

    def __exit__(self, *exc):
        self._ip.__exit__(*exc)


class BranchHandle:
    """
    Handle for ``scf.if`` with results and an else branch.

    Usage::

        with pto.if_(cond, results=(vf32, vf32)) as br:
            with br.then_:
                ...
                pto.yield_(a, b)
            with br.else_:
                pto.yield_(c, d)
        x, y = br.results
    """

    def __init__(self, if_op):
        self._op = if_op
        self.then_ = _BlockCM(if_op.then_block)
        self.else_ = _BlockCM(if_op.else_block)

    @property
    def results(self):
        return tuple(wrap_surface_value(result) for result in self._op.results)


class _IfCM:
    def __init__(self, cond, result_types):
        self._cond = cond
        self._result_types = [_resolve(t) for t in result_types] if result_types else []
        self._if_op = None
        self._ip = None

    def __enter__(self):
        cond = unwrap_surface_value(self._cond)
        if self._result_types:
            # if/else with results: create IfOp but don't enter any block;
            # the caller manages blocks via br.then_ / br.else_
            self._if_op = scf.IfOp(cond, self._result_types, hasElse=True)
            return BranchHandle(self._if_op)
        else:
            # simple if without results: enter then_block automatically
            self._if_op = scf.IfOp(cond)
            self._ip = InsertionPoint(self._if_op.then_block)
            self._ip.__enter__()
            return None

    def __exit__(self, *exc):
        if not self._result_types:
            scf.YieldOp([])
            self._ip.__exit__(*exc)
        # for if/else with results: blocks are managed by BranchHandle; nothing to do


def if_(cond, *, results=None) -> _IfCM:
    """
    ``scf.if`` context manager.

    Without ``results`` – simple if with no else; ``scf.yield`` is inserted
    automatically::

        with pto.if_(has_rows):
            ...

    With ``results`` – if/else pair that produces SSA values; the caller must
    manage ``br.then_`` and ``br.else_`` and emit ``pto.yield_(…)`` in each::

        with pto.if_(has_chunk, results=(vf32, vf32)) as br:
            with br.then_:
                ...
                pto.yield_(merged_max, merged_sum)
            with br.else_:
                pto.yield_(running_max, running_sum)
        x, y = br.results
    """
    return _IfCM(cond, results)


# ── yield_ ────────────────────────────────────────────────────────────────────

def yield_(*vals):
    """Emit ``scf.yield`` with the given values."""
    scf.YieldOp([unwrap_surface_value(value) for value in vals])


__all__ = [
    "vecscope", "LoopHandle", "BranchHandle",
    "for_", "if_", "yield_",
]
