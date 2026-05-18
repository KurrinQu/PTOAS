# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
Scalar arithmetic helpers – exposed as ``pto.scalar.*`` (or ``s = pto.scalar``).

Arithmetic helpers operate on raw ``mlir.ir.Value`` objects and emit the
corresponding arith dialect operations at the active insertion point.
Scalar memory helpers (`load` / `store`) also accept PTODSL surface-level
address views such as `tile[row, col]` and `tile.as_ptr() + offset`.
"""

from ._bootstrap import make_context  # ensure MLIR is on sys.path  # noqa: F401
from ._scalar_coercion import coerce_scalar_to_type
from ._runtime_scalar_ops import (
    classify_runtime_scalar_type,
    emit_runtime_max,
)
from ._surface_values import resolve_address_access, unwrap_surface_value, wrap_surface_value
from ._types import _resolve

from mlir.dialects import arith
from mlir.dialects import math
from mlir.dialects import pto as _pto
from mlir.ir import IndexType, MemRefType, Operation

_CMPI_PREDICATES = {
    "eq":  arith.CmpIPredicate.eq,
    "ne":  arith.CmpIPredicate.ne,
    "slt": arith.CmpIPredicate.slt,
    "sle": arith.CmpIPredicate.sle,
    "sgt": arith.CmpIPredicate.sgt,
    "sge": arith.CmpIPredicate.sge,
    "ult": arith.CmpIPredicate.ult,
    "ule": arith.CmpIPredicate.ule,
    "ugt": arith.CmpIPredicate.ugt,
    "uge": arith.CmpIPredicate.uge,
}


def muli(lhs, rhs):
    """arith.muli"""
    return wrap_surface_value(arith.MulIOp(unwrap_surface_value(lhs), unwrap_surface_value(rhs)).result)


def addi(lhs, rhs):
    """arith.addi"""
    return wrap_surface_value(arith.AddIOp(unwrap_surface_value(lhs), unwrap_surface_value(rhs)).result)


def subi(lhs, rhs):
    """arith.subi"""
    return wrap_surface_value(arith.SubIOp(unwrap_surface_value(lhs), unwrap_surface_value(rhs)).result)


def index_cast(type_or_val, val=None):
    """
    arith.index_cast.

    Two calling conventions::

        index_cast(result_type, value)   # explicit result type
        index_cast(value)                # result type = index (1-arg shorthand)
    """
    if val is None:
        # 1-arg form: cast to index
        return wrap_surface_value(arith.IndexCastOp(IndexType.get(), unwrap_surface_value(type_or_val)).result)
    return wrap_surface_value(arith.IndexCastOp(_resolve(type_or_val), unwrap_surface_value(val)).result)


def cmpi(pred: str, lhs, rhs):
    """
    arith.cmpi with a named predicate string.

    ``pred`` is one of: ``"eq"``, ``"ne"``, ``"slt"``, ``"sle"``,
    ``"sgt"``, ``"sge"``, ``"ult"``, ``"ule"``, ``"ugt"``, ``"uge"``.
    """
    predicate = _CMPI_PREDICATES.get(pred)
    if predicate is None:
        raise ValueError(
            f"Unknown cmpi predicate '{pred}'; known: {list(_CMPI_PREDICATES)}"
        )
    return wrap_surface_value(
        arith.CmpIOp(predicate, unwrap_surface_value(lhs), unwrap_surface_value(rhs)).result
    )


def cmpi_sgt(lhs, rhs):
    """arith.cmpi sgt (signed greater-than)."""
    return wrap_surface_value(arith.CmpIOp(
        arith.CmpIPredicate.sgt,
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ).result)


def select(cond, true_val, false_val):
    """arith.select"""
    return wrap_surface_value(arith.SelectOp(
        unwrap_surface_value(cond),
        unwrap_surface_value(true_val),
        unwrap_surface_value(false_val),
    ).result)


def max(lhs, rhs):
    """Runtime scalar maximum across float / integer / index values."""
    return wrap_surface_value(emit_runtime_max(
        unwrap_surface_value(lhs),
        unwrap_surface_value(rhs),
    ))


def exp(value):
    """Runtime scalar exponential for floating-point values."""
    raw_value = unwrap_surface_value(value)
    kind = classify_runtime_scalar_type(raw_value.type)
    if kind != "float":
        raise TypeError(f"scalar.exp(...) expects a floating-point runtime scalar, got {raw_value.type}")
    return wrap_surface_value(math.ExpOp(raw_value).result)


def load(ptr_or_ref, offset=None):
    """Load one scalar element from a PTODSL address view or tile element."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    result_type = _infer_buffer_element_type(buffer_value.type)
    return wrap_surface_value(Operation.create(
        "pto.load",
        results=[result_type],
        operands=[buffer_value, index_value],
    ).results[0])


def store(value, ptr_or_ref, offset=None):
    """Store one scalar element to a PTODSL address view or tile element."""
    buffer_value, index_value = resolve_address_access(ptr_or_ref, offset)
    elem_type = _infer_buffer_element_type(buffer_value.type)
    Operation.create(
        "pto.store",
        operands=[buffer_value, index_value, coerce_scalar_to_type(value, elem_type, context="scalar.store(...)")],
    )


def _infer_buffer_element_type(buffer_type):
    try:
        return _pto.PtrType(buffer_type).element_type
    except Exception:
        return MemRefType(buffer_type).element_type


__all__ = [
    "muli", "addi", "subi",
    "index_cast",
    "cmpi", "cmpi_sgt",
    "select",
    "max", "exp",
    "load", "store",
]
