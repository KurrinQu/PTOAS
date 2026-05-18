# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""Tracing-time authored scalar operator lowering for runtime values."""

from __future__ import annotations

from mlir.dialects import arith
from mlir.ir import BF16Type, F16Type, F32Type, FloatAttr, IndexType, IntegerType


_INTEGER_BINARY_OPS = {
    "add": arith.AddIOp,
    "sub": arith.SubIOp,
    "mul": arith.MulIOp,
    "floordiv": arith.FloorDivSIOp,
    "mod": arith.RemSIOp,
}

_FLOAT_BINARY_OPS = {
    "add": arith.AddFOp,
    "sub": arith.SubFOp,
    "mul": arith.MulFOp,
    "truediv": arith.DivFOp,
}


def emit_runtime_binary_op(op_name: str, lhs, rhs):
    """Lower one authored runtime scalar binary operator."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind in {"index", "integer"}:
        op_cls = _INTEGER_BINARY_OPS.get(op_name)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for integer/index values")
        return op_cls(lhs, rhs).result
    if kind == "float":
        op_cls = _FLOAT_BINARY_OPS.get(op_name)
        if op_cls is None:
            raise TypeError(f"runtime scalar operator '{op_name}' is not supported for floating-point values")
        return op_cls(lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def emit_runtime_max(lhs, rhs):
    """Lower one authored runtime scalar max operation."""
    lhs, rhs, kind = normalize_runtime_binary_operands(lhs, rhs)
    if kind == "float":
        return arith.MaximumFOp(lhs, rhs).result
    if kind == "integer":
        return arith.MaxSIOp(lhs, rhs).result
    if kind == "index":
        cond = arith.CmpIOp(arith.CmpIPredicate.sge, lhs, rhs).result
        return arith.SelectOp(cond, lhs, rhs).result
    raise TypeError(f"unsupported runtime scalar operand category '{kind}'")


def normalize_runtime_binary_operands(lhs, rhs):
    lhs_is_value = _is_mlir_value(lhs)
    rhs_is_value = _is_mlir_value(rhs)

    if not lhs_is_value and not rhs_is_value:
        raise TypeError("runtime scalar operators require at least one traced runtime operand")

    if lhs_is_value and rhs_is_value:
        return _reconcile_typed_operands(lhs, rhs)

    anchor_type = lhs.type if lhs_is_value else rhs.type
    lhs = lhs if lhs_is_value else _materialize_literal(lhs, anchor_type)
    rhs = rhs if rhs_is_value else _materialize_literal(rhs, anchor_type)
    return _reconcile_typed_operands(lhs, rhs)


def _reconcile_typed_operands(lhs, rhs):
    lhs_type = lhs.type
    rhs_type = rhs.type

    if lhs_type == rhs_type:
        return lhs, rhs, classify_runtime_scalar_type(lhs_type)

    if IndexType.isinstance(lhs_type) and IntegerType.isinstance(rhs_type):
        rhs = arith.IndexCastOp(IndexType.get(), rhs).result
        return lhs, rhs, "index"

    if IntegerType.isinstance(lhs_type) and IndexType.isinstance(rhs_type):
        lhs = arith.IndexCastOp(IndexType.get(), lhs).result
        return lhs, rhs, "index"

    raise TypeError(
        "runtime scalar operators require matching scalar types or an index/integer pair; "
        f"got {lhs_type} and {rhs_type}"
    )


def _materialize_literal(value, anchor_type):
    if isinstance(value, bool):
        raise TypeError("runtime scalar operators do not accept bool literals")

    kind = classify_runtime_scalar_type(anchor_type)
    if kind == "float":
        return arith.ConstantOp(anchor_type, FloatAttr.get(anchor_type, float(value))).result

    if isinstance(value, float):
        raise TypeError(
            "runtime scalar operators cannot materialize a floating-point literal "
            f"against non-floating operand type {anchor_type}"
        )

    return arith.ConstantOp(anchor_type, int(value)).result


def classify_runtime_scalar_type(type_obj):
    if IndexType.isinstance(type_obj):
        return "index"
    if IntegerType.isinstance(type_obj):
        return "integer"
    if any(cls.isinstance(type_obj) for cls in (BF16Type, F16Type, F32Type)):
        return "float"
    raise TypeError(f"runtime scalar operators only support index/int/float values, got {type_obj}")


def _is_mlir_value(value) -> bool:
    return not isinstance(value, (bool, int, float)) and hasattr(value, "type")


__all__ = [
    "classify_runtime_scalar_type",
    "emit_runtime_binary_op",
    "emit_runtime_max",
    "normalize_runtime_binary_operands",
]
