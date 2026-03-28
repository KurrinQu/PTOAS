#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from pathlib import Path
from typing import Any

from family_dsl import ensure_catalog_sync, load_family_dsl, load_snippet_contracts

SCRIPT_DIR = Path(__file__).resolve().parent
SKELETON_DIR = SCRIPT_DIR / "skeletons"
SNIPPET_DIR = SCRIPT_DIR / "families" / "snippets"
CATALOG_PATH = SKELETON_DIR / "catalog.json"
MODULE_TEMPLATE_PATH = SKELETON_DIR / "module.tmpl.mlir"
GENERATED_FILE_BANNER = "// AUTO-GENERATED: do not edit directly."
A5_VECTOR_LENGTH_BITS = 256 * 8

TILE_TYPE_FMT = (
    "!pto.tile_buf<loc=vec, dtype={dtype}, rows={rows}, cols={cols}, v_row=?, "
    "v_col=?, blayout=row_major, slayout=none_box, fractal=512, pad=0>"
)
MEMREF_TYPE_FMT = "memref<?x?x{dtype}, strided<[{row_stride}, 1], offset: 0>, #pto.address_space<vec>>"

SETUP_BEGIN = "// SNIPPET_SETUP_BEGIN"
SETUP_END = "// SNIPPET_SETUP_END"
COMPUTE_BEGIN = "// SNIPPET_COMPUTE_BEGIN"
COMPUTE_END = "// SNIPPET_COMPUTE_END"

FLOAT_BINARY_CORE_OPS = {
    "arith.addf",
    "arith.subf",
    "arith.mulf",
    "arith.divf",
    "arith.maximumf",
    "arith.minimumf",
}

FLOAT_COMPARE_PREDICATES = {
    "EQ": "oeq",
    "NE": "one",
    "LT": "olt",
    "LE": "ole",
    "GT": "ogt",
    "GE": "oge",
}

SIGNED_INT_COMPARE_PREDICATES = {
    "EQ": "eq",
    "NE": "ne",
    "LT": "slt",
    "LE": "sle",
    "GT": "sgt",
    "GE": "sge",
}

UNSIGNED_INT_COMPARE_PREDICATES = {
    "EQ": "eq",
    "NE": "ne",
    "LT": "ult",
    "LE": "ule",
    "GT": "ugt",
    "GE": "uge",
}


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def load_family_doc() -> dict[str, Any]:
    family_doc = load_family_dsl()
    catalog = json.loads(load_text(CATALOG_PATH))
    ensure_catalog_sync(catalog, family_doc)
    return family_doc


def is_generated_concrete_file(path: Path) -> bool:
    if path.suffix != ".mlir" or not path.is_file():
        return False
    lines = path.read_text(encoding="utf-8").splitlines()
    return GENERATED_FILE_BANNER in lines[:4]


def list_existing_generated_outputs() -> list[Path]:
    return sorted(
        path for path in SCRIPT_DIR.glob("*.mlir") if is_generated_concrete_file(path)
    )


def find_stale_generated_outputs(outputs: dict[Path, str]) -> list[Path]:
    expected = set(outputs)
    return [path for path in list_existing_generated_outputs() if path not in expected]


def validate_expected_output(path: Path, content: str) -> None:
    unresolved = sorted(set(re.findall(r"@@[A-Z0-9_]+@@", content)))
    if unresolved:
        unresolved_list = ", ".join(unresolved)
        raise ValueError(
            f"generated output {path.relative_to(SCRIPT_DIR)} still contains unresolved placeholders: {unresolved_list}"
        )


def replace_all(template: str, mapping: dict[str, str]) -> str:
    out = template
    for key, value in mapping.items():
        out = out.replace(f"@@{key}@@", value)
    return out


def sanitize_symbol(value: str) -> str:
    chars: list[str] = []
    for ch in value:
        if ch.isalnum() or ch == "_":
            chars.append(ch.lower())
        else:
            chars.append("_")
    return "".join(chars).strip("_")


def lower_name(value: str) -> str:
    return value.lower()


def carrier_ir_dtype(dtype: str) -> str:
    if dtype.startswith("u") and dtype[1:].isdigit():
        return f"i{dtype[1:]}"
    return dtype


def tile_type(dtype: str, rows: int = 32, cols: int = 32) -> str:
    return TILE_TYPE_FMT.format(dtype=carrier_ir_dtype(dtype), rows=rows, cols=cols)


def scalar_type(dtype: str) -> str:
    return carrier_ir_dtype(dtype)


def scalar_literal(dtype: str, value: str) -> str:
    if value == "zero":
        return "0.0" if is_float_dtype(dtype) else "0"
    if value == "one":
        return "1.0" if is_float_dtype(dtype) else "1"
    if value == "neg_one":
        return "-1.0" if is_float_dtype(dtype) else "-1"
    raise ValueError(f"unsupported scalar literal '{value}' for dtype {dtype}")


def a5_simd_lanes(dtype: str) -> int:
    bitwidth = 0
    ir_dtype = carrier_ir_dtype(dtype)
    if ir_dtype in {"f16", "bf16"}:
        bitwidth = 16
    elif ir_dtype == "f32":
        bitwidth = 32
    elif ir_dtype.startswith("i") and ir_dtype[1:].isdigit():
        bitwidth = int(ir_dtype[1:])
    else:
        raise ValueError(f"unsupported dtype '{dtype}' for A5 SIMD lane inference")
    if bitwidth <= 0 or A5_VECTOR_LENGTH_BITS % bitwidth != 0:
        raise ValueError(
            f"dtype '{dtype}' has unsupported bitwidth {bitwidth} for {A5_VECTOR_LENGTH_BITS}-bit A5 vector length"
        )
    return A5_VECTOR_LENGTH_BITS // bitwidth


def vector_type(dtype: str, lanes: int | None = None) -> str:
    if lanes is None:
        lanes = a5_simd_lanes(dtype)
    return f"vector<{lanes}x{carrier_ir_dtype(dtype)}>"


def mask_vector_type(lanes: int) -> str:
    return f"vector<{lanes}xi1>"


def memref_type(dtype: str, cols: int = 32) -> str:
    return MEMREF_TYPE_FMT.format(dtype=carrier_ir_dtype(dtype), row_stride=cols)


def splat_shuffle_mask(lanes: int = 64, source_lane: int = 0) -> str:
    return ", ".join(str(source_lane) for _ in range(lanes))


def dense_literal(dtype: str, value: str) -> str:
    if value == "zero":
        return "dense<0.0>" if is_float_dtype(dtype) else "dense<0>"
    if value == "one":
        return "dense<1.0>" if is_float_dtype(dtype) else "dense<1>"
    if value == "neg_one":
        return "dense<-1.0>" if is_float_dtype(dtype) else "dense<-1>"
    raise ValueError(f"unsupported dense literal '{value}' for dtype {dtype}")


def is_float_dtype(dtype: str) -> bool:
    return dtype.startswith("f") or dtype == "bf16"


def is_unsigned_dtype(dtype: str) -> bool:
    return dtype.startswith("u")


def resolve_compare_core_op(dtype: str) -> str:
    return "arith.cmpf" if is_float_dtype(dtype) else "arith.cmpi"


def resolve_compare_predicate(dtype: str, cmp_mode: str) -> str:
    cmp_mode = cmp_mode.upper()
    if is_float_dtype(dtype):
        predicate_map = FLOAT_COMPARE_PREDICATES
    elif is_unsigned_dtype(dtype):
        predicate_map = UNSIGNED_INT_COMPARE_PREDICATES
    else:
        predicate_map = SIGNED_INT_COMPARE_PREDICATES
    if cmp_mode not in predicate_map:
        raise ValueError(f"unsupported compare mode '{cmp_mode}' for dtype {dtype}")
    return predicate_map[cmp_mode]


EXEC_MODE_BODY_KINDS = {
    "abs",
    "lrelu",
    "neg",
    "recip",
    "relu",
    "rsqrt",
    "sqrt",
}


def exec_mode_attr(op_info: dict[str, Any], dtype: str) -> str:
    core_op = op_info["core_op"]
    body_kind = op_info.get("body_kind", "")
    if is_float_dtype(dtype) and (
        any(op_name in core_op for op_name in FLOAT_BINARY_CORE_OPS)
        or "math.exp" in core_op
        or "math.log" in core_op
        or "math.sqrt" in core_op
        or body_kind in EXEC_MODE_BODY_KINDS
    ):
        return ' {pto.simd.exec_mode = "MODE_ZEROING"}'
    return ""


def tile_shape_from_metadata(
    metadata: dict[str, Any],
    key: str,
    default_rows: int = 32,
    default_cols: int = 32,
) -> tuple[int, int]:
    shape = metadata.get(key)
    if not isinstance(shape, dict):
        return default_rows, default_cols
    rows = int(shape.get("rows", default_rows))
    cols = int(shape.get("cols", default_cols))
    return rows, cols


def build_arg_tile_shapes(
    parameter_roles: list[dict[str, Any]],
    metadata: dict[str, Any],
    input_tile_shape: tuple[int, int],
    result_tile_shape: tuple[int, int],
) -> list[tuple[int, int] | None]:
    shape_by_role = metadata.get("tile_shapes_by_role", {})
    arg_tile_shapes: list[tuple[int, int] | None] = []
    for role in parameter_roles:
        if role["kind"] != "tile":
            arg_tile_shapes.append(None)
            continue
        if role["io"] == "output":
            arg_tile_shapes.append(result_tile_shape)
            continue
        custom_shape = shape_by_role.get(role["name"], {})
        if isinstance(custom_shape, dict):
            rows = int(custom_shape.get("rows", input_tile_shape[0]))
            cols = int(custom_shape.get("cols", input_tile_shape[1]))
            arg_tile_shapes.append((rows, cols))
            continue
        arg_tile_shapes.append(input_tile_shape)
    return arg_tile_shapes


def build_arg_decls(
    parameter_roles: list[dict[str, Any]],
    input_dtype: str,
    result_dtype: str | None,
    arg_tile_shapes: list[tuple[int, int] | None],
    arg_types_by_role: dict[str, str] | None = None,
) -> str:
    arg_types_by_role = arg_types_by_role or {}
    decls: list[str] = []
    tile_index = 0
    scalar_index = 0
    for idx, role in enumerate(parameter_roles):
        if role["kind"] == "tile":
            if role["io"] == "output":
                name = "dst"
                dtype = arg_types_by_role.get(role["name"], result_dtype or input_dtype)
                rows, cols = arg_tile_shapes[idx] or (32, 32)
            else:
                name = f"src{tile_index}"
                dtype = arg_types_by_role.get(role["name"], input_dtype)
                rows, cols = arg_tile_shapes[idx] or (32, 32)
                tile_index += 1
            decls.append(f"      %{name}: {tile_type(dtype, rows, cols)}")
            continue
        name = "scalar" if scalar_index == 0 else f"scalar{scalar_index}"
        scalar_index += 1
        decls.append(
            f"      %{name}: {arg_types_by_role.get(role['name'], scalar_type(input_dtype))}"
        )
    return ",\n".join(decls)


def resolve_arg_type(
    role: dict[str, Any],
    input_dtype: str,
    result_dtype: str | None,
    arg_types_by_role: dict[str, str],
) -> str:
    if role["kind"] == "tile":
        default_dtype = (
            result_dtype or input_dtype if role["io"] == "output" else input_dtype
        )
        return arg_types_by_role.get(role["name"], default_dtype)
    return arg_types_by_role.get(role["name"], scalar_type(input_dtype))


def build_match_attrs(arg_roles: list[str]) -> str:
    lines: list[str] = []
    for idx, role in enumerate(arg_roles):
        if role != "tile":
            continue
        lines.extend(
            [
                f"        pto.oplib.match.arg{idx}.rows = -1 : i64,",
                f"        pto.oplib.match.arg{idx}.cols = -1 : i64,",
                f'        pto.oplib.match.arg{idx}.blayout = "row_major",',
                f'        pto.oplib.match.arg{idx}.slayout = "any",',
                f"        pto.oplib.match.arg{idx}.fractal = -1 : i64,",
            ]
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_extra_template_attrs(op_info: dict[str, Any]) -> str:
    metadata = op_info.get("metadata", {})
    template_attrs = metadata.get("template_attrs", {})
    if not isinstance(template_attrs, dict):
        raise ValueError(
            f"template_attrs for op {op_info.get('name', '<unknown>')} must be a dict"
        )

    lines: list[str] = []
    for attr_name, attr_value in template_attrs.items():
        if isinstance(attr_value, bool):
            literal = "true" if attr_value else "false"
        elif isinstance(attr_value, int):
            literal = f"{attr_value} : i64"
        else:
            escaped = str(attr_value).replace("\\", "\\\\").replace('"', '\\"')
            literal = f'"{escaped}"'
        lines.append(f"        {attr_name} = {literal},")
    return "\n".join(lines) + ("\n" if lines else "")


def build_func_name(
    pattern: dict[str, Any],
    family: dict[str, Any],
    op_info: dict[str, Any],
    variant_id: str,
    dtype: str,
) -> str:
    func_name_format = op_info.get(
        "func_name_format", family["metadata"].get("func_name_format")
    )
    if func_name_format:
        return func_name_format.format(
            pattern_id=pattern["id"],
            family_id=family["family"],
            kind=family["kind"],
            op=op_info["name"],
            variant_id=variant_id,
            dtype=dtype,
        )
    return sanitize_symbol(
        f"__pto_oplib_{pattern['id']}_{family['kind']}_{op_info['name']}_{variant_id}_{dtype}"
    )


def family_scalar_index(family: dict[str, Any]) -> int | None:
    for index, role in enumerate(family["parameter_roles"]):
        if role["kind"] == "scalar":
            return index
    return None


def expand_ops(family: dict[str, Any]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for op in family["ops"]:
        variants = op.get("variants", [])
        if not variants:
            expanded.append(op)
            continue
        for variant in variants:
            entry = dict(op)
            entry.pop("variants", None)
            entry["variant_id"] = variant["id"]
            for key, value in variant.items():
                if key in {"id", "body", "metadata", "request_keys"}:
                    continue
                entry[key] = value
            body = variant.get("body", {})
            entry.update(body)
            metadata = variant.get("metadata", {})
            entry.update(metadata)
            request_keys = variant.get("request_keys")
            if request_keys:
                entry["request_keys"] = request_keys
            expanded.append(entry)
    return expanded


def snippet_relpath(op_info: dict[str, Any]) -> str:
    snippet = str(op_info.get("snippet", "")).strip()
    if not snippet:
        raise ValueError(f"active generated op {op_info['name']} is missing snippet")
    return snippet


def load_snippet_sections(relative_path: str) -> tuple[str, str]:
    path = SNIPPET_DIR / relative_path
    text = load_text(path)
    if SETUP_BEGIN in text or COMPUTE_BEGIN in text:
        setup = extract_marked_section(text, SETUP_BEGIN, SETUP_END)
        compute = extract_marked_section(text, COMPUTE_BEGIN, COMPUTE_END)
    else:
        setup = ""
        compute = text.strip()
    return setup, compute


def extract_marked_section(text: str, begin: str, end: str) -> str:
    if begin not in text:
        return ""
    pattern = re.compile(rf"{re.escape(begin)}\n(.*?){re.escape(end)}", re.DOTALL)
    match = pattern.search(text)
    if not match:
        raise ValueError(f"snippet section {begin} is missing closing marker {end}")
    return match.group(1).strip()


def indent_block(text: str, prefix: str) -> str:
    body = text.strip()
    if not body:
        return ""
    return "".join(f"{prefix}{line}\n" if line else "\n" for line in body.splitlines())


def validate_rendered_snippet(
    relative_path: str,
    contract: dict[str, Any],
    setup_text: str,
    compute_text: str,
) -> None:
    combined = "\n".join(part for part in (setup_text, compute_text) if part)
    for op_name in contract["forbidden_ops"]:
        if op_name in combined:
            raise ValueError(
                f"snippet {relative_path} uses forbidden op for contract {contract['id']}: {op_name}"
            )

    result_ssa = contract["result_ssa"]
    if not re.search(rf"^\s*{re.escape(result_ssa)}\s*=", compute_text, re.MULTILINE):
        raise ValueError(
            f"snippet {relative_path} must define {result_ssa} in compute section"
        )

    for ssa_name in contract["snippet_must_not_define"]:
        if ssa_name == result_ssa:
            continue
        if re.search(rf"^\s*{re.escape(ssa_name)}\s*=", combined, re.MULTILINE):
            raise ValueError(
                f"snippet {relative_path} redefines reserved SSA {ssa_name} for contract {contract['id']}"
            )


def build_snippet_mapping(
    op_info: dict[str, Any],
    dtype: str,
    result_dtype: str | None,
    scalar_arg_type: str,
    cmp_predicate: str,
    rhs_value: str,
) -> dict[str, str]:
    operand_order = op_info.get("operand_order", "tile_scalar")
    snippet_lhs = "%lhs"
    snippet_rhs = "%scalarVec"
    if operand_order == "scalar_tile":
        snippet_lhs, snippet_rhs = snippet_rhs, snippet_lhs
    vector_lanes = a5_simd_lanes(dtype)
    input_vector_ty = vector_type(dtype)
    result_vector_ty = vector_type(result_dtype or dtype)
    return {
        "CORE_OP": op_info["core_op"],
        "REDUCE_KIND": reduction_kind_from_core_op(op_info["core_op"]),
        "SCALAR_TYPE": scalar_type(result_dtype or dtype),
        "SCALAR_ARG_TYPE": scalar_arg_type,
        "VECTOR_TYPE": result_vector_ty,
        "INPUT_VECTOR_TYPE": input_vector_ty,
        "RESULT_VECTOR_TYPE": result_vector_ty,
        "MASK_VECTOR_TYPE": mask_vector_type(vector_lanes),
        "CMP_PREDICATE": cmp_predicate,
        "RHS_VALUE": rhs_value,
        "SNIPPET_LHS": snippet_lhs,
        "SNIPPET_RHS": snippet_rhs,
        "EXEC_MODE_ATTR": exec_mode_attr(op_info, dtype),
        "ZERO_LITERAL": dense_literal(result_dtype or dtype, "zero"),
        "ONE_LITERAL": dense_literal(result_dtype or dtype, "one"),
        "NEG_ONE_LITERAL": dense_literal(result_dtype or dtype, "neg_one"),
    }


def reduction_kind_from_core_op(core_op: str) -> str:
    reduce_kinds = {
        "arith.addf": "add",
        "arith.addi": "add",
        "arith.maximumf": "maximumf",
        "arith.minimumf": "minimumf",
        "arith.maxsi": "maxsi",
        "arith.minsi": "minsi",
        "arith.maxui": "maxui",
        "arith.minui": "minui",
    }
    return reduce_kinds.get(core_op, core_op)


def format_mlir_attr_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    raise ValueError(f"unsupported matcher attr value type: {type(value).__name__}")


def build_variant_match_attrs(
    family: dict[str, Any],
    variant_matcher: dict[str, Any],
) -> str:
    lines: list[str] = []
    prefix = "variant_axis.matcher."
    for key_info in family["matcher_keys"]:
        if key_info["location"] != "template_attr":
            continue
        source = str(key_info.get("source", ""))
        if not source.startswith(prefix):
            continue
        matcher_key = source[len(prefix) :]
        if matcher_key not in variant_matcher:
            raise ValueError(
                f"family {family['family']} variant matcher is missing key '{matcher_key}' "
                f"for attr {key_info['attr']}"
            )
        lines.append(
            f"        {key_info['attr']} = {format_mlir_attr_value(variant_matcher[matcher_key])},"
        )
    return "\n".join(lines) + ("\n" if lines else "")


def build_snippet_blocks(
    op_info: dict[str, Any],
    contract: dict[str, Any],
    dtype: str,
    result_dtype: str | None,
    scalar_arg_type: str,
    cmp_predicate: str,
    rhs_value: str,
) -> tuple[str, str]:
    relative_path = snippet_relpath(op_info)
    setup_text, compute_text = load_snippet_sections(relative_path)
    mapping = build_snippet_mapping(
        op_info, dtype, result_dtype, scalar_arg_type, cmp_predicate, rhs_value
    )
    rendered_setup = replace_all(setup_text, mapping)
    rendered_compute = replace_all(compute_text, mapping)
    validate_rendered_snippet(relative_path, contract, rendered_setup, rendered_compute)
    return indent_block(rendered_setup, "      "), indent_block(
        rendered_compute, "          "
    )


def expand_family_instances(
    pattern: dict[str, Any], family: dict[str, Any], dtypes: list[str]
) -> list[dict[str, str]]:
    contract_map = {
        contract["id"]: contract for contract in load_snippet_contracts()["contracts"]
    }
    contract = contract_map[family["snippet_contract"]]

    instances: list[dict[str, str]] = []
    variant_axis_basis = family["variant_axis"]["basis"]
    if variant_axis_basis in {"condition_list", "explicit"}:
        variant_entries = family["variant_axis"].get("values", [None])
    else:
        variant_entries = [None]
    family_default_variant = None
    if variant_axis_basis == "placeholder":
        default_variants = family["variant_axis"].get("values", [])
        if default_variants:
            family_default_variant = default_variants[0]["id"]
    result_dtype = family.get("dtype_axis", {}).get("result_dtype")
    scalar_pos = family_scalar_index(family)
    passive_vectors = family.get("dtype_axis", {}).get("passive_vectors", {})
    arg_types_by_role = {
        str(name): str(type_name)
        for name, type_name in family.get("metadata", {})
        .get("arg_types_by_role", {})
        .items()
    }
    has_scalar_operand = any(
        role["kind"] == "scalar" for role in family["parameter_roles"]
    )
    input_tile_shape = tile_shape_from_metadata(family["metadata"], "input_tile_shape")
    result_tile_shape = tile_shape_from_metadata(
        family["metadata"], "result_tile_shape"
    )
    arg_tile_shapes = build_arg_tile_shapes(
        family["parameter_roles"],
        family["metadata"],
        input_tile_shape,
        result_tile_shape,
    )
    arg_roles = [role["kind"] for role in family["parameter_roles"]]
    for op_info in expand_ops(family):
        requested_dtypes = op_info.get("dtypes")
        op_dtypes = [
            dtype
            for dtype in dtypes
            if requested_dtypes is None or dtype in requested_dtypes
        ]
        for dtype in op_dtypes:
            vector_lanes = a5_simd_lanes(dtype)
            arg_types = [
                resolve_arg_type(role, dtype, result_dtype, arg_types_by_role)
                for role in family["parameter_roles"]
            ]
            tile_inputs = [
                (shape, arg_type)
                for role, shape, arg_type in zip(
                    family["parameter_roles"], arg_tile_shapes, arg_types
                )
                if role["kind"] == "tile" and role["io"] == "input"
            ]
            src0_tile_shape = tile_inputs[0][0] if tile_inputs else input_tile_shape
            src1_tile_shape = (
                tile_inputs[1][0] if len(tile_inputs) > 1 else src0_tile_shape
            )
            src2_tile_shape = (
                tile_inputs[2][0] if len(tile_inputs) > 2 else src1_tile_shape
            )
            src0_dtype = tile_inputs[0][1] if tile_inputs else dtype
            src1_dtype = tile_inputs[1][1] if len(tile_inputs) > 1 else src0_dtype
            src2_dtype = tile_inputs[2][1] if len(tile_inputs) > 2 else src1_dtype
            scalar_arg_type = next(
                (
                    arg_type
                    for role, arg_type in zip(family["parameter_roles"], arg_types)
                    if role["kind"] == "scalar"
                ),
                scalar_type(dtype),
            )
            effective_op_info = dict(op_info)
            effective_op_info["core_op"] = op_info.get("core_op_by_dtype", {}).get(
                dtype, op_info["core_op"]
            )
            if pattern["id"] == "compare":
                effective_op_info["core_op"] = resolve_compare_core_op(dtype)
            for variant_info in variant_entries:
                condition = ""
                cmp_predicate = ""
                variant_matcher: dict[str, Any] = {}
                variant_id = effective_op_info.get(
                    "variant_id",
                    family.get("dtype_axis", {})
                    .get("variant_id_by_dtype", {})
                    .get(dtype),
                )
                if variant_id is None:
                    variant_id = family_default_variant
                if isinstance(variant_info, dict):
                    variant_matcher = dict(variant_info.get("matcher", {}))
                    if variant_axis_basis == "condition_list":
                        condition = variant_info["matcher"]["cmpMode"]
                        cmp_predicate = resolve_compare_predicate(dtype, condition)
                        variant_id = variant_info["id"]
                    elif variant_axis_basis == "explicit":
                        variant_id = variant_info["id"]
                if variant_id is None:
                    variant_id = (
                        lower_name(condition)
                        if condition
                        else lower_name(op_info["name"])
                    )

                cost = int(op_info.get("cost", family["metadata"].get("cost", 10)))
                priority = int(
                    op_info.get("priority", family["metadata"].get("priority", 0))
                )
                func_name = build_func_name(
                    pattern, family, effective_op_info, variant_id, dtype
                )

                axis_values = [
                    f"dtype={dtype}",
                    f"core_op={effective_op_info['core_op']}",
                    f"variant_id={variant_id}",
                ]
                if condition:
                    axis_values.insert(1, f"condition={condition}")

                rhs_memref_cast = ""
                rhs_setup = ""
                rhs_load = ""
                rhs_value = "%src1v"
                compare_store = ""
                if pattern["id"] == "compare" and not has_scalar_operand:
                    rhs_memref_cast = (
                        f"    %m1 = pto.simd.tile_to_memref %src1 : {tile_type(dtype)} to "
                        f"{memref_type(dtype, input_tile_shape[1])}\n"
                    )
                    rhs_load = (
                        f"          %src1v = vector.maskedload %m1[%r, %cidx], %mask, %passive "
                        f'{{pto.simd.vld_dist = "NORM"}} : {memref_type(dtype, input_tile_shape[1])}, '
                        f"{mask_vector_type(vector_lanes)}, {vector_type(dtype)} into {vector_type(dtype)}\n"
                    )
                elif has_scalar_operand and pattern["id"] != "compare":
                    rhs_setup = f"      %scalarVec = vector.splat %scalar : {vector_type(dtype)}\n"
                    rhs_value = "%scalarVec"
                elif pattern["id"] in {"binary", "partial_binary"}:
                    rhs_value = "%rhs"

                if pattern["id"] == "compare":
                    cmp_mode_attr = lower_name(condition or "eq")
                    compare_store = (
                        "          %linearBase = arith.muli %r, %cols : index\n"
                        "          %linear = arith.addi %linearBase, %cidx : index\n"
                    )
                    if has_scalar_operand:
                        rhs_value = "%scalar"
                        compare_store += (
                            f"          pto.simd.store_predicate %lhs, {rhs_value}, %dst, %linear, %active "
                            f"{{cmpMode = #pto<cmp {cmp_mode_attr}>}} : "
                            f"{vector_type(dtype)}, {scalar_arg_type}, "
                            f"{tile_type(result_dtype or dtype, *result_tile_shape)}, index, index\n"
                        )
                    else:
                        compare_store += (
                            f"          pto.simd.store_predicate %lhs, {rhs_value}, %dst, %linear, %active "
                            f"{{cmpMode = #pto<cmp {cmp_mode_attr}>}} : "
                            f"{vector_type(dtype)}, {vector_type(dtype)}, "
                            f"{tile_type(result_dtype or dtype, *result_tile_shape)}, index, index\n"
                        )

                passive_vector = op_info.get(
                    "passive_vector_by_dtype",
                    {},
                ).get(
                    dtype,
                    op_info.get(
                        "passive_vector",
                        passive_vectors.get(dtype, dense_literal(dtype, "zero")),
                    ),
                )
                reduce_init = op_info.get(
                    "reduce_init_by_dtype",
                    {},
                ).get(
                    dtype,
                    op_info.get(
                        "reduce_init", scalar_literal(result_dtype or dtype, "zero")
                    ),
                )

                extra_setup, compute = build_snippet_blocks(
                    effective_op_info,
                    contract,
                    dtype,
                    result_dtype,
                    scalar_arg_type,
                    cmp_predicate,
                    rhs_value,
                )

                instances.append(
                    {
                        "family_id": family["family"],
                        "func_name": func_name,
                        "kind": family["kind"],
                        "op_name": effective_op_info["name"],
                        "variant_id": variant_id,
                        "match_dtype": dtype,
                        "arg_decls": build_arg_decls(
                            family["parameter_roles"],
                            dtype,
                            result_dtype,
                            arg_tile_shapes,
                            arg_types_by_role,
                        ),
                        "match_attrs": build_match_attrs(arg_roles),
                        "extra_template_attrs": build_extra_template_attrs(
                            effective_op_info
                        ),
                        "variant_match_attrs": build_variant_match_attrs(
                            family, variant_matcher
                        ),
                        "core_op": effective_op_info["core_op"],
                        "cmp_mode": condition,
                        "cmp_predicate": cmp_predicate,
                        "scalar_pos_attr": (
                            f"        pto.oplib.match.scalar_pos = {scalar_pos} : i64,\n"
                            if scalar_pos is not None
                            else ""
                        ),
                        "scalar_pos": "" if scalar_pos is None else str(scalar_pos),
                        "axis_values": ", ".join(axis_values),
                        "input_tile_type": tile_type(dtype, *input_tile_shape),
                        "result_tile_type": tile_type(
                            result_dtype or dtype, *result_tile_shape
                        ),
                        "input_memref_type": memref_type(dtype, input_tile_shape[1]),
                        "result_memref_type": memref_type(
                            result_dtype or dtype, result_tile_shape[1]
                        ),
                        "input_vector_type": vector_type(dtype),
                        "result_vector_type": vector_type(result_dtype or dtype),
                        "src0_tile_type": tile_type(src0_dtype, *src0_tile_shape),
                        "src1_tile_type": tile_type(src1_dtype, *src1_tile_shape),
                        "src2_tile_type": tile_type(src2_dtype, *src2_tile_shape),
                        "src0_memref_type": memref_type(src0_dtype, src0_tile_shape[1]),
                        "src1_memref_type": memref_type(src1_dtype, src1_tile_shape[1]),
                        "src2_memref_type": memref_type(src2_dtype, src2_tile_shape[1]),
                        "src0_vector_type": (
                            vector_type(src0_dtype, vector_lanes)
                            if pattern["id"] == "select_mask"
                            else vector_type(src0_dtype)
                        ),
                        "src1_vector_type": vector_type(src1_dtype),
                        "src2_vector_type": vector_type(src2_dtype),
                        "scalar_vector_type": vector_type(dtype, 1),
                        "scalar_arg_single_vector_type": vector_type(
                            scalar_arg_type, 1
                        ),
                        "scalar_arg_vector_type": vector_type(scalar_arg_type),
                        "splat_shuffle_mask": splat_shuffle_mask(vector_lanes),
                        "simd_lanes": str(vector_lanes),
                        "scalar_type": scalar_type(result_dtype or dtype),
                        "scalar_arg_type": scalar_arg_type,
                        "mask_vector_type": mask_vector_type(vector_lanes),
                        "passive_vector": passive_vector,
                        "reduce_init": reduce_init,
                        "rhs_memref_cast": rhs_memref_cast,
                        "rhs_setup": rhs_setup,
                        "rhs_load": rhs_load,
                        "rhs_value": rhs_value,
                        "compare_store": compare_store,
                        "extra_setup": extra_setup,
                        "compute": compute,
                        "cost": str(cost),
                        "priority": str(priority),
                    }
                )
    return instances


def render_family_output(
    pattern: dict[str, Any],
    family: dict[str, Any],
    dtypes: list[str],
    output_role: str,
) -> str:
    instance_template = load_text(SKELETON_DIR / pattern["template"])
    module_template = load_text(MODULE_TEMPLATE_PATH)
    instance_blocks: list[str] = []
    for instance in expand_family_instances(pattern, family, dtypes):
        instance_blocks.append(
            replace_all(
                instance_template,
                {
                    "FAMILY_ID": instance["family_id"],
                    "AXIS_VALUES": instance["axis_values"],
                    "FUNC_NAME": instance["func_name"],
                    "ARG_DECLS": instance["arg_decls"],
                    "KIND": instance["kind"],
                    "OP_NAME": instance["op_name"],
                    "VARIANT_ID": instance["variant_id"],
                    "MATCH_DTYPE": instance["match_dtype"],
                    "MATCH_ATTRS": instance["match_attrs"],
                    "EXTRA_TEMPLATE_ATTRS": instance["extra_template_attrs"],
                    "VARIANT_MATCH_ATTRS": instance["variant_match_attrs"],
                    "CORE_OP": instance["core_op"],
                    "CMP_MODE": instance["cmp_mode"],
                    "CMP_PREDICATE": instance["cmp_predicate"],
                    "SCALAR_POS_ATTR": instance["scalar_pos_attr"],
                    "SCALAR_POS": instance["scalar_pos"],
                    "INPUT_TILE_TYPE": instance["input_tile_type"],
                    "RESULT_TILE_TYPE": instance["result_tile_type"],
                    "INPUT_MEMREF_TYPE": instance["input_memref_type"],
                    "RESULT_MEMREF_TYPE": instance["result_memref_type"],
                    "INPUT_VECTOR_TYPE": instance["input_vector_type"],
                    "RESULT_VECTOR_TYPE": instance["result_vector_type"],
                    "SRC0_TILE_TYPE": instance["src0_tile_type"],
                    "SRC1_TILE_TYPE": instance["src1_tile_type"],
                    "SRC2_TILE_TYPE": instance["src2_tile_type"],
                    "SRC0_MEMREF_TYPE": instance["src0_memref_type"],
                    "SRC1_MEMREF_TYPE": instance["src1_memref_type"],
                    "SRC2_MEMREF_TYPE": instance["src2_memref_type"],
                    "SRC0_VECTOR_TYPE": instance["src0_vector_type"],
                    "SRC1_VECTOR_TYPE": instance["src1_vector_type"],
                    "SRC2_VECTOR_TYPE": instance["src2_vector_type"],
                    "SCALAR_VECTOR_TYPE": instance["scalar_vector_type"],
                    "SCALAR_ARG_SINGLE_VECTOR_TYPE": instance[
                        "scalar_arg_single_vector_type"
                    ],
                    "SCALAR_ARG_VECTOR_TYPE": instance["scalar_arg_vector_type"],
                    "SPLAT_SHUFFLE_MASK": instance["splat_shuffle_mask"],
                    "SIMD_LANES": instance["simd_lanes"],
                    "VECTOR_TYPE": instance["input_vector_type"],
                    "SCALAR_TYPE": instance["scalar_type"],
                    "SCALAR_ARG_TYPE": instance["scalar_arg_type"],
                    "MASK_VECTOR_TYPE": instance["mask_vector_type"],
                    "PASSIVE_VECTOR": instance["passive_vector"],
                    "REDUCE_INIT": instance["reduce_init"],
                    "RHS_MEMREF_CAST": instance["rhs_memref_cast"],
                    "RHS_SETUP": instance["rhs_setup"],
                    "RHS_LOAD": instance["rhs_load"],
                    "RHS_VALUE": instance["rhs_value"],
                    "COMPARE_STORE": instance["compare_store"],
                    "EXTRA_SETUP": instance["extra_setup"],
                    "COMPUTE": instance["compute"],
                    "COST": instance["cost"],
                    "PRIORITY": instance["priority"],
                },
            ).rstrip()
        )
    rendered = replace_all(
        module_template,
        {
            "PATTERN_ID": pattern["id"],
            "TEMPLATE_PATH": f"skeletons/{pattern['template']}",
            "AXES": ", ".join(pattern["axes"]),
            "OUTPUT_ROLE": output_role,
            "INSTANCE_BLOCKS": "\n\n".join(instance_blocks),
        },
    )
    return rendered.rstrip() + "\n"


def render_outputs() -> dict[Path, str]:
    family_doc = load_family_doc()
    pattern_map = {pattern["id"]: pattern for pattern in family_doc["patterns"]}
    outputs: dict[Path, str] = {}

    for family in family_doc["families"]:
        if family["status"] != "active":
            continue
        pattern = pattern_map[family["pattern"]]
        metadata = family["metadata"]
        family_dtypes = list(family["dtype_axis"]["values"])

        if "targets" in metadata:
            for target in metadata["targets"]:
                outputs[SCRIPT_DIR / target["output"]] = render_family_output(
                    pattern,
                    family,
                    list(target["dtypes"]),
                    target.get(
                        "output_role",
                        "generated concrete templates synchronized from skeleton source.",
                    ),
                )

        if "output" in metadata:
            outputs[SCRIPT_DIR / metadata["output"]] = render_family_output(
                pattern,
                family,
                family_dtypes,
                metadata.get(
                    "output_role",
                    "generated concrete templates synchronized from skeleton source.",
                ),
            )
    for path, content in outputs.items():
        validate_expected_output(path, content)
    return outputs


def write_outputs(outputs: dict[Path, str]) -> int:
    stale_outputs = find_stale_generated_outputs(outputs)
    for path in stale_outputs:
        path.unlink()
        print(f"removed stale {path.relative_to(SCRIPT_DIR)}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"wrote {path.relative_to(SCRIPT_DIR)}")
    return 0


def check_outputs(outputs: dict[Path, str]) -> int:
    failures = 0
    stale_outputs = find_stale_generated_outputs(outputs)
    for path in stale_outputs:
        print(
            "unexpected generated file: "
            f"{path.relative_to(SCRIPT_DIR)} (remove it or regenerate with --write)",
            file=sys.stderr,
        )
        failures += 1
    for path, expected in outputs.items():
        if not path.exists():
            print(
                f"missing generated file: {path.relative_to(SCRIPT_DIR)}",
                file=sys.stderr,
            )
            failures += 1
            continue
        actual = load_text(path)
        if actual == expected:
            print(f"ok {path.relative_to(SCRIPT_DIR)}")
            continue
        failures += 1
        print(f"drift detected: {path.relative_to(SCRIPT_DIR)}", file=sys.stderr)
        diff = difflib.unified_diff(
            actual.splitlines(),
            expected.splitlines(),
            fromfile=str(path),
            tofile=f"{path} (expected)",
            lineterm="",
        )
        for line in diff:
            print(line, file=sys.stderr)
    return 1 if failures else 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate Level-3 concrete templates from Family DSL and snippet sources."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--write", action="store_true", help="write generated concrete templates"
    )
    mode.add_argument(
        "--check", action="store_true", help="check generated concrete templates"
    )
    mode.add_argument(
        "--list", action="store_true", help="list generated concrete output paths"
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    outputs = render_outputs()
    if args.list:
        for path in sorted(outputs):
            print(path.relative_to(SCRIPT_DIR))
        return 0
    if args.write:
        return write_outputs(outputs)
    return check_outputs(outputs)


if __name__ == "__main__":
    sys.exit(main())
