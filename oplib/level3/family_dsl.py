#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
FAMILY_DSL_PATH = SCRIPT_DIR / "families" / "a5_oplib_v1_family_dsl.json"
SNIPPET_CONTRACTS_PATH = SCRIPT_DIR / "families" / "a5_oplib_v1_snippet_contracts.json"
CATALOG_PATH = SCRIPT_DIR / "skeletons" / "catalog.json"

ALLOWED_ROLE_KINDS = {"tile", "scalar"}
ALLOWED_ROLE_IO = {"input", "output"}
ALLOWED_FAMILY_STATUS = {"active", "reserved"}
ALLOWED_VARIANT_BASIS = {
    "per_op",
    "per_op_variant",
    "dtype_map",
    "condition_list",
    "explicit",
    "placeholder",
}
ALLOWED_MATCHER_KEYS = {
    "kind",
    "op",
    "dtype",
    "variant_id",
    "cmpMode",
    "scalarPos",
    "requiredVariantId",
    "isBinary",
}
ALLOWED_MATCHER_LOCATIONS = {"template_attr", "request_only"}
ALLOWED_SNIPPET_FAMILY_CLASSES = {
    "binary",
    "tile_scalar",
    "unary",
    "ternary",
    "compare",
    "select",
    "reduction",
    "broadcast",
}
EXPECTED_RESULT_SSA_BY_FAMILY_CLASS = {
    "binary": "%result",
    "tile_scalar": "%result",
    "unary": "%result",
    "ternary": "%result",
    "compare": "%cmp",
    "select": "%result",
    "reduction": "%result",
    "broadcast": "%result",
}
FAMILY_CLASS_BY_KIND = {
    "l3_float_binary_elementwise_template": "binary",
    "l3_int_binary_elementwise_template": "binary",
    "l3_float_partial_binary_template": "binary",
    "l3_float_tile_scalar_template": "tile_scalar",
    "l3_int_tile_scalar_elementwise_template": "tile_scalar",
    "l3_float_ternary_tile_template": "ternary",
    "l3_float_ternary_tile_scalar_template": "ternary",
    "l3_float_unary_template": "unary",
    "l3_float_unary_math_template": "unary",
    "l3_int_unary_template": "unary",
    "l3_reduce_row_template": "reduction",
    "l3_reduce_col_template": "reduction",
    "l3_reduce_colsum_template": "reduction",
    "l3_broadcast_row_template": "broadcast",
    "l3_broadcast_col_template": "broadcast",
    "l3_broadcast_row_binary_template": "broadcast",
    "l3_scalar_expand_template": "broadcast",
    "l3_cmp_tile_tile_template": "compare",
    "l3_cmp_tile_scalar_template": "compare",
    "l3_select_mask_template": "select",
    "l3_select_scalar_template": "select",
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def scalar_index(parameter_roles: list[dict[str, Any]]) -> int | None:
    for index, role in enumerate(parameter_roles):
        if role["kind"] == "scalar":
            return index
    return None


def role_signature(parameter_roles: list[dict[str, Any]]) -> list[str]:
    return [role["kind"] for role in parameter_roles]


def family_has_generated_output(family: dict[str, Any]) -> bool:
    metadata = family.get("metadata", {})
    return bool(metadata.get("output")) or bool(metadata.get("targets"))


def matcher_keys_by_name(family: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {entry["key"]: entry for entry in family["matcher_keys"]}


def _validate_patterns(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    patterns = doc.get("patterns")
    _require(
        isinstance(patterns, list) and patterns,
        "family DSL requires a non-empty 'patterns' list",
    )
    seen: set[str] = set()
    pattern_map: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        _require(isinstance(pattern, dict), "each pattern must be an object")
        pattern_id = str(pattern.get("id", "")).strip()
        _require(pattern_id, "pattern missing id")
        _require(pattern_id not in seen, f"duplicate pattern id: {pattern_id}")
        seen.add(pattern_id)
        template = str(pattern.get("template", "")).strip()
        axes = pattern.get("axes")
        _require(template, f"pattern {pattern_id} missing template")
        _require(
            isinstance(axes, list) and axes,
            f"pattern {pattern_id} requires non-empty axes",
        )
        pattern_map[pattern_id] = pattern
    return pattern_map


def _validate_matcher_keys(family: dict[str, Any], has_scalar: bool) -> None:
    keys = family.get("matcher_keys")
    family_name = family["family"]
    _require(
        isinstance(keys, list) and keys,
        f"family {family_name} requires non-empty matcher_keys",
    )
    seen: set[str] = set()
    for entry in keys:
        _require(
            isinstance(entry, dict),
            f"family {family_name} matcher key must be an object",
        )
        key = str(entry.get("key", "")).strip()
        location = str(entry.get("location", "")).strip()
        _require(
            key in ALLOWED_MATCHER_KEYS,
            f"family {family_name} has unsupported matcher key: {key}",
        )
        _require(
            key not in seen, f"family {family_name} has duplicate matcher key: {key}"
        )
        seen.add(key)
        _require(
            location in ALLOWED_MATCHER_LOCATIONS,
            f"family {family_name} matcher key {key} has unsupported location: {location}",
        )
        if location == "template_attr":
            attr = str(entry.get("attr", "")).strip()
            _require(
                attr, f"family {family_name} matcher key {key} missing template attr"
            )
    for required in ("kind", "op", "dtype", "variant_id"):
        _require(
            required in seen,
            f"family {family_name} missing required matcher key: {required}",
        )
    if has_scalar:
        _require(
            "scalarPos" in seen,
            f"family {family_name} has scalar parameters but no scalarPos matcher key",
        )


def _validate_ops(family: dict[str, Any]) -> None:
    family_name = family["family"]
    ops = family.get("ops")
    _require(
        isinstance(ops, list) and ops, f"family {family_name} requires non-empty ops"
    )
    seen: set[str] = set()
    key_names = matcher_keys_by_name(family)
    variant_basis = family["variant_axis"]["basis"]
    require_snippet = family["status"] == "active" and family_has_generated_output(
        family
    )
    for op in ops:
        _require(
            isinstance(op, dict), f"family {family_name} op entry must be an object"
        )
        name = str(op.get("name", "")).strip()
        _require(name, f"family {family_name} has op without name")
        _require(
            name not in seen, f"family {family_name} has duplicate op name: {name}"
        )
        seen.add(name)
        _require(
            str(op.get("core_op", "")).strip(),
            f"family {family_name} op {name} missing core_op",
        )
        if "snippet" in op:
            _require(
                str(op.get("snippet", "")).strip(),
                f"family {family_name} op {name} snippet must be a non-empty string when present",
            )
        if require_snippet and not op.get("variants"):
            _require(
                str(op.get("snippet", "")).strip(),
                f"active family {family_name} op {name} requires snippet",
            )
        variants = op.get("variants", [])
        if variants:
            _require(
                isinstance(variants, list),
                f"family {family_name} op {name} variants must be a list when present",
            )
            variant_ids: set[str] = set()
            for variant in variants:
                _require(
                    isinstance(variant, dict),
                    f"family {family_name} op {name} variant must be an object",
                )
                variant_id = str(variant.get("id", "")).strip()
                _require(
                    variant_id, f"family {family_name} op {name} variant missing id"
                )
                _require(
                    variant_id not in variant_ids,
                    f"family {family_name} op {name} has duplicate variant id: {variant_id}",
                )
                variant_ids.add(variant_id)
                request_keys = variant.get("request_keys", {})
                _require(
                    isinstance(request_keys, dict),
                    f"family {family_name} op {name} variant {variant_id} request_keys must be an object",
                )
                body = variant.get("body", {})
                _require(
                    isinstance(body, dict),
                    f"family {family_name} op {name} variant {variant_id} body must be an object when present",
                )
                if "snippet" in body:
                    _require(
                        str(body.get("snippet", "")).strip(),
                        f"family {family_name} op {name} variant {variant_id} snippet must be a non-empty string",
                    )
                if require_snippet:
                    _require(
                        str(body.get("snippet", "")).strip()
                        or str(op.get("snippet", "")).strip(),
                        f"active family {family_name} op {name} variant {variant_id} requires snippet",
                    )
                if "requiredVariantId" in request_keys:
                    _require(
                        "requiredVariantId" in key_names,
                        f"family {family_name} uses requiredVariantId without matcher key declaration",
                    )
        elif variant_basis == "per_op_variant":
            _require(
                "variant_id" in op,
                f"family {family_name} op {name} requires variant_id or variants for per_op_variant basis",
            )


def validate_snippet_contracts(doc: dict[str, Any]) -> dict[str, dict[str, Any]]:
    _require(isinstance(doc, dict), "snippet contracts root must be an object")
    _require(
        str(doc.get("schema_version", "")).strip()
        == "a5_oplib_v1_snippet_contracts/v1",
        "snippet contracts have unexpected schema_version",
    )
    contracts = doc.get("contracts")
    _require(
        isinstance(contracts, list) and contracts,
        "snippet contracts require a non-empty 'contracts' list",
    )

    seen_ids: set[str] = set()
    contract_map: dict[str, dict[str, Any]] = {}
    for contract in contracts:
        _require(isinstance(contract, dict), "each snippet contract must be an object")
        contract_id = str(contract.get("id", "")).strip()
        family_class = str(contract.get("family_class", "")).strip()
        _require(contract_id, "snippet contract missing id")
        _require(
            contract_id not in seen_ids, f"duplicate snippet contract id: {contract_id}"
        )
        seen_ids.add(contract_id)
        _require(
            family_class in ALLOWED_SNIPPET_FAMILY_CLASSES,
            f"snippet contract {contract_id} has unsupported family_class: {family_class}",
        )

        signature = contract.get("role_signature")
        _require(
            isinstance(signature, list) and signature,
            f"snippet contract {contract_id} requires non-empty role_signature",
        )
        for kind in signature:
            _require(
                kind in ALLOWED_ROLE_KINDS,
                f"snippet contract {contract_id} has unsupported role kind in role_signature: {kind}",
            )

        visible_ssa = contract.get("visible_ssa")
        _require(
            isinstance(visible_ssa, list) and visible_ssa,
            f"snippet contract {contract_id} requires non-empty visible_ssa",
        )
        visible_names: set[str] = set()
        for entry in visible_ssa:
            _require(
                isinstance(entry, dict),
                f"snippet contract {contract_id} visible_ssa entry must be an object",
            )
            name = str(entry.get("name", "")).strip()
            source = str(entry.get("source", "")).strip()
            _require(
                name.startswith("%"),
                f"snippet contract {contract_id} visible SSA must start with %: {name}",
            )
            _require(
                name not in visible_names,
                f"snippet contract {contract_id} duplicates visible SSA: {name}",
            )
            visible_names.add(name)
            _require(
                source,
                f"snippet contract {contract_id} visible SSA {name} missing source",
            )

        result_ssa = str(contract.get("result_ssa", "")).strip()
        expected_result_ssa = EXPECTED_RESULT_SSA_BY_FAMILY_CLASS[family_class]
        _require(result_ssa, f"snippet contract {contract_id} missing result_ssa")
        _require(
            result_ssa == expected_result_ssa,
            f"snippet contract {contract_id} must use result SSA {expected_result_ssa}, got {result_ssa}",
        )

        generator_owned = contract.get("generator_owned")
        _require(
            isinstance(generator_owned, list) and generator_owned,
            f"snippet contract {contract_id} requires non-empty generator_owned",
        )
        for item in generator_owned:
            _require(
                isinstance(item, str) and item.strip(),
                f"snippet contract {contract_id} generator_owned entries must be non-empty strings",
            )

        snippet_must_not_define = contract.get("snippet_must_not_define")
        _require(
            isinstance(snippet_must_not_define, list) and snippet_must_not_define,
            f"snippet contract {contract_id} requires non-empty snippet_must_not_define",
        )
        for name in snippet_must_not_define:
            _require(
                isinstance(name, str) and name.startswith("%"),
                f"snippet contract {contract_id} snippet_must_not_define entries must be SSA names: {name}",
            )

        forbidden_ops = contract.get("forbidden_ops")
        _require(
            isinstance(forbidden_ops, list) and forbidden_ops,
            f"snippet contract {contract_id} requires non-empty forbidden_ops",
        )
        for op_name in forbidden_ops:
            _require(
                isinstance(op_name, str) and op_name.strip(),
                f"snippet contract {contract_id} forbidden_ops entries must be non-empty strings",
            )

        contract_map[contract_id] = contract

    return contract_map


def load_snippet_contracts(path: Path = SNIPPET_CONTRACTS_PATH) -> dict[str, Any]:
    doc = load_json(path)
    validate_snippet_contracts(doc)
    return doc


def validate_family_dsl(
    doc: dict[str, Any],
    contract_map: dict[str, dict[str, Any]] | None = None,
) -> None:
    if contract_map is None:
        contract_map = validate_snippet_contracts(load_json(SNIPPET_CONTRACTS_PATH))

    _require(isinstance(doc, dict), "family DSL root must be an object")
    _require(
        str(doc.get("schema_version", "")).strip() == "a5_oplib_v1_family_dsl/v1",
        "family DSL has unexpected schema_version",
    )
    pattern_map = _validate_patterns(doc)
    families = doc.get("families")
    _require(
        isinstance(families, list) and families,
        "family DSL requires a non-empty 'families' list",
    )

    seen_family: set[str] = set()
    seen_kind: set[str] = set()
    for family in families:
        _require(isinstance(family, dict), "each family entry must be an object")
        family_name = str(family.get("family", "")).strip()
        kind = str(family.get("kind", "")).strip()
        status = str(family.get("status", "")).strip()
        pattern_id = str(family.get("pattern", "")).strip()
        snippet_contract_id = str(family.get("snippet_contract", "")).strip()
        _require(family_name, "family entry missing family name")
        _require(
            family_name not in seen_family, f"duplicate family name: {family_name}"
        )
        seen_family.add(family_name)
        _require(kind, f"family {family_name} missing kind")
        _require(kind not in seen_kind, f"duplicate family kind: {kind}")
        seen_kind.add(kind)
        _require(
            status in ALLOWED_FAMILY_STATUS,
            f"family {family_name} has unsupported status: {status}",
        )
        _require(
            pattern_id in pattern_map,
            f"family {family_name} references unknown pattern: {pattern_id}",
        )
        _require(
            kind in FAMILY_CLASS_BY_KIND,
            f"family {family_name} uses unsupported kind for A5 OpLib V1 snippet contracts: {kind}",
        )
        _require(snippet_contract_id, f"family {family_name} missing snippet_contract")
        _require(
            snippet_contract_id in contract_map,
            f"family {family_name} references unknown snippet contract: {snippet_contract_id}",
        )

        roles = family.get("parameter_roles")
        _require(
            isinstance(roles, list) and roles,
            f"family {family_name} requires parameter_roles",
        )
        role_names: set[str] = set()
        output_count = 0
        for role in roles:
            _require(
                isinstance(role, dict),
                f"family {family_name} parameter role must be an object",
            )
            role_name = str(role.get("name", "")).strip()
            role_kind = str(role.get("kind", "")).strip()
            role_io = str(role.get("io", "")).strip()
            _require(role_name, f"family {family_name} has parameter role without name")
            _require(
                role_name not in role_names,
                f"family {family_name} has duplicate parameter role: {role_name}",
            )
            role_names.add(role_name)
            _require(
                role_kind in ALLOWED_ROLE_KINDS,
                f"family {family_name} has invalid role kind: {role_kind}",
            )
            _require(
                role_io in ALLOWED_ROLE_IO,
                f"family {family_name} has invalid role io: {role_io}",
            )
            if role_io == "output":
                output_count += 1
        _require(
            output_count == 1, f"family {family_name} must have exactly one output role"
        )
        _require(
            roles[-1]["io"] == "output",
            f"family {family_name} must place output role last",
        )

        contract = contract_map[snippet_contract_id]
        expected_family_class = FAMILY_CLASS_BY_KIND[kind]
        _require(
            contract["family_class"] == expected_family_class,
            "family "
            f"{family_name} kind {kind} expects snippet contract class {expected_family_class}, "
            f"got {contract['family_class']}",
        )
        _require(
            contract["role_signature"] == role_signature(roles),
            "family "
            f"{family_name} role signature {role_signature(roles)} does not match snippet contract "
            f"{snippet_contract_id} signature {contract['role_signature']}",
        )

        dtype_axis = family.get("dtype_axis")
        _require(
            isinstance(dtype_axis, dict), f"family {family_name} requires dtype_axis"
        )
        dtype_values = dtype_axis.get("values")
        _require(
            isinstance(dtype_values, list) and dtype_values,
            f"family {family_name} dtype_axis requires non-empty values",
        )
        if "variant_id_by_dtype" in dtype_axis:
            variant_map = dtype_axis["variant_id_by_dtype"]
            _require(
                isinstance(variant_map, dict) and variant_map,
                f"family {family_name} variant_id_by_dtype must be a non-empty object",
            )
            for dtype in variant_map:
                _require(
                    dtype in dtype_values,
                    f"family {family_name} variant_id_by_dtype references unknown dtype {dtype}",
                )

        variant_axis = family.get("variant_axis")
        _require(
            isinstance(variant_axis, dict),
            f"family {family_name} requires variant_axis",
        )
        _require(
            str(variant_axis.get("name", "")).strip() == "variant_id",
            f"family {family_name} variant_axis must be named variant_id",
        )
        basis = str(variant_axis.get("basis", "")).strip()
        _require(
            basis in ALLOWED_VARIANT_BASIS,
            f"family {family_name} has unsupported variant axis basis: {basis}",
        )
        values = variant_axis.get("values", [])
        if basis in {"condition_list", "explicit", "placeholder"}:
            _require(
                isinstance(values, list) and values,
                f"family {family_name} variant axis basis {basis} requires non-empty values",
            )
        if basis == "condition_list":
            for value in values:
                _require(
                    isinstance(value, dict),
                    f"family {family_name} condition entry must be an object",
                )
                matcher = value.get("matcher", {})
                _require(
                    isinstance(matcher, dict)
                    and str(matcher.get("cmpMode", "")).strip(),
                    f"family {family_name} condition entry requires cmpMode matcher",
                )

        metadata = family.get("metadata")
        _require(isinstance(metadata, dict), f"family {family_name} requires metadata")
        _require(
            str(metadata.get("entry_role", "")).strip() == "variant",
            f"family {family_name} metadata.entry_role must be variant",
        )
        has_scalar = scalar_index(roles) is not None
        _validate_matcher_keys(family, has_scalar)
        _validate_ops(family)


def load_family_dsl(
    path: Path = FAMILY_DSL_PATH,
    snippet_contracts_path: Path = SNIPPET_CONTRACTS_PATH,
) -> dict[str, Any]:
    doc = load_json(path)
    contract_map = validate_snippet_contracts(load_json(snippet_contracts_path))
    validate_family_dsl(doc, contract_map)
    return doc


def project_catalog_from_family_dsl(
    doc: dict[str, Any],
    contract_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_family_dsl(doc, contract_map)
    pattern_order = [pattern["id"] for pattern in doc["patterns"]]
    projected_patterns: dict[str, dict[str, Any]] = {
        pattern["id"]: {
            "id": pattern["id"],
            "template": pattern["template"],
            "axes": list(pattern["axes"]),
            "families": [],
        }
        for pattern in doc["patterns"]
    }

    for family in doc["families"]:
        if family["status"] != "active" or not family_has_generated_output(family):
            continue
        dtype_axis = family["dtype_axis"]
        metadata = family["metadata"]
        projected_family: dict[str, Any] = {
            "family_id": family["family"],
            "kind": family["kind"],
            "arg_roles": [role["kind"] for role in family["parameter_roles"]],
            "dtypes": list(dtype_axis["values"]),
            "ops": [],
        }
        if "output" in metadata:
            projected_family["output"] = metadata["output"]
        if "output_role" in metadata:
            projected_family["output_role"] = metadata["output_role"]
        if "func_name_format" in metadata:
            projected_family["func_name_format"] = metadata["func_name_format"]
        if "cost" in metadata:
            projected_family["cost"] = metadata["cost"]
        if "priority" in metadata:
            projected_family["priority"] = metadata["priority"]
        if "targets" in metadata:
            projected_family["targets"] = metadata["targets"]
        if "result_dtype" in dtype_axis:
            projected_family["result_dtype"] = dtype_axis["result_dtype"]
        if "passive_vectors" in dtype_axis:
            projected_family["passive_vectors"] = dtype_axis["passive_vectors"]
        if "variant_id_by_dtype" in dtype_axis:
            projected_family["variant_id_by_dtype"] = dtype_axis["variant_id_by_dtype"]
        pos = scalar_index(family["parameter_roles"])
        if pos is not None:
            projected_family["scalar_pos"] = pos
        if family["variant_axis"]["basis"] == "condition_list":
            conditions: list[dict[str, Any]] = []
            for entry in family["variant_axis"]["values"]:
                condition = {
                    "mode": entry["matcher"]["cmpMode"],
                    "variant_id": entry["id"],
                }
                metadata_entry = entry.get("metadata", {})
                if "predicate" in metadata_entry:
                    condition["predicate"] = metadata_entry["predicate"]
                if "cost" in metadata_entry:
                    condition["cost"] = metadata_entry["cost"]
                if "priority" in metadata_entry:
                    condition["priority"] = metadata_entry["priority"]
                conditions.append(condition)
            projected_family["conditions"] = conditions

        for op in family["ops"]:
            base = {
                "op": op["name"],
                "core_op": op["core_op"],
            }
            if "body_kind" in op:
                base["body_kind"] = op["body_kind"]
            if "operand_order" in op:
                base["operand_order"] = op["operand_order"]
            if "cost" in op:
                base["cost"] = op["cost"]
            if "priority" in op:
                base["priority"] = op["priority"]
            if "func_name_format" in op:
                base["func_name_format"] = op["func_name_format"]
            if "dtypes" in op:
                base["dtypes"] = op["dtypes"]
            if "variant_id" in op:
                base["variant_id"] = op["variant_id"]

            variants = op.get("variants", [])
            if variants:
                for variant in variants:
                    entry = dict(base)
                    entry["variant_id"] = variant["id"]
                    if "dtypes" in variant:
                        entry["dtypes"] = variant["dtypes"]
                    body = variant.get("body", {})
                    if "body_kind" in body:
                        entry["body_kind"] = body["body_kind"]
                    if "operand_order" in body:
                        entry["operand_order"] = body["operand_order"]
                    metadata_entry = variant.get("metadata", {})
                    if "cost" in metadata_entry:
                        entry["cost"] = metadata_entry["cost"]
                    if "priority" in metadata_entry:
                        entry["priority"] = metadata_entry["priority"]
                    if "func_name_format" in metadata_entry:
                        entry["func_name_format"] = metadata_entry["func_name_format"]
                    projected_family["ops"].append(entry)
                continue

            projected_family["ops"].append(base)

        projected_patterns[family["pattern"]]["families"].append(projected_family)

    return {
        "patterns": [
            projected_patterns[pattern_id]
            for pattern_id in pattern_order
            if projected_patterns[pattern_id]["families"]
        ]
    }


def ensure_catalog_sync(
    catalog: dict[str, Any],
    family_doc: dict[str, Any] | None = None,
    contract_map: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    family_doc = family_doc or load_family_dsl()
    projected = project_catalog_from_family_dsl(family_doc, contract_map)
    if catalog == projected:
        return projected

    actual = json.dumps(catalog, indent=2, ensure_ascii=False).splitlines()
    expected = json.dumps(projected, indent=2, ensure_ascii=False).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            actual,
            expected,
            fromfile=str(CATALOG_PATH),
            tofile=f"{FAMILY_DSL_PATH} (projected catalog)",
            lineterm="",
        )
    )
    raise ValueError(
        "skeleton catalog is not synchronized with the Family DSL in "
        f"{FAMILY_DSL_PATH}\n{diff}"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate and inspect the A5 OpLib V1 Family DSL."
    )
    parser.add_argument("--family-dsl", type=Path, default=FAMILY_DSL_PATH)
    parser.add_argument(
        "--snippet-contracts", type=Path, default=SNIPPET_CONTRACTS_PATH
    )
    parser.add_argument("--catalog", type=Path, default=CATALOG_PATH)
    parser.add_argument(
        "--check-snippet-contracts",
        action="store_true",
        help="check snippet contract data and family-to-contract bindings",
    )
    parser.add_argument(
        "--check-catalog",
        action="store_true",
        help="check catalog.json against the Family DSL",
    )
    parser.add_argument(
        "--print-projected-catalog",
        action="store_true",
        help="print the catalog projection derived from the Family DSL",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    snippet_contract_doc = load_snippet_contracts(args.snippet_contracts)
    contract_map = validate_snippet_contracts(snippet_contract_doc)
    family_doc = load_json(args.family_dsl)
    validate_family_dsl(family_doc, contract_map)
    if args.print_projected_catalog:
        print(
            json.dumps(
                project_catalog_from_family_dsl(family_doc, contract_map),
                indent=2,
                ensure_ascii=False,
            )
        )
        return 0
    if args.check_snippet_contracts:
        print("OK: snippet contracts parsed and Family DSL bindings validated.")
        return 0
    if args.check_catalog:
        catalog = load_json(args.catalog)
        ensure_catalog_sync(catalog, family_doc, contract_map)
        print(
            "OK: Family DSL, snippet contracts, and skeleton catalog are synchronized."
        )
        return 0
    print("OK: Family DSL and snippet contracts parsed and validated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
