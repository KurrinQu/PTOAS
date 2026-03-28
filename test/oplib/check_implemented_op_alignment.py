#!/usr/bin/env python3
"""Check that every implemented manifest op has dtype-level template and lowering coverage."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


DTYPE_PATTERN = re.compile(r"dtype=([a-z0-9]+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that every implemented A5 OpLib V1 operator has concrete "
            "template coverage and lowering use cases for every manifest "
            "dtype_support entry."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--template-dir", type=Path, required=True)
    parser.add_argument("--test-dir", type=Path, required=True)
    return parser.parse_args()


def load_implemented_ops(manifest_path: Path) -> dict[str, list[str]]:
    manifest = json.loads(manifest_path.read_text())
    implemented: dict[str, list[str]] = {}
    for entry in manifest["operators"]:
        if entry.get("a5_status") != "implemented":
            continue
        dtypes = sorted(set(entry.get("dtype_support", [])))
        implemented[entry["op"]] = dtypes
    return dict(sorted(implemented.items()))


def collect_template_hits(template_dir: Path) -> dict[str, dict[str, list[str]]]:
    hits: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    func_pattern = re.compile(r"func\.func\b.*?attributes\s*\{(.*?)\}", re.S)
    op_pattern = re.compile(r'pto\.oplib\.op = "([^"]+)"')
    dtype_pattern = re.compile(r'pto\.oplib\.match\.dtype = "([^"]+)"')
    for template in sorted(template_dir.glob("*.mlir")):
        text = template.read_text()
        for func_match in func_pattern.finditer(text):
            attrs = func_match.group(1)
            op_match = op_pattern.search(attrs)
            dtype_match = dtype_pattern.search(attrs)
            if not op_match or not dtype_match:
                continue
            hits[op_match.group(1)][dtype_match.group(1)].add(template.name)
    return {
        op: {dtype: sorted(files) for dtype, files in sorted(dtype_map.items())}
        for op, dtype_map in sorted(hits.items())
    }


def collect_op_dtypes_from_text(
    text: str, implemented_ops: dict[str, list[str]]
) -> dict[str, set[str]]:
    hits: dict[str, set[str]] = defaultdict(set)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//"):
            continue
        if "pto." not in line:
            continue
        for op, manifest_dtypes in implemented_ops.items():
            if f"pto.{op}" not in line:
                continue
            line_dtypes = set(DTYPE_PATTERN.findall(line))
            if line_dtypes:
                hits[op].update(dtype for dtype in line_dtypes if dtype in manifest_dtypes)
    return hits


def merge_hit_maps(
    dst: dict[str, dict[str, set[str]]],
    src: dict[str, set[str]],
    source_label: str,
) -> None:
    for op, dtypes in src.items():
        for dtype in dtypes:
            dst[op][dtype].add(source_label)


def collect_lowering_hits(
    test_dir: Path, implemented_ops: dict[str, list[str]]
) -> dict[str, dict[str, list[str]]]:
    hits: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    harness_refs: dict[str, list[str]] = defaultdict(list)
    for harness in sorted(test_dir.glob("*.mlir")):
        text = harness.read_text()
        if "--op-lib-dir" not in text:
            continue
        for fixture in sorted(test_dir.glob("*.pto")):
            if fixture.name in text:
                harness_refs[fixture.name].append(harness.name)
        merge_hit_maps(hits, collect_op_dtypes_from_text(text, implemented_ops), harness.name)

    for fixture in sorted(test_dir.glob("*.pto")):
        used_by = sorted(set(harness_refs.get(fixture.name, [])))
        if not used_by:
            continue
        text = fixture.read_text()
        use_case = f"{fixture.name} <- {', '.join(used_by)}"
        merge_hit_maps(hits, collect_op_dtypes_from_text(text, implemented_ops), use_case)

    return {
        op: {dtype: sorted(entries) for dtype, entries in sorted(dtype_map.items())}
        for op, dtype_map in sorted(hits.items())
    }


def main() -> int:
    args = parse_args()
    implemented_ops = load_implemented_ops(args.manifest)
    template_hits = collect_template_hits(args.template_dir)
    lowering_hits = collect_lowering_hits(args.test_dir, implemented_ops)

    missing_templates: dict[str, list[str]] = {}
    missing_lowering: dict[str, list[str]] = {}

    for op, dtypes in implemented_ops.items():
        template_dtype_hits = template_hits.get(op, {})
        lowering_dtype_hits = lowering_hits.get(op, {})
        missing_template_dtypes = [dtype for dtype in dtypes if dtype not in template_dtype_hits]
        missing_lowering_dtypes = [dtype for dtype in dtypes if dtype not in lowering_dtype_hits]
        if missing_template_dtypes:
            missing_templates[op] = missing_template_dtypes
        if missing_lowering_dtypes:
            missing_lowering[op] = missing_lowering_dtypes

    if missing_templates or missing_lowering:
        if missing_templates:
            print("implemented ops missing concrete template dtype coverage:", file=sys.stderr)
            for op, dtypes in sorted(missing_templates.items()):
                print(f"  - {op}: missing dtypes {', '.join(dtypes)}", file=sys.stderr)
        if missing_lowering:
            print("implemented ops missing lowering dtype coverage:", file=sys.stderr)
            for op, dtypes in sorted(missing_lowering.items()):
                print(f"  - {op}: missing dtypes {', '.join(dtypes)}", file=sys.stderr)
        return 1

    print(
        f"implemented ops aligned: {len(implemented_ops)} ops with dtype-level "
        "concrete templates and lowering use cases"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
