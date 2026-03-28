#!/usr/bin/env python3
"""Generate PTOAS ST sample-style cases under test/pto_isa_st."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "test" / "pto_isa_st"


DTYPE_INFO = {
    "f32": {"label": "float", "np": "np.float32", "float_like": True},
    "f16": {"label": "half", "np": "np.float16", "float_like": True},
    "i32": {"label": "int32", "np": "np.int32", "float_like": False},
    "i16": {"label": "int16", "np": "np.int16", "float_like": False},
    "i8": {"label": "int8", "np": "np.int8", "float_like": False},
    "u32": {"label": "uint32", "np": "np.uint32", "float_like": False},
    "u16": {"label": "uint16", "np": "np.uint16", "float_like": False},
    "u8": {"label": "uint8", "np": "np.uint8", "float_like": False},
}


@dataclass(frozen=True)
class BinaryCase:
    dtype: str
    dst: tuple[int, int]
    src0: tuple[int, int]
    src1: tuple[int, int]
    valid: tuple[int, int]

    def base_name(self, op: str) -> str:
        label = DTYPE_INFO[self.dtype]["label"]
        return (
            f"{op}_{label}_"
            f"{self.dst[0]}x{self.dst[1]}_"
            f"{self.src0[0]}x{self.src0[1]}_"
            f"{self.src1[0]}x{self.src1[1]}_"
            f"{self.valid[0]}x{self.valid[1]}"
        )


@dataclass(frozen=True)
class ScalarCase:
    dtype: str
    dst: tuple[int, int]
    src: tuple[int, int]
    valid: tuple[int, int]
    pad: str = "null"

    def base_name(self, op: str) -> str:
        label = DTYPE_INFO[self.dtype]["label"]
        suffix = "" if self.pad == "null" else f"_pad{self.pad}"
        return (
            f"{op}_{label}_"
            f"{self.dst[0]}x{self.dst[1]}_"
            f"{self.src[0]}x{self.src[1]}_"
            f"{self.valid[0]}x{self.valid[1]}"
            f"{suffix}"
        )


COMMON_BINARY_CASES = [
    BinaryCase("f32", (64, 64), (64, 64), (64, 64), (64, 64)),
    BinaryCase("i32", (64, 64), (64, 64), (64, 64), (64, 64)),
    BinaryCase("i16", (64, 64), (64, 64), (64, 64), (64, 64)),
    BinaryCase("f16", (16, 256), (16, 256), (16, 256), (16, 256)),
    BinaryCase("f16", (16, 64), (16, 128), (16, 128), (16, 64)),
    BinaryCase("f32", (16, 32), (16, 64), (16, 32), (16, 32)),
    BinaryCase("i16", (32, 128), (32, 128), (32, 256), (32, 128)),
    BinaryCase("i32", (16, 32), (16, 64), (16, 32), (16, 32)),
    BinaryCase("f16", (16, 64), (16, 128), (16, 128), (16, 63)),
    BinaryCase("f32", (16, 32), (16, 64), (16, 32), (16, 31)),
    BinaryCase("i16", (32, 128), (32, 128), (32, 256), (32, 127)),
    BinaryCase("i32", (16, 32), (16, 64), (16, 32), (16, 31)),
]


BINARY_CASES = {
    "tadd": COMMON_BINARY_CASES
    + [BinaryCase("f16", (2, 128), (2, 128), (2, 128), (1, 106))],
    "tsub": COMMON_BINARY_CASES,
    "tmul": COMMON_BINARY_CASES,
    "tdiv": COMMON_BINARY_CASES,
    "tmax": COMMON_BINARY_CASES,
    "tmin": COMMON_BINARY_CASES,
}


SCALAR_CASES = {
    "tadds": [
        ScalarCase("f32", (32, 128), (32, 64), (32, 64)),
        ScalarCase("f16", (63, 128), (63, 64), (63, 64)),
        ScalarCase("i32", (31, 256), (31, 128), (31, 128)),
        ScalarCase("i16", (15, 192), (15, 192), (15, 192)),
        ScalarCase("f32", (7, 512), (7, 448), (7, 448)),
        ScalarCase("f32", (256, 32), (256, 16), (256, 16)),
    ],
    "tsubs": [
        ScalarCase("f32", (32, 128), (32, 64), (32, 64)),
        ScalarCase("f16", (63, 128), (63, 64), (63, 64)),
        ScalarCase("i32", (31, 256), (31, 128), (31, 128)),
        ScalarCase("i16", (15, 192), (15, 192), (15, 192)),
        ScalarCase("f32", (7, 512), (7, 448), (7, 448)),
        ScalarCase("f32", (256, 32), (256, 16), (256, 16)),
    ],
    "tmuls": [
        ScalarCase("f32", (32, 128), (32, 64), (32, 64)),
        ScalarCase("f16", (63, 128), (63, 64), (63, 64)),
        ScalarCase("i32", (31, 256), (31, 128), (31, 128)),
        ScalarCase("i16", (15, 192), (15, 192), (15, 192)),
        ScalarCase("f32", (7, 512), (7, 448), (7, 448)),
        ScalarCase("f32", (256, 32), (256, 16), (256, 16)),
    ],
    "tdivs": [
        ScalarCase("f32", (32, 128), (32, 64), (32, 64)),
        ScalarCase("f16", (63, 128), (63, 64), (63, 64)),
        ScalarCase("i32", (31, 256), (31, 128), (31, 128)),
        ScalarCase("i16", (15, 192), (15, 192), (15, 192)),
        ScalarCase("f32", (7, 512), (7, 448), (7, 448)),
        ScalarCase("f32", (256, 32), (256, 16), (256, 16)),
    ],
    "tmaxs": [
        ScalarCase("f32", (64, 64), (32, 32), (32, 32), "null"),
        ScalarCase("f32", (128, 128), (64, 64), (64, 64), "null"),
        ScalarCase("f32", (60, 128), (64, 64), (60, 60), "max"),
        ScalarCase("f32", (16, 200), (20, 512), (16, 200), "max"),
        ScalarCase("f32", (1, 3600), (2, 4096), (1, 3600), "max"),
        ScalarCase("i32", (32, 32), (32, 32), (32, 32), "null"),
        ScalarCase("u32", (32, 32), (32, 32), (32, 32), "null"),
        ScalarCase("i16", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("u16", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("i8", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("u8", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("f16", (16, 256), (20, 224), (16, 200), "max"),
    ],
    "tmins": [
        ScalarCase("f32", (64, 64), (32, 32), (32, 32), "null"),
        ScalarCase("f32", (128, 128), (64, 64), (64, 64), "null"),
        ScalarCase("f32", (60, 128), (64, 64), (60, 60), "max"),
        ScalarCase("f32", (16, 200), (20, 512), (16, 200), "max"),
        ScalarCase("f32", (1, 3600), (2, 4096), (1, 3600), "max"),
        ScalarCase("i32", (32, 32), (32, 32), (32, 32), "null"),
        ScalarCase("u32", (32, 32), (32, 32), (32, 32), "null"),
        ScalarCase("i16", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("u16", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("i8", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("u8", (32, 128), (32, 128), (32, 128), "null"),
        ScalarCase("f16", (16, 256), (20, 224), (16, 200), "max"),
    ],
}


DIR_NAMES = {
    "tadd": "TAdd",
    "tsub": "TSub",
    "tmul": "TMul",
    "tdiv": "TDiv",
    "tmax": "TMax",
    "tmin": "TMin",
    "tadds": "TAdds",
    "tsubs": "TSubs",
    "tmuls": "TMuls",
    "tdivs": "TDivs",
    "tmaxs": "TMaxs",
    "tmins": "TMins",
}


def _py_header() -> str:
    return dedent("""\
        from pathlib import Path
        import sys


        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        """)


def _build_script(op: str, case) -> str:
    if isinstance(case, BinaryCase):
        return _py_header() + dedent(f"""\
            from sample_utils.pto_isa_st_cases import build_binary_case


            def build():
                return build_binary_case(
                    kernel_name="{case.base_name(op)}_kernel",
                    op_name="{op}",
                    dtype_token="{case.dtype}",
                    dst_shape={case.dst},
                    src0_shape={case.src0},
                    src1_shape={case.src1},
                    valid_shape={case.valid},
                )


            if __name__ == "__main__":
                print(build())
            """)

    return _py_header() + dedent(f"""\
        from sample_utils.pto_isa_st_cases import build_scalar_case


        def build():
            return build_scalar_case(
                kernel_name="{case.base_name(op)}_kernel",
                op_name="{op}",
                dtype_token="{case.dtype}",
                dst_shape={case.dst},
                src_shape={case.src},
                valid_shape={case.valid},
                pad_value="{case.pad}",
            )


        if __name__ == "__main__":
            print(build())
        """)


def _golden_script(op: str, cases: list[BinaryCase] | list[ScalarCase]) -> str:
    mode = "binary" if isinstance(cases[0], BinaryCase) else "scalar"
    seed = (
        19
        if op in {"tadd", "tsub", "tmul", "tdiv", "tmax", "tmin", "tmaxs", "tmins"}
        else 23
    )
    case_lines: list[str] = []
    for case in cases:
        base_name = case.base_name(op)
        if mode == "binary":
            case_lines.append(
                f'    "{base_name}": {{"dtype": "{case.dtype}", "dst": {case.dst}, "src0": {case.src0}, "src1": {case.src1}, "valid": {case.valid}}},'
            )
        else:
            case_lines.append(
                f'    "{base_name}": {{"dtype": "{case.dtype}", "dst": {case.dst}, "src": {case.src}, "valid": {case.valid}, "pad": "{case.pad}"}},'
            )
    lines = [
        "#!/usr/bin/python3",
        "import numpy as np",
        "from pathlib import Path",
        "import sys",
        "",
        "for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):",
        '    if (search_root / "validation_runtime.py").is_file():',
        "        sys.path.insert(0, str(search_root))",
        "        break",
        "",
        "from validation_runtime import default_buffers, load_case_meta, single_output, write_buffers, write_golden",
        "",
        "",
        "CASES = {",
        *case_lines,
        "}",
        "",
        "",
        "def _case_name():",
        "    cwd_name = Path.cwd().name",
        "    if cwd_name in CASES:",
        "        return cwd_name",
        "    return Path(__file__).resolve().parent.name",
        "",
        "",
        "def _rng(seed):",
        "    return np.random.RandomState(seed)",
        "",
        "",
        "def _randint_array(generator, shape, dtype):",
        "    return generator.randint(1, 10, size=shape).astype(dtype)",
        "",
        "",
        "def _uniform_array(generator, shape, dtype, low, high):",
        "    return generator.uniform(low=low, high=high, size=shape).astype(dtype)",
        "",
        "",
        "def main():",
        "    spec = CASES[_case_name()]",
        "    meta = load_case_meta()",
        "    buffers = default_buffers(meta)",
        "    out_name = single_output(meta)",
        "    dtype = np.dtype(meta.np_types[out_name])",
        '    dst_rows, dst_cols = spec["dst"]',
        '    valid_rows, valid_cols = spec["valid"]',
        "    out = np.zeros((dst_rows, dst_cols), dtype=dtype)",
        f"    generator = _rng({seed})",
        "",
        f'    if "{mode}" == "binary":',
        "        lhs_name, rhs_name = meta.inputs",
        '        src0_rows, src0_cols = spec["src0"]',
        '        src1_rows, src1_cols = spec["src1"]',
        "        lhs = _randint_array(generator, (src0_rows, src0_cols), dtype)",
        "        rhs = _randint_array(generator, (src1_rows, src1_cols), dtype)",
        "        buffers[lhs_name] = lhs.reshape(-1)",
        "        buffers[rhs_name] = rhs.reshape(-1)",
    ]
    binary_exprs = {
        "tadd": "lhs[:valid_rows, :valid_cols] + rhs[:valid_rows, :valid_cols]",
        "tsub": "lhs[:valid_rows, :valid_cols] - rhs[:valid_rows, :valid_cols]",
        "tmul": "lhs[:valid_rows, :valid_cols] * rhs[:valid_rows, :valid_cols]",
        "tdiv": "lhs[:valid_rows, :valid_cols] / rhs[:valid_rows, :valid_cols]",
        "tmax": "np.maximum(lhs[:valid_rows, :valid_cols], rhs[:valid_rows, :valid_cols])",
        "tmin": "np.minimum(lhs[:valid_rows, :valid_cols], rhs[:valid_rows, :valid_cols])",
    }
    scalar_exprs = {
        "tadds": "src[:valid_rows, :valid_cols] + scalar",
        "tsubs": "src[:valid_rows, :valid_cols] - scalar",
        "tmuls": "src[:valid_rows, :valid_cols] * scalar",
        "tdivs": "src[:valid_rows, :valid_cols] / scalar",
        "tmaxs": "np.maximum(src[:valid_rows, :valid_cols], scalar)",
        "tmins": "np.minimum(src[:valid_rows, :valid_cols], scalar)",
    }
    if mode == "binary":
        lines.extend(
            [
                f"        out[:valid_rows, :valid_cols] = {binary_exprs[op]}",
                "    else:",
                '        raise ValueError("unsupported binary op")',
            ]
        )
    else:
        lines.extend(
            [
                "    else:",
                "        src_name, scalar_name = meta.inputs",
                '        src_rows, src_cols = spec["src"]',
                '        float_like = spec["dtype"] in {"f32", "f16"}',
                f'        if "{op}" in {{"tadds", "tsubs", "tmuls", "tdivs"}}:',
                "            src = _uniform_array(generator, (src_rows, src_cols), dtype, -8.0, 8.0)",
                "            scalar_arr = _uniform_array(generator, (1,), dtype, -8.0, 8.0)",
                "        else:",
                "            if float_like:",
                "                src = _uniform_array(generator, (src_rows, src_cols), dtype, -13.013, 130.013)",
                "                scalar_arr = _uniform_array(generator, (1,), dtype, -13.013, 130.013)",
                "            else:",
                "                src = _randint_array(generator, (src_rows, src_cols), dtype)",
                "                scalar_arr = _randint_array(generator, (1,), dtype)",
                "        buffers[src_name] = src.reshape(-1)",
                "        buffers[scalar_name] = scalar_arr.reshape(-1)",
                "        scalar = scalar_arr.reshape(-1)[0]",
                f"        out[:valid_rows, :valid_cols] = {scalar_exprs[op]}",
            ]
        )
    lines.extend(
        [
            "",
            "    write_buffers(meta, buffers)",
            "    write_golden(meta, {out_name: out.reshape(-1)})",
            "",
            "",
            'if __name__ == "__main__":',
            "    main()",
        ]
    )
    return "\n".join(lines) + "\n"


def _compare_script(op: str, cases: list[BinaryCase] | list[ScalarCase]) -> str:
    tol = 1e-4 if op in {"tmaxs", "tmins"} else 1e-3
    case_lines: list[str] = []
    for case in cases:
        case_lines.append(f'    "{case.base_name(op)}": "{case.dtype}",')
    lines = [
        "#!/usr/bin/python3",
        "import numpy as np",
        "from pathlib import Path",
        "import sys",
        "",
        "for search_root in (Path(__file__).resolve().parent, Path(__file__).resolve().parents[1]):",
        '    if (search_root / "validation_runtime.py").is_file():',
        "        sys.path.insert(0, str(search_root))",
        "        break",
        "",
        "from validation_runtime import compare_outputs, load_case_meta, single_output",
        "",
        "",
        "CASE_DTYPES = {",
        *case_lines,
        "}",
        "",
        "NP_TYPES = {",
        '    "f32": np.float32,',
        '    "f16": np.float16,',
        '    "i32": np.int32,',
        '    "i16": np.int16,',
        '    "i8": np.int8,',
        '    "u32": np.uint32,',
        '    "u16": np.uint16,',
        '    "u8": np.uint8,',
        "}",
        "",
        "",
        "def _case_name():",
        "    cwd_name = Path.cwd().name",
        "    if cwd_name in CASE_DTYPES:",
        "        return cwd_name",
        "    return Path(__file__).resolve().parent.name",
        "",
        "",
        'if __name__ == "__main__":',
        "    meta = load_case_meta()",
        "    _ = single_output(meta)",
        f"    compare_outputs(NP_TYPES[CASE_DTYPES[_case_name()]], atol={tol})",
    ]
    return "\n".join(lines) + "\n"


def _write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _emit_group(op: str, cases: list[BinaryCase] | list[ScalarCase]):
    out_dir = OUT_ROOT / DIR_NAMES[op]
    for case in cases:
        _write(out_dir / f"{case.base_name(op)}.py", _build_script(op, case))
    _write(out_dir / "npu_validation" / "golden.py", _golden_script(op, cases))
    _write(out_dir / "npu_validation" / "compare.py", _compare_script(op, cases))


def main():
    for op, cases in BINARY_CASES.items():
        _emit_group(op, cases)
    for op, cases in SCALAR_CASES.items():
        _emit_group(op, cases)


if __name__ == "__main__":
    main()
