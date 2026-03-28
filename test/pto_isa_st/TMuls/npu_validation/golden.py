#!/usr/bin/python3
import numpy as np
from pathlib import Path
import sys

for search_root in (
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parents[1],
):
    if (search_root / "validation_runtime.py").is_file():
        sys.path.insert(0, str(search_root))
        break

from validation_runtime import (
    default_buffers,
    load_case_meta,
    scalar_input_names,
    single_output,
    write_buffers,
    write_golden,
)

CASES = {
    "tmuls_float_32x128_32x64_32x64": {
        "dtype": "f32",
        "dst": (32, 128),
        "src": (32, 64),
        "valid": (32, 64),
        "pad": "null",
    },
    "tmuls_half_63x128_63x64_63x64": {
        "dtype": "f16",
        "dst": (63, 128),
        "src": (63, 64),
        "valid": (63, 64),
        "pad": "null",
    },
    "tmuls_int32_31x256_31x128_31x128": {
        "dtype": "i32",
        "dst": (31, 256),
        "src": (31, 128),
        "valid": (31, 128),
        "pad": "null",
    },
    "tmuls_int16_15x192_15x192_15x192": {
        "dtype": "i16",
        "dst": (15, 192),
        "src": (15, 192),
        "valid": (15, 192),
        "pad": "null",
    },
    "tmuls_float_7x512_7x448_7x448": {
        "dtype": "f32",
        "dst": (7, 512),
        "src": (7, 448),
        "valid": (7, 448),
        "pad": "null",
    },
    "tmuls_float_256x32_256x16_256x16": {
        "dtype": "f32",
        "dst": (256, 32),
        "src": (256, 16),
        "valid": (256, 16),
        "pad": "null",
    },
}


def _case_name():
    cwd_name = Path.cwd().name
    if cwd_name in CASES:
        return cwd_name
    return Path(__file__).resolve().parent.name


def _rng(seed):
    return np.random.RandomState(seed)


def _randint_array(generator, shape, dtype):
    return generator.randint(1, 10, size=shape).astype(dtype)


def _uniform_array(generator, shape, dtype, low, high):
    return generator.uniform(low=low, high=high, size=shape).astype(dtype)


def main():
    spec = CASES[_case_name()]
    meta = load_case_meta()
    buffers = default_buffers(meta)
    out_name = single_output(meta)
    dtype = np.dtype(meta.np_types[out_name])
    valid_rows, valid_cols = spec["valid"]
    out = np.zeros((valid_rows, valid_cols), dtype=dtype)
    generator = _rng(23)

    if "scalar" == "binary":
        lhs_name, rhs_name = meta.inputs
        src0_rows, src0_cols = spec["src0"]
        src1_rows, src1_cols = spec["src1"]
        lhs = _randint_array(generator, (src0_rows, src0_cols), dtype)
        rhs = _randint_array(generator, (src1_rows, src1_cols), dtype)
        buffers[lhs_name] = lhs.reshape(-1)
        buffers[rhs_name] = rhs.reshape(-1)
    else:
        src_name, scalar_name = scalar_input_names(meta)
        src_rows, src_cols = spec["src"]
        float_like = spec["dtype"] in {"f32", "f16"}
        if "tmuls" in {"tadds", "tsubs", "tmuls", "tdivs"}:
            src = _uniform_array(generator, (src_rows, src_cols), dtype, -8.0, 8.0)
            scalar_arr = _uniform_array(generator, (1,), dtype, -8.0, 8.0)
        else:
            if float_like:
                src = _uniform_array(
                    generator, (src_rows, src_cols), dtype, -13.013, 130.013
                )
                scalar_arr = _uniform_array(generator, (1,), dtype, -13.013, 130.013)
            else:
                src = _randint_array(generator, (src_rows, src_cols), dtype)
                scalar_arr = _randint_array(generator, (1,), dtype)
        buffers[src_name] = src.reshape(-1)
        buffers[scalar_name] = scalar_arr.reshape(-1)
        scalar = scalar_arr.reshape(-1)[0]
        out[:valid_rows, :valid_cols] = src[:valid_rows, :valid_cols] * scalar

    write_buffers(meta, buffers)
    write_golden(meta, {out_name: out.reshape(-1)})


if __name__ == "__main__":
    main()
