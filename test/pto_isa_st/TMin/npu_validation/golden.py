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
    single_output,
    write_buffers,
    write_golden,
)

CASES = {
    "tmin_float_64x64_64x64_64x64_64x64": {
        "dtype": "f32",
        "dst": (64, 64),
        "src0": (64, 64),
        "src1": (64, 64),
        "valid": (64, 64),
    },
    "tmin_int32_64x64_64x64_64x64_64x64": {
        "dtype": "i32",
        "dst": (64, 64),
        "src0": (64, 64),
        "src1": (64, 64),
        "valid": (64, 64),
    },
    "tmin_int16_64x64_64x64_64x64_64x64": {
        "dtype": "i16",
        "dst": (64, 64),
        "src0": (64, 64),
        "src1": (64, 64),
        "valid": (64, 64),
    },
    "tmin_half_16x256_16x256_16x256_16x256": {
        "dtype": "f16",
        "dst": (16, 256),
        "src0": (16, 256),
        "src1": (16, 256),
        "valid": (16, 256),
    },
    "tmin_half_16x64_16x128_16x128_16x64": {
        "dtype": "f16",
        "dst": (16, 64),
        "src0": (16, 128),
        "src1": (16, 128),
        "valid": (16, 64),
    },
    "tmin_float_16x32_16x64_16x32_16x32": {
        "dtype": "f32",
        "dst": (16, 32),
        "src0": (16, 64),
        "src1": (16, 32),
        "valid": (16, 32),
    },
    "tmin_int16_32x128_32x128_32x256_32x128": {
        "dtype": "i16",
        "dst": (32, 128),
        "src0": (32, 128),
        "src1": (32, 256),
        "valid": (32, 128),
    },
    "tmin_int32_16x32_16x64_16x32_16x32": {
        "dtype": "i32",
        "dst": (16, 32),
        "src0": (16, 64),
        "src1": (16, 32),
        "valid": (16, 32),
    },
    "tmin_half_16x64_16x128_16x128_16x63": {
        "dtype": "f16",
        "dst": (16, 64),
        "src0": (16, 128),
        "src1": (16, 128),
        "valid": (16, 63),
    },
    "tmin_float_16x32_16x64_16x32_16x31": {
        "dtype": "f32",
        "dst": (16, 32),
        "src0": (16, 64),
        "src1": (16, 32),
        "valid": (16, 31),
    },
    "tmin_int16_32x128_32x128_32x256_32x127": {
        "dtype": "i16",
        "dst": (32, 128),
        "src0": (32, 128),
        "src1": (32, 256),
        "valid": (32, 127),
    },
    "tmin_int32_16x32_16x64_16x32_16x31": {
        "dtype": "i32",
        "dst": (16, 32),
        "src0": (16, 64),
        "src1": (16, 32),
        "valid": (16, 31),
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
    generator = _rng(19)

    if "binary" == "binary":
        lhs_name, rhs_name = meta.inputs
        src0_rows, src0_cols = spec["src0"]
        src1_rows, src1_cols = spec["src1"]
        lhs = _randint_array(generator, (src0_rows, src0_cols), dtype)
        rhs = _randint_array(generator, (src1_rows, src1_cols), dtype)
        buffers[lhs_name] = lhs.reshape(-1)
        buffers[rhs_name] = rhs.reshape(-1)
        out[:valid_rows, :valid_cols] = np.minimum(
            lhs[:valid_rows, :valid_cols], rhs[:valid_rows, :valid_cols]
        )
    else:
        raise ValueError("unsupported binary op")

    write_buffers(meta, buffers)
    write_golden(meta, {out_name: out.reshape(-1)})


if __name__ == "__main__":
    main()
