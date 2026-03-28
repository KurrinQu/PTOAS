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
    "tmins_float_64x64_32x32_32x32": {
        "dtype": "f32",
        "dst": (64, 64),
        "src": (32, 32),
        "valid": (32, 32),
        "pad": "null",
    },
    "tmins_float_128x128_64x64_64x64": {
        "dtype": "f32",
        "dst": (128, 128),
        "src": (64, 64),
        "valid": (64, 64),
        "pad": "null",
    },
    "tmins_float_60x128_64x64_60x60_padmax": {
        "dtype": "f32",
        "dst": (60, 128),
        "src": (64, 64),
        "valid": (60, 60),
        "pad": "max",
    },
    "tmins_float_16x200_20x512_16x200_padmax": {
        "dtype": "f32",
        "dst": (16, 200),
        "src": (20, 512),
        "valid": (16, 200),
        "pad": "max",
    },
    "tmins_float_1x3600_2x4096_1x3600_padmax": {
        "dtype": "f32",
        "dst": (1, 3600),
        "src": (2, 4096),
        "valid": (1, 3600),
        "pad": "max",
    },
    "tmins_int32_32x32_32x32_32x32": {
        "dtype": "i32",
        "dst": (32, 32),
        "src": (32, 32),
        "valid": (32, 32),
        "pad": "null",
    },
    "tmins_uint32_32x32_32x32_32x32": {
        "dtype": "u32",
        "dst": (32, 32),
        "src": (32, 32),
        "valid": (32, 32),
        "pad": "null",
    },
    "tmins_int16_32x128_32x128_32x128": {
        "dtype": "i16",
        "dst": (32, 128),
        "src": (32, 128),
        "valid": (32, 128),
        "pad": "null",
    },
    "tmins_uint16_32x128_32x128_32x128": {
        "dtype": "u16",
        "dst": (32, 128),
        "src": (32, 128),
        "valid": (32, 128),
        "pad": "null",
    },
    "tmins_int8_32x128_32x128_32x128": {
        "dtype": "i8",
        "dst": (32, 128),
        "src": (32, 128),
        "valid": (32, 128),
        "pad": "null",
    },
    "tmins_uint8_32x128_32x128_32x128": {
        "dtype": "u8",
        "dst": (32, 128),
        "src": (32, 128),
        "valid": (32, 128),
        "pad": "null",
    },
    "tmins_half_16x256_20x224_16x200_padmax": {
        "dtype": "f16",
        "dst": (16, 256),
        "src": (20, 224),
        "valid": (16, 200),
        "pad": "max",
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
        if "tmins" in {"tadds", "tsubs", "tmuls", "tdivs"}:
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
        out[:valid_rows, :valid_cols] = np.minimum(
            src[:valid_rows, :valid_cols], scalar
        )

    write_buffers(meta, buffers)
    write_golden(meta, {out_name: out.reshape(-1)})


if __name__ == "__main__":
    main()
