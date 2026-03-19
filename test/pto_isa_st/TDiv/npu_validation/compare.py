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

from validation_runtime import compare_outputs, load_case_meta, single_output

CASE_DTYPES = {
    "tdiv_float_64x64_64x64_64x64_64x64": "f32",
    "tdiv_int32_64x64_64x64_64x64_64x64": "i32",
    "tdiv_int16_64x64_64x64_64x64_64x64": "i16",
    "tdiv_half_16x256_16x256_16x256_16x256": "f16",
    "tdiv_half_16x64_16x128_16x128_16x64": "f16",
    "tdiv_float_16x32_16x64_16x32_16x32": "f32",
    "tdiv_int16_32x128_32x128_32x256_32x128": "i16",
    "tdiv_int32_16x32_16x64_16x32_16x32": "i32",
    "tdiv_half_16x64_16x128_16x128_16x63": "f16",
    "tdiv_float_16x32_16x64_16x32_16x31": "f32",
    "tdiv_int16_32x128_32x128_32x256_32x127": "i16",
    "tdiv_int32_16x32_16x64_16x32_16x31": "i32",
}

NP_TYPES = {
    "f32": np.float32,
    "f16": np.float16,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "u32": np.uint32,
    "u16": np.uint16,
    "u8": np.uint8,
}


def _case_name():
    cwd_name = Path.cwd().name
    if cwd_name in CASE_DTYPES:
        return cwd_name
    return Path(__file__).resolve().parent.name


if __name__ == "__main__":
    meta = load_case_meta()
    _ = single_output(meta)
    compare_outputs(NP_TYPES[CASE_DTYPES[_case_name()]], atol=0.001)
