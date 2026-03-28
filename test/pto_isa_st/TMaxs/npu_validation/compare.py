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
    "tmaxs_float_64x64_32x32_32x32": "f32",
    "tmaxs_float_128x128_64x64_64x64": "f32",
    "tmaxs_float_60x128_64x64_60x60_padmax": "f32",
    "tmaxs_float_16x200_20x512_16x200_padmax": "f32",
    "tmaxs_float_1x3600_2x4096_1x3600_padmax": "f32",
    "tmaxs_int32_32x32_32x32_32x32": "i32",
    "tmaxs_uint32_32x32_32x32_32x32": "u32",
    "tmaxs_int16_32x128_32x128_32x128": "i16",
    "tmaxs_uint16_32x128_32x128_32x128": "u16",
    "tmaxs_int8_32x128_32x128_32x128": "i8",
    "tmaxs_uint8_32x128_32x128_32x128": "u8",
    "tmaxs_half_16x256_20x224_16x200_padmax": "f16",
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
    compare_outputs(NP_TYPES[CASE_DTYPES[_case_name()]], atol=0.0001)
