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
    "tmuls_float_32x128_32x64_32x64": "f32",
    "tmuls_half_63x128_63x64_63x64": "f16",
    "tmuls_int32_31x256_31x128_31x128": "i32",
    "tmuls_int16_15x192_15x192_15x192": "i16",
    "tmuls_float_7x512_7x448_7x448": "f32",
    "tmuls_float_256x32_256x16_256x16": "f32",
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
