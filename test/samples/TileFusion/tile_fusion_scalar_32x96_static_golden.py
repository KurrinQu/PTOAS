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
    float_values,
    load_case_meta,
    rng,
    single_output,
    write_buffers,
    write_golden,
)

MUL = np.float32(1.25)
ADD = np.float32(-0.75)
MAXV = np.float32(-1.5)
MINV = np.float32(2.5)
DIV = np.float32(0.5)


def main():
    meta = load_case_meta()
    src_name = meta.inputs[0]

    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style="signed_small")

    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    out = np.minimum(np.maximum(src * MUL + ADD, MAXV), MINV).astype(np.float32)
    out = (out / DIV).astype(np.float32)

    write_golden(meta, {single_output(meta): out})


if __name__ == "__main__":
    main()
