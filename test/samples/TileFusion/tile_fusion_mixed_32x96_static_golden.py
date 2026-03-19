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

SCALE = np.float32(0.1767767)
NEG_CLAMP = np.float32(-4.0001)
POS_CLAMP = np.float32(4.0001)
C2 = np.float32(0.5)
C3 = np.float32(0.1666667)
C4 = np.float32(0.04166667)
BIAS = np.float32(1.0001)
DIVISOR = np.float32(96.0001)


def main():
    meta = load_case_meta()
    src_name = meta.inputs[0]

    generator = rng()
    src = float_values(generator, meta.elem_counts[src_name], style="signed_small")

    buffers = default_buffers(meta)
    buffers[src_name] = src
    write_buffers(meta, buffers)

    clamped = np.minimum(np.maximum(src * SCALE, NEG_CLAMP), POS_CLAMP).astype(
        np.float32
    )
    tmp0 = (clamped * clamped).astype(np.float32)
    tmp1 = (tmp0 * clamped).astype(np.float32)
    tmp2 = (tmp1 * clamped).astype(np.float32)
    out = ((clamped + tmp0 * C2 + tmp1 * C3 + tmp2 * C4 + BIAS) / DIVISOR).astype(
        np.float32
    )

    write_golden(meta, {single_output(meta): out})


if __name__ == "__main__":
    main()
