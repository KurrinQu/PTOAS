#!/usr/bin/env python3

import os
import sys

import numpy as np


BATCH = 4096
HIDDEN = 4096


def compare_bin(golden_path: str, output_path: str, shape, atol: float, rtol: float) -> bool:
    golden = np.fromfile(golden_path, dtype=np.float32)
    output = np.fromfile(output_path, dtype=np.float32)
    expected_elems = int(np.prod(shape))
    if golden.size != expected_elems or output.size != expected_elems:
        print(
            f"[ERROR] Shape mismatch for {output_path}: "
            f"golden={golden.size}, output={output.size}, expected={expected_elems}"
        )
        return False

    golden = golden.reshape(shape)
    output = output.reshape(shape)
    if np.allclose(golden, output, atol=atol, rtol=rtol):
        return True

    abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
    idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
    print(
        f"[ERROR] Mismatch for {output_path}: max diff={abs_diff[idx]} at idx={idx}, "
        f"golden={golden[idx]}, output={output[idx]}"
    )
    return False


def main() -> None:
    strict = os.environ.get("COMPARE_STRICT", "0") == "1"
    y_atol = 2.0e-3 if strict else 5.0e-3
    rstd_atol = 2.0e-4 if strict else 1.0e-3
    ok = True
    ok = compare_bin("golden_v2.bin", "v2.bin", (BATCH, HIDDEN), y_atol, y_atol) and ok
    ok = compare_bin("golden_v4.bin", "v4.bin", (BATCH,), rstd_atol, rstd_atol) and ok
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
