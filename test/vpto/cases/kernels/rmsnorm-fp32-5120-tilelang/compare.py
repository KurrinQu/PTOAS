#!/usr/bin/env python3

import os
import sys

import numpy as np


BATCH = 4096
HIDDEN = 5120


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

    finite_golden = np.isfinite(golden)
    finite_output = np.isfinite(output)
    if not np.all(finite_output):
        bad_idx = np.argwhere(~finite_output)[0]
        bad_idx = tuple(int(i) for i in bad_idx)
        print(
            f"[ERROR] Non-finite output for {output_path}: first idx={bad_idx}, "
            f"golden={golden[bad_idx]}, output={output[bad_idx]}"
        )
    if not np.all(finite_golden):
        bad_idx = np.argwhere(~finite_golden)[0]
        bad_idx = tuple(int(i) for i in bad_idx)
        print(
            f"[ERROR] Non-finite golden for {golden_path}: first idx={bad_idx}, "
            f"golden={golden[bad_idx]}, output={output[bad_idx]}"
        )

    abs_diff = np.abs(golden.astype(np.float64) - output.astype(np.float64))
    finite_diff = np.where(np.isfinite(abs_diff), abs_diff, -1.0)
    idx = np.unravel_index(int(np.argmax(finite_diff)), finite_diff.shape)
    if finite_diff[idx] >= 0.0:
        print(
            f"[ERROR] Mismatch for {output_path}: max finite diff={finite_diff[idx]} at idx={idx}, "
            f"golden={golden[idx]}, output={output[idx]}"
        )
        if len(shape) == 2:
            row = idx[0]
            row_diff = abs_diff[row]
            finite_row_diff = np.where(np.isfinite(row_diff), row_diff, -1.0)
            row_col = int(np.argmax(finite_row_diff))
            print(
                f"[ERROR] Row {row} max finite diff={finite_row_diff[row_col]} at col={row_col}, "
                f"golden={golden[row, row_col]}, output={output[row, row_col]}"
            )
    print(
        f"[ERROR] finite counts for {output_path}: "
        f"golden={finite_golden.sum()}/{golden.size}, output={finite_output.sum()}/{output.size}"
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
