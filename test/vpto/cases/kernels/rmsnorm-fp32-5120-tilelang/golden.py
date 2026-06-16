#!/usr/bin/env python3

from pathlib import Path

import numpy as np


BATCH = 4096
HIDDEN = 5120
EPS = np.float32(1.0e-6)


def main() -> None:
    output_dir = Path(".")
    rng = np.random.default_rng(20260605)

    x = (rng.standard_normal((BATCH, HIDDEN), dtype=np.float32) * np.float32(0.25)).astype(
        np.float32, copy=False
    )
    w = (rng.standard_normal(HIDDEN, dtype=np.float32) * np.float32(0.5)).astype(
        np.float32, copy=False
    )

    sum_sq = np.sum(x * x, axis=1, dtype=np.float32)
    rstd = (np.float32(1.0) / np.sqrt(sum_sq / np.float32(HIDDEN) + EPS)).astype(np.float32)
    y = (x * rstd[:, None] * w[None, :]).astype(np.float32)

    x.reshape(-1).tofile(output_dir / "v1.bin")
    np.zeros_like(y).reshape(-1).tofile(output_dir / "v2.bin")
    w.tofile(output_dir / "v3.bin")
    np.zeros_like(rstd).tofile(output_dir / "v4.bin")
    y.reshape(-1).tofile(output_dir / "golden_v2.bin")
    rstd.tofile(output_dir / "golden_v4.bin")


if __name__ == "__main__":
    main()
