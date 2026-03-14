#!/usr/bin/python3
# coding=utf-8

import numpy as np


def main():
    np.random.seed(19)

    # Inputs
    src = np.random.random(size=(16, 16)).astype(np.float32)
    dst = np.random.random(size=(32, 32)).astype(np.float32)

    # Kernel reads an initial output buffer; keep deterministic content.
    out_init = np.zeros((32, 32), dtype=np.float32)

    # Golden for TAssemble semantics:
    # dst[i + 8, j + 8] = src[i, j]
    golden = dst.copy()
    golden[8:24, 8:24] = src

    src.tofile("v1.bin")
    dst.tofile("v2.bin")
    out_init.tofile("v3.bin")
    golden.tofile("golden_v3.bin")


if __name__ == "__main__":
    main()
