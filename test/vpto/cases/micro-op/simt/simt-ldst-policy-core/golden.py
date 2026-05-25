#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

import argparse
from pathlib import Path

import numpy as np

ELEMS = 1024


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    v1 = np.full(ELEMS, -1, dtype=np.int32)
    golden_v1 = np.full(ELEMS, -1, dtype=np.int32)
    v2 = np.full(ELEMS, 0xBC00, dtype=np.uint16)
    v3 = np.full(ELEMS, 0xBF80, dtype=np.uint16)
    v4 = np.full(ELEMS, -1, dtype=np.int8)
    v5 = np.full(ELEMS, -1, dtype=np.int16)
    v6 = np.full(ELEMS, -1, dtype=np.int64)
    v7 = np.full(ELEMS, -1.0, dtype=np.float32)
    v8 = np.full(ELEMS, -1.0, dtype=np.float64)
    golden_v2 = v2.copy()
    golden_v3 = v3.copy()
    golden_v4 = v4.copy()
    golden_v5 = v5.copy()
    golden_v6 = v6.copy()
    golden_v7 = v7.copy()
    golden_v8 = v8.copy()
    inputs = np.array([0x10203040, -1234567], dtype=np.int32)
    v1[:2] = inputs
    golden_v1[:2] = inputs
    golden_v1[2] = inputs[0]
    golden_v1[3] = inputs[1]
    golden_v1[4] = np.int32(inputs[0] + inputs[1])
    v2[:2] = np.array([0x3E00, 0xC000], dtype=np.uint16)  # f16: 1.5, -2.0
    v3[:2] = np.array([0x3FC0, 0xC000], dtype=np.uint16)  # bf16: 1.5, -2.0
    golden_v2[:2] = v2[:2]
    golden_v2[2:4] = v2[:2]
    golden_v3[:2] = v3[:2]
    golden_v3[2:4] = v3[:2]
    v4[:2] = np.array([0x12, -0x34], dtype=np.int8)
    v5[:2] = np.array([0x1234, -0x3456], dtype=np.int16)
    v6[:2] = np.array([0x1020304050607080, -0x102030405060708], dtype=np.int64)
    v7[:2] = np.array([2.5, -3.5], dtype=np.float32)
    v8[:2] = np.array([4.5, -5.5], dtype=np.float64)
    golden_v4[:2] = v4[:2]
    golden_v4[2:4] = v4[:2]
    golden_v5[:2] = v5[:2]
    golden_v5[2:4] = v5[:2]
    golden_v6[:2] = v6[:2]
    golden_v6[2:4] = v6[:2]
    golden_v7[:2] = v7[:2]
    golden_v7[2:4] = v7[:2]
    golden_v8[:2] = v8[:2]
    golden_v8[2:4] = v8[:2]
    v1.tofile(output_dir / "v1.bin")
    v2.tofile(output_dir / "v2.bin")
    v3.tofile(output_dir / "v3.bin")
    v4.tofile(output_dir / "v4.bin")
    v5.tofile(output_dir / "v5.bin")
    v6.tofile(output_dir / "v6.bin")
    v7.tofile(output_dir / "v7.bin")
    v8.tofile(output_dir / "v8.bin")
    golden_v1.tofile(output_dir / "golden_v1.bin")
    golden_v2.tofile(output_dir / "golden_v2.bin")
    golden_v3.tofile(output_dir / "golden_v3.bin")
    golden_v4.tofile(output_dir / "golden_v4.bin")
    golden_v5.tofile(output_dir / "golden_v5.bin")
    golden_v6.tofile(output_dir / "golden_v6.bin")
    golden_v7.tofile(output_dir / "golden_v7.bin")
    golden_v8.tofile(output_dir / "golden_v8.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
