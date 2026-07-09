#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Generate input data and golden output for mixed lowp/int packed vector copy test.

Each buffer has 4 packed elements: [lane0_in, lane1_in, lane0_out, lane1_out].
The kernel copies input region (offsets 0..1) to output region (offsets 2..3),
so golden output = input (output region matches input region).

- hif8x2:  raw 16-bit payload per element, stored as uint16
- i8x2:    2 bytes per vector, stored as uint8 (flat byte array)
- fp8e4x2: raw 16-bit payload per element, stored as uint16
- fp8e5x2: raw 16-bit payload per element, stored as uint16
"""

import argparse
from pathlib import Path

import numpy as np

LANES = 2
ELEMS = 4  # 2 input + 2 output


def generate(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # -- hif8x2: 4 × uint16 (8 bytes total) --
    hif8x2 = np.zeros(ELEMS, dtype=np.uint16)
    hif8x2[0] = 0xABCD
    hif8x2[1] = 0x1234
    hif8x2[2] = 0xCDCD
    hif8x2[3] = 0xCDCD
    golden_hif8x2 = np.copy(hif8x2)
    golden_hif8x2[2] = hif8x2[0]
    golden_hif8x2[3] = hif8x2[1]
    hif8x2.tofile(output_dir / "hif8x2.bin")
    golden_hif8x2.tofile(output_dir / "golden_hif8x2.bin")

    # -- i8x2: 4 vectors × 2 bytes = 8 bytes, stored as uint8 --
    i8x2 = np.zeros(ELEMS * 2, dtype=np.uint8)
    i8x2[0] = 0x01; i8x2[1] = 0x02
    i8x2[2] = 0x11; i8x2[3] = 0x12
    i8x2[4] = 0xCD; i8x2[5] = 0xCD
    i8x2[6] = 0xCD; i8x2[7] = 0xCD
    golden_i8x2 = np.copy(i8x2)
    golden_i8x2[4] = i8x2[0]; golden_i8x2[5] = i8x2[1]
    golden_i8x2[6] = i8x2[2]; golden_i8x2[7] = i8x2[3]
    i8x2.tofile(output_dir / "i8x2.bin")
    golden_i8x2.tofile(output_dir / "golden_i8x2.bin")

    # -- fp8e4x2: 4 × uint16 (8 bytes total) --
    fp8e4x2 = np.zeros(ELEMS, dtype=np.uint16)
    fp8e4x2[0] = 0x4251
    fp8e4x2[1] = 0x3A60
    fp8e4x2[2] = 0xCDCD
    fp8e4x2[3] = 0xCDCD
    golden_fp8e4x2 = np.copy(fp8e4x2)
    golden_fp8e4x2[2] = fp8e4x2[0]
    golden_fp8e4x2[3] = fp8e4x2[1]
    fp8e4x2.tofile(output_dir / "fp8e4x2.bin")
    golden_fp8e4x2.tofile(output_dir / "golden_fp8e4x2.bin")

    # -- fp8e5x2: 4 × uint16 (8 bytes total) --
    fp8e5x2 = np.zeros(ELEMS, dtype=np.uint16)
    fp8e5x2[0] = 0x5210
    fp8e5x2[1] = 0x3B80
    fp8e5x2[2] = 0xCDCD
    fp8e5x2[3] = 0xCDCD
    golden_fp8e5x2 = np.copy(fp8e5x2)
    golden_fp8e5x2[2] = fp8e5x2[0]
    golden_fp8e5x2[3] = fp8e5x2[1]
    fp8e5x2.tofile(output_dir / "fp8e5x2.bin")
    golden_fp8e5x2.tofile(output_dir / "golden_fp8e5x2.bin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    generate(args.output_dir)


if __name__ == "__main__":
    main()
