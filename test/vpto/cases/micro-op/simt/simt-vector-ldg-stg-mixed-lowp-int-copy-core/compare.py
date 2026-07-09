#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Compare raw binary output against golden for mixed lowp/int packed vector test.

Each buffer is compared with np.array_equal (exact match, no tolerance)
since we are verifying raw payload roundtrip, not floating-point computation.
"""

import os
import sys

import numpy as np

STRICT = os.getenv("COMPARE_STRICT", "1") != "0"


def _check_one(name: str, golden_path: str, out_path: str, dtype) -> bool:
    """Compare one buffer.  Returns True on pass, False on failure."""
    golden = np.fromfile(golden_path, dtype=dtype)
    out = np.fromfile(out_path, dtype=dtype)

    if golden.shape != out.shape:
        print(
            f"[ERROR] {name} shape mismatch: golden {golden.shape} vs out {out.shape}"
        )
        return False

    ok = np.array_equal(golden, out)
    if not ok:
        idxs = np.nonzero(golden != out)[0]
        idx = int(idxs[0]) if idxs.size else 0
        print(
            f"[ERROR] {name} mismatch at idx={idx}:"
            f" golden=0x{int(golden[idx]):04X}, out=0x{int(out[idx]):04X}"
        )
        if idxs.size > 1:
            print(f"        ({idxs.size} total mismatches, first shown)")
        return False

    print(f"[INFO] {name} compare passed")
    return True


def main():
    all_ok = True
    u16 = np.uint16

    all_ok &= _check_one("hif8x2", "golden_hif8x2.bin", "hif8x2.bin", u16)
    all_ok &= _check_one("i8x2", "golden_i8x2.bin", "i8x2.bin", np.uint8)
    all_ok &= _check_one("fp8e4x2", "golden_fp8e4x2.bin", "fp8e4x2.bin", u16)
    all_ok &= _check_one("fp8e5x2", "golden_fp8e5x2.bin", "fp8e5x2.bin", u16)

    if not all_ok:
        if STRICT:
            sys.exit(2)
        print("[WARN] compare failed (non-gating)")
    else:
        print("[INFO] all checks passed")


if __name__ == "__main__":
    main()
