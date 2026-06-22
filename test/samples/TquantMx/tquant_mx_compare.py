#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root directory of this software repository for the full text of the License.

import sys
import numpy as np


def main():
    # dst: fp8 e4m3fn packed as int8, [16, 32] = 512 bytes
    golden = np.fromfile("golden.bin", dtype=np.int8).reshape(16, 32)
    output = np.fromfile("output.bin", dtype=np.int8).reshape(16, 32)

    if not np.array_equal(golden, output):
        diff = golden.astype(np.int32) - output.astype(np.int32)
        idx = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        print(
            f"[ERROR] dst mismatch at {idx}: golden={int(golden[idx])} "
            f"output={int(output[idx])} diff={int(diff[idx])}"
        )
        sys.exit(2)

    # exp: ui8 [1, 16]
    exp_golden = np.fromfile("exp.bin", dtype=np.uint8).reshape(1, 16)
    exp_output = np.fromfile("exp_out.bin", dtype=np.uint8).reshape(1, 16)
    if not np.array_equal(exp_golden, exp_output):
        print(f"[ERROR] exp mismatch: golden={exp_golden.tolist()} output={exp_output.tolist()}")
        sys.exit(2)

    # max: f32 [1, 16]
    max_golden = np.fromfile("max.bin", dtype=np.float32).reshape(1, 16)
    max_output = np.fromfile("max_out.bin", dtype=np.float32).reshape(1, 16)
    if not np.allclose(max_golden, max_output, atol=1e-5, rtol=1e-5):
        print(f"[ERROR] max mismatch: golden={max_golden.tolist()} output={max_output.tolist()}")
        sys.exit(2)

    # scaling: f32 [1, 16] per-group reciprocal scale
    sc_golden = np.fromfile("scaling.bin", dtype=np.float32).reshape(1, 16)
    sc_output = np.fromfile("scaling_out.bin", dtype=np.float32).reshape(1, 16)
    if not np.allclose(sc_golden, sc_output, atol=1e-5, rtol=1e-5):
        print(f"[ERROR] scaling mismatch: golden={sc_golden.tolist()} output={sc_output.tolist()}")
        sys.exit(2)

    print("[INFO] compare passed")


if __name__ == "__main__":
    main()
