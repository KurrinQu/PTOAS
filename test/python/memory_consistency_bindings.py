#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from mlir.ir import Context, InsertionPoint, Location, Module
from mlir.dialects import pto


def assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"missing {needle!r} in:\n{text}")


def main() -> None:
    with Context() as ctx, Location.unknown(ctx):
        pto.register_dialect(ctx, load=True)
        module = Module.create()
        with InsertionPoint(module.body):
            pto.CmoCleanOp(pto.AddressSpace.GM)
            pto.CmoInvalidateOp(pto.AddressSpace.GM)
            pto.FenceReleaseOp(pto.FenceScope.DDR)
            pto.FenceAcquireOp(pto.FenceScope.DDR)

        text = str(module)
        assert_contains(text, "pto.cmo.clean all <gm>")
        assert_contains(text, "pto.cmo.invalidate all <gm>")
        assert_contains(text, "pto.fence.release <ddr>")
        assert_contains(text, "pto.fence.acquire <ddr>")

    print("memory_consistency_bindings: PASS")


if __name__ == "__main__":
    main()
