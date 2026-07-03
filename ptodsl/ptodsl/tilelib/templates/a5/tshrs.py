# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""PTODSL TileLib template for pto.tshrs."""

from ptodsl import pto

from ._elementwise import register_scalar_binary


template_tshrs = register_scalar_binary(
    op="pto.tshrs",
    name="template_tshrs",
    vector_op=pto.vshrs,
    dtypes=[
        ("i8", "i16", "i8"),
        ("i16", "i16", "i16"),
        ("i32", "i16", "i32"),
    ],
)
