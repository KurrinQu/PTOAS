# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
"""
``pto`` – the public DSL namespace.

Import as::

    import pto

or as the sub-namespace ``pto`` from the ptodsl package::

    from ptodsl import pto

All user-facing symbols live here.  Low-level MLIR bindings are accessed
internally as ``_pto`` (``from mlir.dialects import pto as _pto``).
"""

# ── Types ─────────────────────────────────────────────────────────────────────
from ._types import (           # noqa: F401
    float32, float16, bf16,
    int1, int8, int16, int32, int64,
    index,
    ptr, vreg_type, mask_type,
    tile_buf_type, tensor_view_type, part_tensor_view_type,
    _resolve,
)
from ._surface_types import (   # noqa: F401
    constexpr,
    tensor_spec,
    TensorSpec,
    BarrierType,
    Pipe,
    MemorySpace,
    TensorView,
    PartitionTensorView,
    Tile,
)
from ._tensor_factories import empty_like  # noqa: F401

# ── Operations ────────────────────────────────────────────────────────────────
from ._ops import (             # noqa: F401
    const,
    castptr, addptr,
    vlds, vbrc_load, vsts, vsts_1pt,
    plt_b32, pset_b32,
    make_mask,
    vadd, vmul, vmax, vdiv,
    vcmax, vcadd, vdup, vexpdif,
    vexp, vcgmax, vcgadd, vsubs,
    make_tensor_view, partition_view,
    alloc_tile, tload, tstore, tmov, as_ptr,
    mte_load, mte_store, mem_bar,
    mte_l1_l0a, mte_l1_l0b, mte_l0c_ub, mad,
    get_block_idx, get_block_num, get_subblock_idx, get_subblock_num,
    store_vfsimt_info, get_tid_x, get_tid_y, get_tid_z,
    pipe_barrier,
    set_flag, wait_flag,
)

# ── Control flow ──────────────────────────────────────────────────────────────
from ._control_flow import (    # noqa: F401
    vecscope,
    for_, if_, yield_,
    LoopHandle, BranchHandle,
)

# ── Decorator ─────────────────────────────────────────────────────────────────
from ._jit import jit, KernelHandle      # noqa: F401
from ._subkernels import ukernel, cube, simd, simt     # noqa: F401

# ── Scalar sub-namespace ──────────────────────────────────────────────────────
from . import scalar            # noqa: F401

# ── Shorthand dtype aliases ───────────────────────────────────────────────────
f32 = float32
f16 = float16
i1 = int1
i8 = int8
i16 = int16
i32 = int32
i64 = int64
