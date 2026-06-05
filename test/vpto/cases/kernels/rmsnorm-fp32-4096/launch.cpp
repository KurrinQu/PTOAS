// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#include <stdint.h>

#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

extern "C" __global__ [aicore] void main_kernel(
    __gm__ float *x, __gm__ float *y, __gm__ float *w, __gm__ float *rstd,
    float eps);

void LaunchRmsnorm_fp32_4096(float *x, float *y, float *w, float *rstd,
                             float eps, void *stream) {
  constexpr int32_t blocks = 64;
  constexpr int32_t dynSharedBytes = 82496;
  main_kernel<<<blocks, dynSharedBytes, stream>>>((__gm__ float *)x,
                                                  (__gm__ float *)y,
                                                  (__gm__ float *)w,
                                                  (__gm__ float *)rstd, eps);
}
