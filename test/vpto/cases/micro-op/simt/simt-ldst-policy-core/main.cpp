// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
#include "test_common.h"
#include "acl/acl.h"
#include <cstdio>
#include <cstdlib>
#include <stdint.h>

using namespace PtoTestCommon;

#define ACL_CHECK(expr) do { const aclError _ret = (expr); if (_ret != ACL_SUCCESS) { std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__, __LINE__); rc = 1; goto cleanup; } } while (0)

void LaunchSimt_ldst_policy_core_kernel(int *v1, uint16_t *v2, uint16_t *v3,
                                        int8_t *v4, int16_t *v5, int64_t *v6,
                                        float *v7, double *v8,
                                        void *stream);

int main() {
  size_t elemCount_v1 = 1024;
  size_t fileSize_v1 = elemCount_v1 * sizeof(int);
  size_t elemCount_v2 = 1024;
  size_t fileSize_v2 = elemCount_v2 * sizeof(uint16_t);
  size_t fileSize_v4 = elemCount_v2 * sizeof(int8_t);
  size_t fileSize_v5 = elemCount_v2 * sizeof(int16_t);
  size_t fileSize_v6 = elemCount_v2 * sizeof(int64_t);
  size_t fileSize_v7 = elemCount_v2 * sizeof(float);
  size_t fileSize_v8 = elemCount_v2 * sizeof(double);
  int *v1Host = nullptr;
  uint16_t *v2Host = nullptr;
  uint16_t *v3Host = nullptr;
  int8_t *v4Host = nullptr;
  int16_t *v5Host = nullptr;
  int64_t *v6Host = nullptr;
  float *v7Host = nullptr;
  double *v8Host = nullptr;
  int *v1Device = nullptr;
  uint16_t *v2Device = nullptr;
  uint16_t *v3Device = nullptr;
  int8_t *v4Device = nullptr;
  int16_t *v5Device = nullptr;
  int64_t *v6Device = nullptr;
  float *v7Device = nullptr;
  double *v8Device = nullptr;
  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMallocHost((void **)(&v1Host), fileSize_v1));
  ACL_CHECK(aclrtMallocHost((void **)(&v2Host), fileSize_v2));
  ACL_CHECK(aclrtMallocHost((void **)(&v3Host), fileSize_v2));
  ACL_CHECK(aclrtMallocHost((void **)(&v4Host), fileSize_v4));
  ACL_CHECK(aclrtMallocHost((void **)(&v5Host), fileSize_v5));
  ACL_CHECK(aclrtMallocHost((void **)(&v6Host), fileSize_v6));
  ACL_CHECK(aclrtMallocHost((void **)(&v7Host), fileSize_v7));
  ACL_CHECK(aclrtMallocHost((void **)(&v8Host), fileSize_v8));
  ACL_CHECK(aclrtMalloc((void **)&v1Device, fileSize_v1, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v2Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v3Device, fileSize_v2, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v4Device, fileSize_v4, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v5Device, fileSize_v5, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v6Device, fileSize_v6, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v7Device, fileSize_v7, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&v8Device, fileSize_v8, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./v1.bin", fileSize_v1, v1Host, fileSize_v1);
  ReadFile("./v2.bin", fileSize_v2, v2Host, fileSize_v2);
  ReadFile("./v3.bin", fileSize_v2, v3Host, fileSize_v2);
  ReadFile("./v4.bin", fileSize_v4, v4Host, fileSize_v4);
  ReadFile("./v5.bin", fileSize_v5, v5Host, fileSize_v5);
  ReadFile("./v6.bin", fileSize_v6, v6Host, fileSize_v6);
  ReadFile("./v7.bin", fileSize_v7, v7Host, fileSize_v7);
  ReadFile("./v8.bin", fileSize_v8, v8Host, fileSize_v8);
  ACL_CHECK(aclrtMemcpy(v1Device, fileSize_v1, v1Host, fileSize_v1, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v2Device, fileSize_v2, v2Host, fileSize_v2, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v3Device, fileSize_v2, v3Host, fileSize_v2, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v4Device, fileSize_v4, v4Host, fileSize_v4, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v5Device, fileSize_v5, v5Host, fileSize_v5, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v6Device, fileSize_v6, v6Host, fileSize_v6, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v7Device, fileSize_v7, v7Host, fileSize_v7, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(v8Device, fileSize_v8, v8Host, fileSize_v8, ACL_MEMCPY_HOST_TO_DEVICE));
  LaunchSimt_ldst_policy_core_kernel(v1Device, v2Device, v3Device, v4Device,
                                     v5Device, v6Device, v7Device, v8Device,
                                     stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(v1Host, fileSize_v1, v1Device, fileSize_v1, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v2Host, fileSize_v2, v2Device, fileSize_v2, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v3Host, fileSize_v2, v3Device, fileSize_v2, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v4Host, fileSize_v4, v4Device, fileSize_v4, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v5Host, fileSize_v5, v5Device, fileSize_v5, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v6Host, fileSize_v6, v6Device, fileSize_v6, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v7Host, fileSize_v7, v7Device, fileSize_v7, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(v8Host, fileSize_v8, v8Device, fileSize_v8, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v1.bin", v1Host, fileSize_v1);
  WriteFile("./v2.bin", v2Host, fileSize_v2);
  WriteFile("./v3.bin", v3Host, fileSize_v2);
  WriteFile("./v4.bin", v4Host, fileSize_v4);
  WriteFile("./v5.bin", v5Host, fileSize_v5);
  WriteFile("./v6.bin", v6Host, fileSize_v6);
  WriteFile("./v7.bin", v7Host, fileSize_v7);
  WriteFile("./v8.bin", v8Host, fileSize_v8);

cleanup:
  aclrtFree(v8Device);
  aclrtFree(v7Device);
  aclrtFree(v6Device);
  aclrtFree(v5Device);
  aclrtFree(v4Device);
  aclrtFree(v3Device);
  aclrtFree(v2Device);
  aclrtFree(v1Device);
  aclrtFreeHost(v8Host);
  aclrtFreeHost(v7Host);
  aclrtFreeHost(v6Host);
  aclrtFreeHost(v5Host);
  aclrtFreeHost(v4Host);
  aclrtFreeHost(v3Host);
  aclrtFreeHost(v2Host);
  aclrtFreeHost(v1Host);
  if (stream)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
