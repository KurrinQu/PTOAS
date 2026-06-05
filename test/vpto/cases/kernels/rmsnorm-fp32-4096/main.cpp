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

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                                          \
  do {                                                                                           \
    const aclError _ret = (expr);                                                                \
    if (_ret != ACL_SUCCESS) {                                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, (int)_ret, __FILE__,        \
                   __LINE__);                                                                    \
      const char *_recent = aclGetRecentErrMsg();                                                \
      if (_recent != nullptr && _recent[0] != '\0')                                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);                            \
      rc = 1;                                                                                    \
      goto cleanup;                                                                              \
    }                                                                                            \
  } while (0)

void LaunchRmsnorm_fp32_4096(float *x, float *y, float *w, float *rstd,
                             float eps, void *stream);

int main() {
  constexpr size_t batch = 4096;
  constexpr size_t hidden = 4096;
  constexpr size_t elemCountX = batch * hidden;
  constexpr size_t elemCountW = hidden;
  constexpr size_t elemCountRstd = batch;
  constexpr size_t expectedFileSizeX = elemCountX * sizeof(float);
  constexpr size_t expectedFileSizeW = elemCountW * sizeof(float);
  constexpr size_t expectedFileSizeRstd = elemCountRstd * sizeof(float);
  constexpr float eps = 1.0e-6f;

  float *xHost = nullptr;
  float *yHost = nullptr;
  float *wHost = nullptr;
  float *rstdHost = nullptr;
  float *xDevice = nullptr;
  float *yDevice = nullptr;
  float *wDevice = nullptr;
  float *rstdDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t fileSizeX = expectedFileSizeX;
  size_t fileSizeY = expectedFileSizeX;
  size_t fileSizeW = expectedFileSizeW;
  size_t fileSizeRstd = expectedFileSizeRstd;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)(&xHost), expectedFileSizeX));
  ACL_CHECK(aclrtMallocHost((void **)(&yHost), expectedFileSizeX));
  ACL_CHECK(aclrtMallocHost((void **)(&wHost), expectedFileSizeW));
  ACL_CHECK(aclrtMallocHost((void **)(&rstdHost), expectedFileSizeRstd));

  ACL_CHECK(aclrtMalloc((void **)&xDevice, expectedFileSizeX, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&yDevice, expectedFileSizeX, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&wDevice, expectedFileSizeW, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&rstdDevice, expectedFileSizeRstd, ACL_MEM_MALLOC_HUGE_FIRST));

  ReadFile("./v1.bin", fileSizeX, xHost, fileSizeX);
  ReadFile("./v2.bin", fileSizeY, yHost, fileSizeY);
  ReadFile("./v3.bin", fileSizeW, wHost, fileSizeW);
  ReadFile("./v4.bin", fileSizeRstd, rstdHost, fileSizeRstd);

  ACL_CHECK(aclrtMemcpy(xDevice, expectedFileSizeX, xHost, expectedFileSizeX, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(yDevice, expectedFileSizeX, yHost, expectedFileSizeX, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(wDevice, expectedFileSizeW, wHost, expectedFileSizeW, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(rstdDevice, expectedFileSizeRstd, rstdHost, expectedFileSizeRstd, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchRmsnorm_fp32_4096(xDevice, yDevice, wDevice, rstdDevice, eps, stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(yHost, expectedFileSizeX, yDevice, expectedFileSizeX, ACL_MEMCPY_DEVICE_TO_HOST));
  ACL_CHECK(aclrtMemcpy(rstdHost, expectedFileSizeRstd, rstdDevice, expectedFileSizeRstd,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./v2.bin", yHost, expectedFileSizeX);
  WriteFile("./v4.bin", rstdHost, expectedFileSizeRstd);

cleanup:
  aclrtFree(xDevice);
  aclrtFree(yDevice);
  aclrtFree(wDevice);
  aclrtFree(rstdDevice);
  aclrtFreeHost(xHost);
  aclrtFreeHost(yHost);
  aclrtFreeHost(wHost);
  aclrtFreeHost(rstdHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
