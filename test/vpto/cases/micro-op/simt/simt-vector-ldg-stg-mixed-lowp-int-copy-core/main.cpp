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

void LaunchSimt_vector_ldg_stg_mixed_lowp_int_copy_core_kernel(
    void *hif8x2, void *i8x2, void *fp8e4x2, void *fp8e5x2, void *stream);

int main() {
  // Each buffer: 4 elements (2 input + 2 output), each element = 2 bytes → 8 bytes
  size_t elemCount = 4;
  size_t fileSize = elemCount * 2;

  void *hif8x2Host = nullptr;
  void *hif8x2Device = nullptr;
  void *i8x2Host = nullptr;
  void *i8x2Device = nullptr;
  void *fp8e4x2Host = nullptr;
  void *fp8e4x2Device = nullptr;
  void *fp8e5x2Host = nullptr;
  void *fp8e5x2Device = nullptr;
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

  // hif8x2
  ACL_CHECK(aclrtMallocHost(&hif8x2Host, fileSize));
  ACL_CHECK(aclrtMalloc((void **)&hif8x2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./hif8x2.bin", fileSize, hif8x2Host, fileSize);
  ACL_CHECK(aclrtMemcpy(hif8x2Device, fileSize, hif8x2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // i8x2
  ACL_CHECK(aclrtMallocHost(&i8x2Host, fileSize));
  ACL_CHECK(aclrtMalloc((void **)&i8x2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./i8x2.bin", fileSize, i8x2Host, fileSize);
  ACL_CHECK(aclrtMemcpy(i8x2Device, fileSize, i8x2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // fp8e4x2
  ACL_CHECK(aclrtMallocHost(&fp8e4x2Host, fileSize));
  ACL_CHECK(aclrtMalloc((void **)&fp8e4x2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./fp8e4x2.bin", fileSize, fp8e4x2Host, fileSize);
  ACL_CHECK(aclrtMemcpy(fp8e4x2Device, fileSize, fp8e4x2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  // fp8e5x2
  ACL_CHECK(aclrtMallocHost(&fp8e5x2Host, fileSize));
  ACL_CHECK(aclrtMalloc((void **)&fp8e5x2Device, fileSize, ACL_MEM_MALLOC_HUGE_FIRST));
  ReadFile("./fp8e5x2.bin", fileSize, fp8e5x2Host, fileSize);
  ACL_CHECK(aclrtMemcpy(fp8e5x2Device, fileSize, fp8e5x2Host, fileSize, ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchSimt_vector_ldg_stg_mixed_lowp_int_copy_core_kernel(
      hif8x2Device, i8x2Device, fp8e4x2Device, fp8e5x2Device, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(hif8x2Host, fileSize, hif8x2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./hif8x2.bin", hif8x2Host, fileSize);

  ACL_CHECK(aclrtMemcpy(i8x2Host, fileSize, i8x2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./i8x2.bin", i8x2Host, fileSize);

  ACL_CHECK(aclrtMemcpy(fp8e4x2Host, fileSize, fp8e4x2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./fp8e4x2.bin", fp8e4x2Host, fileSize);

  ACL_CHECK(aclrtMemcpy(fp8e5x2Host, fileSize, fp8e5x2Device, fileSize, ACL_MEMCPY_DEVICE_TO_HOST));
  WriteFile("./fp8e5x2.bin", fp8e5x2Host, fileSize);

cleanup:
  aclrtFree(hif8x2Device);   aclrtFreeHost(hif8x2Host);
  aclrtFree(i8x2Device);     aclrtFreeHost(i8x2Host);
  aclrtFree(fp8e4x2Device);  aclrtFreeHost(fp8e4x2Host);
  aclrtFree(fp8e5x2Device);  aclrtFreeHost(fp8e5x2Host);
  if (stream)   aclrtDestroyStream(stream);
  if (deviceSet) aclrtResetDevice(deviceId);
  if (aclInited) aclFinalize();
  return rc;
}
