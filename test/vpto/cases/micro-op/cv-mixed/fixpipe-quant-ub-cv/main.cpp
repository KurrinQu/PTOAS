// Copyright (c) 2026 Huawei Technologies Co., Ltd.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.

#include "acl/acl.h"
#include "test_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstdint>

using namespace PtoTestCommon;

#define ACL_CHECK(expr)                                                          \
  do {                                                                           \
    const aclError _ret = (expr);                                                \
    if (_ret != ACL_SUCCESS) {                                                   \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,             \
                   (int)_ret, __FILE__, __LINE__);                               \
      const char *_recent = aclGetRecentErrMsg();                                \
      if (_recent != nullptr && _recent[0] != '\0')                              \
        std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", _recent);             \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

#define FILE_CHECK(expr, path)                                                   \
  do {                                                                           \
    if (!(expr)) {                                                               \
      std::fprintf(stderr, "[ERROR] file operation failed: %s (%s:%d)\n",        \
                   path, __FILE__, __LINE__);                                    \
      rc = 1;                                                                    \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

void LaunchFixpipe_quant_ub_cv_kernel(__fp16 *src, __fp16 *id, uint32_t *fp,
                                      __fp16 *out, void *stream);

int main() {
  constexpr size_t kSizeSrcElems = 50 * 64;
  constexpr size_t kSizeIdElems = 40 * 50;
  constexpr size_t kFpElems = 128;
  constexpr size_t kOutElems = 40 * 64;
  constexpr size_t kSizeSrc = kSizeSrcElems * sizeof(__fp16);
  constexpr size_t kSizeId = kSizeIdElems * sizeof(__fp16);
  constexpr size_t kSizeFp = kFpElems * sizeof(uint32_t);
  constexpr size_t kSizeOut = kOutElems * sizeof(__fp16);

  __fp16 *srcHost = nullptr;
  __fp16 *idHost = nullptr;
  uint32_t *fpHost = nullptr;
  __fp16 *outHost = nullptr;
  __fp16 *srcDevice = nullptr;
  __fp16 *idDevice = nullptr;
  uint32_t *fpDevice = nullptr;
  __fp16 *outDevice = nullptr;

  int rc = 0;
  bool aclInited = false;
  bool deviceSet = false;
  int deviceId = 0;
  aclrtStream stream = nullptr;
  size_t inputSize = 0;

  ACL_CHECK(aclInit(nullptr));
  aclInited = true;
  if (const char *envDevice = std::getenv("ACL_DEVICE_ID"))
    deviceId = std::atoi(envDevice);
  ACL_CHECK(aclrtSetDevice(deviceId));
  deviceSet = true;
  ACL_CHECK(aclrtCreateStream(&stream));

  ACL_CHECK(aclrtMallocHost((void **)&srcHost, kSizeSrc));
  ACL_CHECK(aclrtMallocHost((void **)&idHost, kSizeId));
  ACL_CHECK(aclrtMallocHost((void **)&fpHost, kSizeFp));
  ACL_CHECK(aclrtMallocHost((void **)&outHost, kSizeOut));
  ACL_CHECK(aclrtMalloc((void **)&srcDevice, kSizeSrc, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&idDevice, kSizeId, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&fpDevice, kSizeFp, ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc((void **)&outDevice, kSizeOut, ACL_MEM_MALLOC_HUGE_FIRST));

  inputSize = kSizeId;
  FILE_CHECK(ReadFile("./v1.bin", inputSize, idHost, kSizeId) && inputSize == kSizeId,
             "./v1.bin");
  inputSize = kSizeSrc;
  FILE_CHECK(ReadFile("./v2.bin", inputSize, srcHost, kSizeSrc) && inputSize == kSizeSrc,
             "./v2.bin");
  inputSize = kSizeFp;
  FILE_CHECK(ReadFile("./v3.bin", inputSize, fpHost, kSizeFp) && inputSize == kSizeFp,
             "./v3.bin");
  inputSize = kSizeOut;
  FILE_CHECK(ReadFile("./v4.bin", inputSize, outHost, kSizeOut) &&
                 inputSize == kSizeOut,
             "./v4.bin");

  ACL_CHECK(aclrtMemcpy(srcDevice, kSizeSrc, srcHost, kSizeSrc, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(idDevice, kSizeId, idHost, kSizeId, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(fpDevice, kSizeFp, fpHost, kSizeFp, ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(outDevice, kSizeOut, outHost, kSizeOut,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchFixpipe_quant_ub_cv_kernel(srcDevice, idDevice, fpDevice, outDevice, stream);
  ACL_CHECK(aclrtSynchronizeStream(stream));

  ACL_CHECK(aclrtMemcpy(outHost, kSizeOut, outDevice, kSizeOut,
                        ACL_MEMCPY_DEVICE_TO_HOST));
  FILE_CHECK(WriteFile("./v4.bin", outHost, kSizeOut), "./v4.bin");

cleanup:
  aclrtFree(srcDevice);
  aclrtFree(idDevice);
  aclrtFree(fpDevice);
  aclrtFree(outDevice);
  aclrtFreeHost(srcHost);
  aclrtFreeHost(idHost);
  aclrtFreeHost(fpHost);
  aclrtFreeHost(outHost);
  if (stream != nullptr)
    aclrtDestroyStream(stream);
  if (deviceSet)
    aclrtResetDevice(deviceId);
  if (aclInited)
    aclFinalize();
  return rc;
}
