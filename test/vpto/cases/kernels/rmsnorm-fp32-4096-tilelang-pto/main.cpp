#include "acl/acl.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <limits>
#include <string>
#include <vector>

namespace {

constexpr size_t kBatch = 4096;
constexpr size_t kD = 4096;
constexpr size_t kXElems = kBatch * kD;
constexpr size_t kYElems = kXElems;
constexpr size_t kWElems = kD;
constexpr size_t kRstdElems = kBatch;
constexpr float kEps = 1.0e-6f;

using CallKernel = int (*)(void*, void*, void*, void*, float, void*);

bool CheckAcl(aclError ret, const char* expr, const char* file, int line) {
  if (ret == ACL_SUCCESS) {
    return true;
  }
  std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", expr, static_cast<int>(ret), file,
               line);
  const char* recent = aclGetRecentErrMsg();
  if (recent != nullptr && recent[0] != '\0') {
    std::fprintf(stderr, "[ERROR] RecentErrMsg: %s\n", recent);
  }
  return false;
}

#define ACL_CHECK(expr) CheckAcl((expr), #expr, __FILE__, __LINE__)

void FillInputs(std::vector<float>& x, std::vector<float>& w) {
  for (size_t row = 0; row < kBatch; ++row) {
    const size_t base = row * kD;
    for (size_t i = 0; i < kD; ++i) {
      const int centered = static_cast<int>((row * 13 + i) % 31) - 15;
      x[base + i] = 0.25f + static_cast<float>(centered) * 0.0078125f +
                    static_cast<float>(row % 7) * 0.015625f;
    }
  }

  for (size_t i = 0; i < kWElems; ++i) {
    w[i] = 0.75f + static_cast<float>(i % 17) * 0.03125f;
  }
}

float ReferenceRstd(const std::vector<float>& x, size_t row) {
  const size_t base = row * kD;
  float sum_sq = 0.0f;
  for (size_t i = 0; i < kD; ++i) {
    sum_sq += x[base + i] * x[base + i];
  }
  return 1.0f / std::sqrt(sum_sq / static_cast<float>(kD) + kEps);
}

bool Near(float actual, float expected, float atol, float rtol) {
  const float diff = std::fabs(actual - expected);
  return diff <= atol + rtol * std::fabs(expected);
}

int GetEnvInt(const char* name, int fallback) {
  const char* value = std::getenv(name);
  if (value == nullptr || value[0] == '\0') {
    return fallback;
  }
  return std::atoi(value);
}

void* DevicePtrAdd(void* ptr, size_t byte_offset) {
  return static_cast<void*>(static_cast<unsigned char*>(ptr) + byte_offset);
}

}  // namespace

int main(int argc, char** argv) {
  const char* so_path = (argc > 1) ? argv[1] : "./lib_kernel.so";
  const int device_id = GetEnvInt("ACL_DEVICE_ID", 0);
  const int warmup_iters = std::max(0, GetEnvInt("WARMUP_ITERS", 0));
  const int run_iters = std::max(1, GetEnvInt("RUN_ITERS", 1));
  const bool skip_validation = GetEnvInt("SKIP_VALIDATION", 0) != 0;

  const size_t x_bytes = kXElems * sizeof(float);
  const size_t y_bytes = kYElems * sizeof(float);
  const size_t w_bytes = kWElems * sizeof(float);
  const size_t rstd_bytes = kRstdElems * sizeof(float);

  std::vector<float> x_host(kXElems);
  std::vector<float> w_host(kWElems);
  std::vector<float> y_host(kYElems, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> y_init(kYElems, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> rstd_host(kRstdElems, std::numeric_limits<float>::quiet_NaN());
  std::vector<float> rstd_init(kRstdElems, std::numeric_limits<float>::quiet_NaN());

  void* x_device = nullptr;
  void* y_device = nullptr;
  void* w_device = nullptr;
  void* rstd_device = nullptr;
  aclrtStream stream = nullptr;
  void* handle = nullptr;
  CallKernel call_kernel = nullptr;
  bool acl_inited = false;
  bool device_set = false;
  int rc = 0;

  FillInputs(x_host, w_host);

  handle = dlopen(so_path, RTLD_NOW | RTLD_LOCAL);
  if (handle == nullptr) {
    std::fprintf(stderr, "[ERROR] dlopen(%s) failed: %s\n", so_path, dlerror());
    return 1;
  }
  call_kernel = reinterpret_cast<CallKernel>(dlsym(handle, "call"));
  if (call_kernel == nullptr) {
    std::fprintf(stderr, "[ERROR] dlsym(call) failed: %s\n", dlerror());
    rc = 1;
    goto cleanup;
  }

  if (!ACL_CHECK(aclInit(nullptr))) {
    rc = 1;
    goto cleanup;
  }
  acl_inited = true;
  if (!ACL_CHECK(aclrtSetDevice(device_id))) {
    rc = 1;
    goto cleanup;
  }
  device_set = true;
  if (!ACL_CHECK(aclrtCreateStream(&stream))) {
    rc = 1;
    goto cleanup;
  }

  if (!ACL_CHECK(aclrtMalloc(&x_device, x_bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
      !ACL_CHECK(aclrtMalloc(&y_device, y_bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
      !ACL_CHECK(aclrtMalloc(&w_device, w_bytes, ACL_MEM_MALLOC_HUGE_FIRST)) ||
      !ACL_CHECK(aclrtMalloc(&rstd_device, rstd_bytes, ACL_MEM_MALLOC_HUGE_FIRST))) {
    rc = 1;
    goto cleanup;
  }

  if (!ACL_CHECK(aclrtMemcpy(x_device, x_bytes, x_host.data(), x_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE)) ||
      !ACL_CHECK(aclrtMemcpy(y_device, y_bytes, y_init.data(), y_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE)) ||
      !ACL_CHECK(aclrtMemcpy(w_device, w_bytes, w_host.data(), w_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE)) ||
      !ACL_CHECK(aclrtMemcpy(rstd_device, rstd_bytes, rstd_init.data(), rstd_bytes,
                             ACL_MEMCPY_HOST_TO_DEVICE))) {
    rc = 1;
    goto cleanup;
  }

  std::printf("launch %s: batch=%zu d=%zu grid=64 dyn_ub=82496 warmup=%d run=%d\n", so_path,
              kBatch, kD, warmup_iters, run_iters);

  for (int i = 0; i < warmup_iters; ++i) {
    if (call_kernel(x_device, y_device, w_device, rstd_device, kEps, stream) != 0) {
      std::fprintf(stderr, "[ERROR] warmup call returned non-zero\n");
      rc = 1;
      goto cleanup;
    }
  }
  if (warmup_iters > 0 && !ACL_CHECK(aclrtSynchronizeStream(stream))) {
    rc = 1;
    goto cleanup;
  }

  {
    const auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < run_iters; ++i) {
      if (call_kernel(x_device, y_device, w_device, rstd_device, kEps, stream) != 0) {
        std::fprintf(stderr, "[ERROR] call returned non-zero\n");
        rc = 1;
        goto cleanup;
      }
    }
    if (!ACL_CHECK(aclrtSynchronizeStream(stream))) {
      rc = 1;
      goto cleanup;
    }
    const auto end = std::chrono::steady_clock::now();
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(end - begin).count();
    std::printf("elapsed: %.6f ms total, %.6f ms/iter\n", elapsed_ms,
                elapsed_ms / static_cast<double>(run_iters));
  }

  if (skip_validation) {
    std::printf("validation skipped by SKIP_VALIDATION=1\n");
    goto cleanup;
  }

  if (!ACL_CHECK(aclrtMemcpy(rstd_host.data(), rstd_bytes, rstd_device, rstd_bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST)) ||
      !ACL_CHECK(aclrtMemcpy(y_host.data(), y_bytes, y_device, y_bytes,
                             ACL_MEMCPY_DEVICE_TO_HOST))) {
    rc = 1;
    goto cleanup;
  }

  {
    size_t rstd_mismatch = 0;
    size_t y_mismatch = 0;
    float max_rstd_abs = 0.0f;
    float max_rstd_rel = 0.0f;
    float max_y_abs = 0.0f;
    float max_y_rel = 0.0f;
    bool printed_rstd = false;
    bool printed_y = false;

    for (size_t row = 0; row < kBatch; ++row) {
      const float rstd_ref = ReferenceRstd(x_host, row);
      const float rstd_abs = std::fabs(rstd_host[row] - rstd_ref);
      const float rstd_rel = rstd_abs / std::max(std::fabs(rstd_ref), 1.0e-20f);
      max_rstd_abs = std::max(max_rstd_abs, rstd_abs);
      max_rstd_rel = std::max(max_rstd_rel, rstd_rel);
      if (!Near(rstd_host[row], rstd_ref, 3.0e-5f, 3.0e-5f)) {
        ++rstd_mismatch;
        if (!printed_rstd) {
          printed_rstd = true;
          std::fprintf(stderr,
                       "[VALIDATION] first RSTD mismatch: row=%zu actual=%.9g "
                       "expected=%.9g abs=%.9g rel=%.9g\n",
                       row, rstd_host[row], rstd_ref, rstd_abs, rstd_rel);
        }
      }

      const size_t base = row * kD;
      for (size_t i = 0; i < kD; ++i) {
        const float y_ref = x_host[base + i] * rstd_ref * w_host[i];
        const float y_abs = std::fabs(y_host[base + i] - y_ref);
        const float y_rel = y_abs / std::max(std::fabs(y_ref), 1.0e-20f);
        max_y_abs = std::max(max_y_abs, y_abs);
        max_y_rel = std::max(max_y_rel, y_rel);
        if (!Near(y_host[base + i], y_ref, 4.0e-5f, 4.0e-5f)) {
          ++y_mismatch;
          if (!printed_y) {
            printed_y = true;
            std::fprintf(stderr,
                         "[VALIDATION] first Y mismatch: row=%zu col=%zu actual=%.9g "
                         "expected=%.9g abs=%.9g rel=%.9g\n",
                         row, i, y_host[base + i], y_ref, y_abs, y_rel);
          }
        }
      }
    }

    if (rstd_mismatch == 0 && y_mismatch == 0) {
      std::printf("validation PASS: checked %zu rows, %zu Y values, "
                  "max_rstd_abs=%.9g max_rstd_rel=%.9g max_y_abs=%.9g "
                  "max_y_rel=%.9g\n",
                  kBatch, kYElems, max_rstd_abs, max_rstd_rel, max_y_abs, max_y_rel);
    } else {
      std::fprintf(stderr,
                   "validation FAIL: rstd_mismatch=%zu/%zu y_mismatch=%zu/%zu "
                   "max_rstd_abs=%.9g max_rstd_rel=%.9g max_y_abs=%.9g "
                   "max_y_rel=%.9g\n",
                   rstd_mismatch, kBatch, y_mismatch, kYElems, max_rstd_abs, max_rstd_rel,
                   max_y_abs, max_y_rel);
      rc = 1;
    }
  }

cleanup:
  if (x_device != nullptr) {
    aclrtFree(x_device);
  }
  if (y_device != nullptr) {
    aclrtFree(y_device);
  }
  if (w_device != nullptr) {
    aclrtFree(w_device);
  }
  if (rstd_device != nullptr) {
    aclrtFree(rstd_device);
  }
  if (stream != nullptr) {
    aclrtDestroyStream(stream);
  }
  if (device_set) {
    aclrtResetDevice(device_id);
  }
  if (acl_inited) {
    aclFinalize();
  }
  if (handle != nullptr) {
    dlclose(handle);
  }
  return rc;
}
