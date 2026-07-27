#include "acl/acl.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <cstring>
#include <vector>

namespace {
using CallKernel = int (*)(void*, void*, void*, void*, float, void*);

struct ShapeBuffers {
  size_t d;
  size_t elems;
  std::vector<float> x;
  std::vector<float> y;
  std::vector<float> w;
  std::vector<float> rstd;
  void* xd = nullptr;
  void* yd = nullptr;
  void* wd = nullptr;
  void* rd = nullptr;
};

void Fill(ShapeBuffers& b) {
  for (size_t row = 0; row < 4096; ++row) {
    for (size_t i = 0; i < b.d; ++i) {
      const int centered = static_cast<int>(i % 31) - 15;
      b.x[row * b.d + i] = 0.25f + centered * 0.0078125f + (row % 7) * 0.015625f;
    }
  }
  for (size_t i = 0; i < b.d; ++i) b.w[i] = 0.75f + (i % 17) * 0.03125f;
}

float Rstd(const ShapeBuffers& b, size_t row) {
  float sum = 0.0f;
  for (size_t i = 0; i < b.d; ++i) {
    const float v = b.x[row * b.d + i];
    sum += v * v;
  }
  return 1.0f / std::sqrt(sum / static_cast<float>(b.d) + 1.0e-6f);
}

}  // namespace

#define ACL_CHECK(expr)                                                                    \
  do {                                                                                     \
    const aclError _ret = (expr);                                                          \
    if (_ret != ACL_SUCCESS) {                                                             \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr, static_cast<int>(_ret), \
                   __FILE__, __LINE__);                                                    \
      const char* recent = aclGetRecentErrMsg();                                           \
      if (recent != nullptr && recent[0] != '\0') std::fprintf(stderr, "%s\n", recent);    \
      rc = 1;                                                                              \
      goto cleanup;                                                                        \
    }                                                                                      \
  } while (0)

int main(int argc, char** argv) {
  // Keep launch markers ordered with asynchronous ACL error output.
  std::setbuf(stdout, nullptr);
  if (argc < 4) {
    std::fprintf(stderr, "usage: %s kernel4096.so kernel5120.so kernel7168.so\n", argv[0]);
    return 2;
  }
  ShapeBuffers b4096{4096, 4096 * 4096, std::vector<float>(4096 * 4096),
                     std::vector<float>(4096 * 4096), std::vector<float>(4096),
                     std::vector<float>(4096), nullptr, nullptr, nullptr, nullptr};
  ShapeBuffers b5120{5120, 4096 * 5120, std::vector<float>(4096 * 5120),
                     std::vector<float>(4096 * 5120), std::vector<float>(5120),
                     std::vector<float>(4096), nullptr, nullptr, nullptr, nullptr};
  ShapeBuffers b7168{7168, 4096 * 7168, std::vector<float>(4096 * 7168),
                     std::vector<float>(4096 * 7168), std::vector<float>(7168),
                     std::vector<float>(4096), nullptr, nullptr, nullptr, nullptr};
  Fill(b4096);
  Fill(b5120);
  Fill(b7168);

  void* h0 = nullptr;
  void* h1 = nullptr;
  void* h2 = nullptr;
  CallKernel k0 = nullptr;
  CallKernel k1 = nullptr;
  CallKernel k2 = nullptr;
  aclrtStream stream = nullptr;
  bool inited = false;
  bool device_set = false;
  int device_id = 0;
  int rc = 0;
  if (const char* env = std::getenv("ACL_DEVICE_ID")) device_id = std::atoi(env);
  int repeats = 2;
  if (const char* env = std::getenv("RMSNORM_SEQUENCE_REPEATS")) {
    repeats = std::atoi(env);
    if (repeats < 1) repeats = 1;
  }
  const char* only = std::getenv("RMSNORM_ONLY");

  h0 = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  h1 = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  h2 = dlopen(argv[3], RTLD_NOW | RTLD_LOCAL);
  if (h0 == nullptr || h1 == nullptr || h2 == nullptr) {
    std::fprintf(stderr, "[ERROR] dlopen failed: %s\n", dlerror());
    return 1;
  }
  k0 = reinterpret_cast<CallKernel>(dlsym(h0, "call"));
  k1 = reinterpret_cast<CallKernel>(dlsym(h1, "call"));
  k2 = reinterpret_cast<CallKernel>(dlsym(h2, "call"));
  if (k0 == nullptr || k1 == nullptr || k2 == nullptr) {
    std::fprintf(stderr, "[ERROR] dlsym(call) failed\n");
    return 1;
  }

  ACL_CHECK(aclInit(nullptr));
  inited = true;
  ACL_CHECK(aclrtSetDevice(device_id));
  device_set = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  for (ShapeBuffers* b : {&b4096, &b5120, &b7168}) {
    ACL_CHECK(aclrtMalloc(&b->xd, b->elems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b->yd, b->elems * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b->wd, b->d * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMalloc(&b->rd, 4096 * sizeof(float), ACL_MEM_MALLOC_HUGE_FIRST));
    ACL_CHECK(aclrtMemcpy(b->xd, b->elems * sizeof(float), b->x.data(),
                         b->elems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(b->yd, b->elems * sizeof(float), b->y.data(),
                         b->elems * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(b->wd, b->d * sizeof(float), b->w.data(), b->d * sizeof(float),
                         ACL_MEMCPY_HOST_TO_DEVICE));
    ACL_CHECK(aclrtMemcpy(b->rd, 4096 * sizeof(float), b->rstd.data(),
                         4096 * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE));
  }
  std::printf("allocated x pointers: 0x%llx 0x%llx 0x%llx\n",
              (unsigned long long)b4096.xd, (unsigned long long)b5120.xd,
              (unsigned long long)b7168.xd);

  if (only == nullptr || std::strcmp(only, "4096") == 0) {
    for (int repeat = 0; repeat < repeats; ++repeat) {
      std::printf("launch 4096 repeat=%d\n", repeat);
      if (k0(b4096.xd, b4096.yd, b4096.wd, b4096.rd, 1.0e-6f, stream) != 0) { rc = 1; goto cleanup; }
      ACL_CHECK(aclrtSynchronizeStream(stream));
    }
  }
  if (only == nullptr || std::strcmp(only, "5120") == 0) {
    for (int repeat = 0; repeat < repeats; ++repeat) {
      std::printf("launch 5120 repeat=%d\n", repeat);
      if (k1(b5120.xd, b5120.yd, b5120.wd, b5120.rd, 1.0e-6f, stream) != 0) { rc = 1; goto cleanup; }
      ACL_CHECK(aclrtSynchronizeStream(stream));
    }
  }
  if (only == nullptr || std::strcmp(only, "7168") == 0) {
    std::printf("launch 7168\n");
    if (k2(b7168.xd, b7168.yd, b7168.wd, b7168.rd, 1.0e-6f, stream) != 0) { rc = 1; goto cleanup; }
    ACL_CHECK(aclrtSynchronizeStream(stream));
    ACL_CHECK(aclrtMemcpy(b7168.rstd.data(), 4096 * sizeof(float), b7168.rd,
                         4096 * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST));
    for (size_t row : {size_t(0), size_t(1), size_t(63), size_t(4095)}) {
      const float expected = Rstd(b7168, row);
      std::printf("7168 row=%zu rstd=%.9g expected=%.9g\n", row, b7168.rstd[row], expected);
      if (std::fabs(b7168.rstd[row] - expected) > 2.0e-5f) rc = 1;
    }
  }

cleanup:
  for (ShapeBuffers* b : {&b4096, &b5120, &b7168}) {
    if (b->xd) aclrtFree(b->xd);
    if (b->yd) aclrtFree(b->yd);
    if (b->wd) aclrtFree(b->wd);
    if (b->rd) aclrtFree(b->rd);
  }
  if (stream) aclrtDestroyStream(stream);
  if (device_set) aclrtResetDevice(device_id);
  if (inited) aclFinalize();
  if (h0) dlclose(h0);
  if (h1) dlclose(h1);
  if (h2) dlclose(h2);
  return rc;
}
