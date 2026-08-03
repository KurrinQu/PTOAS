#include "acl/acl.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <vector>

namespace {
using CallKernel = int (*)(int*, void*);
constexpr int kThreads = 128;
constexpr int kBufElems = 512;  // k1 额外写 out[256+tid]
constexpr int kSentinel = static_cast<int>(0xdeadbeef);

int g_rc = 0;
}  // namespace

#define ACL_CHECK(expr)                                                                    \
  do {                                                                                     \
    const aclError _ret = (expr);                                                          \
    if (_ret != ACL_SUCCESS) {                                                             \
      std::fprintf(stderr, "[ERROR] %s failed: %d (%s:%d)\n", #expr,                       \
                   static_cast<int>(_ret), __FILE__, __LINE__);                            \
      const char* recent = aclGetRecentErrMsg();                                           \
      if (recent != nullptr && recent[0] != '\0') std::fprintf(stderr, "%s\n", recent);    \
      g_rc = 1;                                                                            \
      goto cleanup;                                                                        \
    }                                                                                      \
  } while (0)

// 对 kernel 的一次 launch + 同步 + 读回校验。允许同步失败（复现场景预期
// kernel2 在 kernel1 之后失败），失败时打印错误并返回 false，不中断进程。
bool LaunchAndCheck(CallKernel fn, const char* name, int* dev_out, aclrtStream stream,
                    std::vector<int>& host_out) {
  std::printf("launch %s\n", name);
  if (fn(dev_out, stream) != 0) {
    std::fprintf(stderr, "[ERROR] %s host launch wrapper failed\n", name);
    return false;
  }
  const aclError sync_ret = aclrtSynchronizeStream(stream);
  if (sync_ret != ACL_SUCCESS) {
    const char* recent = aclGetRecentErrMsg();
    std::fprintf(stderr, "[FAULT] %s aclrtSynchronizeStream -> %d\n%s\n", name,
                 static_cast<int>(sync_ret), recent != nullptr ? recent : "");
    return false;
  }
  const aclError cp_ret = aclrtMemcpy(host_out.data(), kThreads * sizeof(int), dev_out,
                                      kThreads * sizeof(int), ACL_MEMCPY_DEVICE_TO_HOST);
  if (cp_ret != ACL_SUCCESS) {
    std::fprintf(stderr, "[ERROR] %s D2H failed: %d\n", name, static_cast<int>(cp_ret));
    return false;
  }
  int good = 0;
  for (int i = 0; i < kThreads; ++i) {
    if (host_out[i] == i) ++good;
  }
  std::printf("%s output: tid[0]=%d tid[1]=%d tid[127]=%d  (%d/%d lanes correct)\n", name,
              host_out[0], host_out[1], host_out[kThreads - 1], good, kThreads);
  return good == kThreads;
}

int main(int argc, char** argv) {
  std::setbuf(stdout, nullptr);
  if (argc < 3) {
    std::fprintf(stderr, "usage: %s k1_inlineasm.so k2_plain.so\n", argv[0]);
    return 2;
  }

  const char* only = std::getenv("TID_REPRO_ONLY");        // k1 | k2 | null
  const char* reverse = std::getenv("TID_REPRO_REVERSE");  // 非空则 k2 -> k1

  void* h1 = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
  void* h2 = dlopen(argv[2], RTLD_NOW | RTLD_LOCAL);
  if (h1 == nullptr || h2 == nullptr) {
    std::fprintf(stderr, "[ERROR] dlopen failed: %s\n", dlerror());
    return 1;
  }
  CallKernel k1 = reinterpret_cast<CallKernel>(dlsym(h1, "call"));
  CallKernel k2 = reinterpret_cast<CallKernel>(dlsym(h2, "call"));
  if (k1 == nullptr || k2 == nullptr) {
    std::fprintf(stderr, "[ERROR] dlsym(call) failed\n");
    return 1;
  }

  int device_id = 0;
  if (const char* env = std::getenv("ACL_DEVICE_ID")) device_id = std::atoi(env);
  aclrtStream stream = nullptr;
  bool inited = false;
  bool device_set = false;
  int* out1 = nullptr;
  int* out2 = nullptr;
  std::vector<int> host1(kBufElems, kSentinel);
  std::vector<int> host2(kBufElems, kSentinel);
  bool ok1 = true, ok2 = true;
  const bool run1 = (only == nullptr || std::strcmp(only, "k1") == 0);
  const bool run2 = (only == nullptr || std::strcmp(only, "k2") == 0);

  ACL_CHECK(aclInit(nullptr));
  inited = true;
  ACL_CHECK(aclrtSetDevice(device_id));
  device_set = true;
  ACL_CHECK(aclrtCreateStream(&stream));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&out1), kBufElems * sizeof(int),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void**>(&out2), kBufElems * sizeof(int),
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMemcpy(out1, kBufElems * sizeof(int), host1.data(),
                        kBufElems * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(out2, kBufElems * sizeof(int), host2.data(),
                        kBufElems * sizeof(int), ACL_MEMCPY_HOST_TO_DEVICE));

  if (reverse != nullptr && reverse[0] != '\0' && reverse[0] != '0') {
    if (run2) ok2 = LaunchAndCheck(k2, "k2_plain", out2, stream, host2);
    if (run1) ok1 = LaunchAndCheck(k1, "k1_inlineasm", out1, stream, host1);
  } else {
    if (run1) ok1 = LaunchAndCheck(k1, "k1_inlineasm", out1, stream, host1);
    if (run2) ok2 = LaunchAndCheck(k2, "k2_plain", out2, stream, host2);
  }

  std::printf("summary: k1_inlineasm=%s k2_plain=%s\n", ok1 ? "PASS" : "FAIL",
              ok2 ? "PASS" : "FAIL");
  g_rc = (ok1 && ok2) ? 0 : 1;

cleanup:
  if (out1) aclrtFree(out1);
  if (out2) aclrtFree(out2);
  if (stream) aclrtDestroyStream(stream);
  if (device_set) aclrtResetDevice(device_id);
  if (inited) aclFinalize();
  if (h1) dlclose(h1);
  if (h2) dlclose(h2);
  return g_rc;
}
