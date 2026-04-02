#include "acl/acl.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void LaunchScope3Incore0Incore0(float *attn_out, uint16_t *hidden_states,
                                float *resid_out, uint16_t *wo, int32_t ob_idx,
                                void *stream);

#define ACL_CHECK(expr)                                                        \
  do {                                                                         \
    aclError _ret = (expr);                                                    \
    if (_ret != ACL_SUCCESS) {                                                 \
      std::fprintf(stderr, "[ACL ERROR] %s failed: %d (%s:%d)\n", #expr,       \
                   (int)_ret, __FILE__, __LINE__);                             \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static uint16_t floatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  bits += 0x7FFFu + lsb;
  return static_cast<uint16_t>(bits >> 16);
}

static float bf16BitsToFloat(uint16_t bits) {
  uint32_t value = static_cast<uint32_t>(bits) << 16;
  float result = 0.0f;
  std::memcpy(&result, &value, sizeof(result));
  return result;
}

static float attnValue(int row, int col) {
  const int v = (row * 17 + col * 29 + 3) % 23;
  return static_cast<float>(v - 11) / 16.0f;
}

static float hiddenValue(int row, int col) {
  const int v = (row * 11 + col * 7 + 5) % 19;
  return static_cast<float>(v - 9) / 8.0f;
}

static float weightValue(int row, int col) {
  const int v = (row * 5 + col * 13 + 1) % 29;
  return static_cast<float>(v - 14) / 32.0f;
}

int main() {
  constexpr int Rows = 16;
  constexpr int Dim = 5120;
  constexpr int BlockCols = 64;
  constexpr int ObInner = 8;
  constexpr int ObWidth = BlockCols * ObInner;
  constexpr int ObIdx = 0;
  constexpr float InitValue = -777.0f;
  constexpr float Atol = 1e-2f;
  constexpr float Rtol = 2e-2f;

  constexpr size_t attnElems = static_cast<size_t>(Rows) * Dim;
  constexpr size_t hiddenElems = static_cast<size_t>(Rows) * Dim;
  constexpr size_t outElems = static_cast<size_t>(Rows) * Dim;
  constexpr size_t weightElems = static_cast<size_t>(Dim) * Dim;

  constexpr size_t attnBytes = attnElems * sizeof(float);
  constexpr size_t hiddenBytes = hiddenElems * sizeof(uint16_t);
  constexpr size_t outBytes = outElems * sizeof(float);
  constexpr size_t weightBytes = weightElems * sizeof(uint16_t);

  std::vector<float> hostAttn(attnElems, 0.0f);
  std::vector<float> hostAttnRounded(attnElems, 0.0f);
  std::vector<uint16_t> hostHidden(hiddenElems, 0);
  std::vector<float> hostHiddenF32(hiddenElems, 0.0f);
  std::vector<uint16_t> hostWeight(weightElems, 0);
  std::vector<float> hostOut(outElems, InitValue);
  std::vector<float> hostGolden(outElems, InitValue);

  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < Dim; ++col) {
      const size_t idx = static_cast<size_t>(row) * Dim + col;
      hostAttn[idx] = attnValue(row, col);
      hostAttnRounded[idx] = bf16BitsToFloat(floatToBf16Bits(hostAttn[idx]));
      hostHidden[idx] = floatToBf16Bits(hiddenValue(row, col));
      hostHiddenF32[idx] = bf16BitsToFloat(hostHidden[idx]);
    }
  }

  for (int row = 0; row < Dim; ++row) {
    for (int col = 0; col < Dim; ++col) {
      const size_t idx = static_cast<size_t>(row) * Dim + col;
      hostWeight[idx] = floatToBf16Bits(weightValue(row, col));
    }
  }

  const int colBase = ObIdx * ObWidth;
  for (int row = 0; row < Rows; ++row) {
    for (int localCol = 0; localCol < ObWidth; ++localCol) {
      const int outCol = colBase + localCol;
      float acc = 0.0f;
      for (int k = 0; k < Dim; ++k) {
        const float a = hostAttnRounded[static_cast<size_t>(row) * Dim + k];
        const float b =
            bf16BitsToFloat(hostWeight[static_cast<size_t>(k) * Dim + outCol]);
        acc += a * b;
      }
      hostGolden[static_cast<size_t>(row) * Dim + outCol] =
          acc + hostHiddenF32[static_cast<size_t>(row) * Dim + outCol];
    }
  }

  ACL_CHECK(aclInit(nullptr));
  ACL_CHECK(aclrtSetDevice(0));

  aclrtStream stream = nullptr;
  ACL_CHECK(aclrtCreateStream(&stream));

  float *devAttn = nullptr;
  uint16_t *devHidden = nullptr;
  float *devOut = nullptr;
  uint16_t *devWeight = nullptr;

  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devAttn), attnBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devHidden), hiddenBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devOut), outBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));
  ACL_CHECK(aclrtMalloc(reinterpret_cast<void **>(&devWeight), weightBytes,
                        ACL_MEM_MALLOC_HUGE_FIRST));

  ACL_CHECK(aclrtMemcpy(devAttn, attnBytes, hostAttn.data(), attnBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devHidden, hiddenBytes, hostHidden.data(), hiddenBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devOut, outBytes, hostOut.data(), outBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));
  ACL_CHECK(aclrtMemcpy(devWeight, weightBytes, hostWeight.data(), weightBytes,
                        ACL_MEMCPY_HOST_TO_DEVICE));

  LaunchScope3Incore0Incore0(devAttn, devHidden, devOut, devWeight, ObIdx,
                             stream);

  ACL_CHECK(aclrtSynchronizeStream(stream));
  ACL_CHECK(aclrtMemcpy(hostOut.data(), outBytes, devOut, outBytes,
                        ACL_MEMCPY_DEVICE_TO_HOST));

  int mismatchCount = 0;
  for (int row = 0; row < Rows; ++row) {
    for (int col = 0; col < Dim; ++col) {
      const size_t idx = static_cast<size_t>(row) * Dim + col;
      const float got = hostOut[idx];
      const float expect = hostGolden[idx];
      const float diff = std::fabs(got - expect);
      const float limit = Atol + Rtol * std::fabs(expect);
      if (diff > limit) {
        if (mismatchCount < 16) {
          std::fprintf(stderr,
                       "Mismatch at (%d, %d): got %.6f, expect %.6f, diff %.6f, "
                       "limit %.6f\n",
                       row, col, got, expect, diff, limit);
        }
        ++mismatchCount;
      }
    }
  }

  if (mismatchCount == 0) {
    std::puts("scope3_incore_0_incore_0 passed.");
  } else {
    std::fprintf(stderr, "Found %d mismatches.\n", mismatchCount);
  }

  aclrtFree(devWeight);
  aclrtFree(devOut);
  aclrtFree(devHidden);
  aclrtFree(devAttn);
  aclrtDestroyStream(stream);
  aclrtResetDevice(0);
  aclFinalize();
  return mismatchCount == 0 ? 0 : 1;
}
