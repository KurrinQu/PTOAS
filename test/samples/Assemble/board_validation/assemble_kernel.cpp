// ---------------------------------------------------------------------------
// PTOAS compatibility layer
//
// The upstream pto-isa headers reference some FP8/FP4 types and the
// __VEC_SCOPE__ marker that are not available on every AICore arch/toolchain
// combination (e.g. __NPU_ARCH__==2201).
//
// For our PTOAS-generated kernels we don't rely on these types today, but the
// headers still mention them in templates/static_asserts. Provide minimal
// fallbacks to keep compilation working on dav-c220.
// ---------------------------------------------------------------------------
#ifndef __VEC_SCOPE__
#define __VEC_SCOPE__
#endif

#if defined(__CCE_AICORE__) && defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2201)
typedef struct { unsigned char v; } hifloat8_t;
typedef struct { unsigned char v; } float8_e4m3_t;
typedef struct { unsigned char v; } float8_e5m2_t;
typedef struct { unsigned char v; } float8_e8m0_t;
typedef struct { unsigned char v; } float4_e1m2x2_t;
typedef struct { unsigned char v; } float4_e2m1x2_t;
#endif
#include <stdint.h>

// AICore printf support is gated behind `--cce-enable-print` on some
// toolchains. When enabled, include the CCE print header so `cce::printf`
// resolves in device compilation.
#if defined(__CCE_AICORE__) && defined(PTOAS_ENABLE_CCE_PRINT)
#include <ccelib/print/print.h>
#endif
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

// Some PTO-ISA types are only available in the __CCE_AICORE__ compilation
// path, but `bisheng -xcce` still performs a host-side parse pass.
// Provide minimal fallbacks only when the corresponding header wasn't
// pulled in by the selected arch implementation.
#if !defined(__CCE_AICORE__) && !defined(TMRGSORT_HPP)
namespace pto {
struct MrgSortExecutedNumList {
    uint16_t mrgSortList0;
    uint16_t mrgSortList1;
    uint16_t mrgSortList2;
    uint16_t mrgSortList3;
};
} // namespace pto
#endif
#ifndef __CPU_SIM
#include "acl/acl.h"
#endif

#include "pto/pto-inst.hpp"
using namespace pto;
__global__ AICORE void assemble_kernel(__gm__ float* v1, __gm__ float* v2, __gm__ float* v3) {
  unsigned v4 = 1024;
  unsigned v5 = 32;
  unsigned v6 = 256;
  unsigned v7 = 16;
  unsigned v8 = 1;
  unsigned v9 = 0;
  int32_t v10 = 32;
  int32_t v11 = 16;
  int32_t v12 = 8;
  int32_t v13 = 1;
  int64_t v14 = 0;
  int64_t v15 = 1024;
  using T = float;
  pto::Shape<1, 1, 1, 16, 16> v16 = pto::Shape<1, 1, 1, 16, 16>();
  pto::Stride<256, 256, 256, 16, 1> v17 = pto::Stride<256, 256, 256, 16, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND> v18 = GlobalTensor<float, pto::Shape<1, 1, 1, 16, 16>, pto::Stride<256, 256, 256, 16, 1>, pto::Layout::ND>(v1 + (v9 + v9 * (unsigned) v11 + v9 * (unsigned) v13), v16, v17);
  pto::Shape<1, 1, 1, 32, 32> v19 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v20 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v21 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v2 + (v9 + v9 * (unsigned) v10 + v9 * (unsigned) v13), v19, v20);
  pto::Shape<1, 1, 1, 32, 32> v22 = pto::Shape<1, 1, 1, 32, 32>();
  pto::Stride<1024, 1024, 1024, 32, 1> v23 = pto::Stride<1024, 1024, 1024, 32, 1>();
  GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND> v24 = GlobalTensor<float, pto::Shape<1, 1, 1, 32, 32>, pto::Stride<1024, 1024, 1024, 32, 1>, pto::Layout::ND>(v3 + (v9 + v9 * (unsigned) v10 + v9 * (unsigned) v13), v22, v23);
  Tile<TileType::Vec, float, 16, 16, BLayout::RowMajor, 16, 16, SLayout::NoneBox, 512, PadValue::Null> v25;
  TASSIGN(v25, v14);
  Tile<TileType::Vec, float, 32, 32, BLayout::RowMajor, 32, 32, SLayout::NoneBox, 512, PadValue::Null> v26;
  TASSIGN(v26, v15);
  TLOAD(v25, v18);
  TLOAD(v26, v21);
  set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
  TINSERT(v26, v25, v12, v12);
  set_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE3, EVENT_ID0);
  TSTORE(v24, v26);
  pipe_barrier(PIPE_ALL);
  return;
}

