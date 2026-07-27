#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void rmsnorm_d4096_kernel(
    __gm__ void*, __gm__ void*, __gm__ void*, __gm__ void*, float);
extern "C" int call(
    float* X, float* Y, float* W, float* RSTD, float eps, void* stream) {
  rmsnorm_d4096_kernel<<<64, 82432, stream>>>(
      (__gm__ float*)RSTD, (__gm__ float*)W, (__gm__ float*)X,
      (__gm__ float*)Y, eps);
  return 0;
}
