#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void tid_asm_kernel(__gm__ void*);
extern "C" int call(int* out, void* stream) {
  tid_asm_kernel<<<64, 0, stream>>>((__gm__ int*)out);
  return 0;
}
