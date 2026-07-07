
#ifdef _WIN32
#define TL_EXPORT __declspec(dllexport)
#else
#define TL_EXPORT
#endif

#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" TL_EXPORT const char* get_last_error() {
    return error_buf;
}

extern "C" TL_EXPORT int init() {
    error_buf[0] = '\0';
    
    return 0;
}


#ifndef AICORE
#define AICORE [aicore]
#endif
extern "C" __global__ AICORE void main_kernel(__gm__ void *, __gm__ void *, __gm__ void *, __gm__ void *, float);

extern "C" TL_EXPORT int call(void * X, void * Y, void * W, void * RSTD, float eps, void *stream) {
  main_kernel<<<(64 * 1 * 1), 82496, stream>>>((__gm__ float *)RSTD, (__gm__ float *)W, (__gm__ float *)X, (__gm__ float *)Y, (float)eps);
  return 0;
}
