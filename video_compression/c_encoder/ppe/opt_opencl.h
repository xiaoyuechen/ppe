#ifndef OPT_OPENCL_H
#define OPT_OPENCL_H

#include <stdio.h>

#ifdef __cplusplus

extern "C"
{
#endif

  void initCL (int width, int height, FILE* file);
  void convertCL(int size, float* in[3], float* out[3]);

#ifdef __cplusplus
}
#endif

#endif
