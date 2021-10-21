#ifndef OPT_OPENCL_H
#define OPT_OPENCL_H

#include <stdio.h>

#ifdef __cplusplus

extern "C"
{
#endif

  void initCL (int width, int height, FILE *file);
  void convertCL (size_t size, const float *R, const float *G, const float *B,
                  float *Y, float *Cb, float *Cr, size_t num_thd);

#ifdef __cplusplus
}
#endif

#endif
