#ifndef OPT_OPENCL_H
#define OPT_OPENCL_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus

extern "C"
{
#endif

  void initCL (int width, int height, FILE *file);
  void convertCL (size_t size, const float *R, const float *G, const float *B,
                  float *Y, float *Cb, float *Cr, size_t num_thd);
  void motionCL (size_t size[2], size_t block_size, const float *s[3],
                 const float *m[3], int *out_motion_vector);

#ifdef __cplusplus
}
#endif

#endif
