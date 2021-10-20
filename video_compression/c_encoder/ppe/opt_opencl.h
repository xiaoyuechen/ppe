#ifndef OPT_OPENCL_H
#define OPT_OPENCL_H

#include <stdio.h>
#include <vector>
#include "custom_types.h"

#ifdef __cplusplus

extern "C"
{
#endif

  void initCL (int width, int height, FILE *file);
  void convertCL (size_t size, const float *R, const float *G, const float *B,
                  float *Y, float *Cb, float *Cr, size_t num_thd);
  void initMVCL ();
  void motionVectorsCL (std::vector<mVector> *motion_vectors, Frame *source, Frame *match, int width, int height);

#ifdef __cplusplus
}
#endif

#endif
