#ifndef OPT_OPENACC_H
#define OPT_OPENACC_H

#include <stddef.h>

#ifdef __cplusplus

extern "C"
{
#endif

  void convertACC (size_t size, const float *R, const float *G, const float *B,
                   float *Y, float *Cb, float *Cr);

#ifdef __cplusplus
}
#endif

#endif
