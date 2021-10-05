#include "opt_openacc.h"

void
convertACC (size_t size, const float *R, const float *G, const float *B,
            float *Y, float *Cb, float *Cr)
{
#pragma acc parallel loop
  for (int i = 0; i < size; ++i)
    {
      float r = R[i];
      float g = G[i];
      float b = B[i];
      float y = 0 + (0.299f * r) + (0.587f * g) + (0.113f * b);
      float cb = 128 - (0.168736f * r) - (0.331264f * g) + (0.5f * b);
      float cr = 128 + (0.5f * r) - (0.418688f * g) - (0.081312f * b);
      Y[i] = y;
      Cb[i] = cb;
      Cr[i] = cr;
    }
}
