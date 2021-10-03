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
      float y
          = 0 + ((float)0.299 * r) + ((float)0.587 * g) + ((float)0.113 * b);
      float cb = 128 - ((float)0.168736 * r) - ((float)0.331264 * g)
                 + ((float)0.5 * b);
      float cr = 128 + ((float)0.5 * r) - ((float)0.418688 * g)
                 - ((float)0.081312 * b);
      Y[i] = y;
      Cb[i] = cb;
      Cr[i] = cr;
    }
}
