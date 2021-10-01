#ifndef OPT_OPENCL_H
#define OPT_OPENCL_H

#ifdef __cplusplus
extern "C"
{
#endif

  void initCL (int width, int height);
  void convertCL(int size, float* in[3], float* out[3]);

#ifdef __cplusplus
}
#endif

#endif
