#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

  enum Optimization
  {
    Cache = 1,
    SIMD = 1 << 1,
    OpenMP = 1 << 2,
    OpenCL = 1 << 3,
    OpenACC = 1 << 4,
    MotionVectorOpenCL = 1 << 5
  };

  typedef struct Args
  {
    uint8_t optimization_mode;
    int opencl_num_threads;
  } Args;

  Args parseArgs (int argc, char *argv[]);

#ifdef __cplusplus
}
#endif

#endif
