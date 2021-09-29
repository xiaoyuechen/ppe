#ifndef CMD_ARGS_H
#define CMD_ARGS_H

#include <cstdint>

struct Args
{
  enum class Opt : std::uint8_t
  {
    Cache = 1,
    SIMD = 1 << 1,
    OpenMP = 1 << 2,
    OpenCL = 1 << 3,
  };

  std::uint8_t optimization_mode = 0;
};

Args parseArgs(int argc, char* argv[]);

#endif
