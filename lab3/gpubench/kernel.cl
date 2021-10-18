#include <stddef.h>

kernel void
add_char (size_t s, int scale, global char *out)
{
  /* size_t local_iter = iter / get_global_size (0); */
  /* char sum = 1; */
  /* for (size_t i = 0; i < local_iter; ++i) */
  /*   sum += scale; */
  /* out[get_global_id (0)] = sum; */
}
