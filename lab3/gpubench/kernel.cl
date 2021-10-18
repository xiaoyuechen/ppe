kernel void
add_char (char scale)
{
  char sum = 1;
  sum += scale;
}

kernel void
add_float (float scale)
{
  char sum = 1.0f;
  sum += scale;
}

kernel void
load_seq (int size, global int *array)
{
  int loaded;
  for (int i = 0; i < size; ++i)
    {
      loaded = array[i];
    }
}

kernel void
load_rand (int size, global int *array)
{
  for (int next = 0; next < size;)
    {
      next = array[next];
    }
}
