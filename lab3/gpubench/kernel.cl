/*
 * gpubench --- benchmarks for GPUs
 * Copyright (C) 2021  Xiaoyue Chen and Hannah Atmer
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

kernel void
add_char (int iter, char scale, global char *out)
{
  char sum = 1;
  for (int i = 0; i < iter; ++i)
    {
      sum += scale;
    }
  out[get_global_id (0)] = sum;
}

kernel void
add_float (int iter, float scale, global float *out)
{
  float sum = 1.0f;
  for (int i = 0; i < iter; ++i)
    {
      sum += scale;
    }
  out[get_global_id (0)] = sum;
}

kernel void
load_seq (int size, global int *array, global int *out)
{
  int loaded;
  for (int i = 0; i < size; ++i)
    {
      loaded = array[i];
    }
  *out = loaded;
}

kernel void
load_rand (int size, global int *array, global int *out)
{
  int next = 0;
  for (int i = 0; i < size; ++i)
    {
      next = array[next];
    }
  *out = next;
}
