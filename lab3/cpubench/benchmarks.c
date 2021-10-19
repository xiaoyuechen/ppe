#include <immintrin.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define ITERS 1000000000
#define NUMBER_OF_RUNS 1

struct timeval start_time, end_time;

int arraySizes[] = { 64, 128, 1024, 65536, 33554432, 1000 * 1000 * 1000 };
const int numArraySizes = sizeof (arraySizes) / sizeof (int);

void cpu_seqLoad (int *array, int arraySize);
void cpu_randLoad (int *array);

void
cpu_8Add ()
{

  char sum = 1;

  __m256i sum_v0
      = _mm256_set_epi8 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m256i sum_v1
      = _mm256_set_epi8 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m256i sum_v2
      = _mm256_set_epi8 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
  __m256i addthis_v
      = _mm256_set_epi8 (0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

  for (int run = 0; run < NUMBER_OF_RUNS; run++)
    {

      gettimeofday (&start_time, NULL);

#pragma omp parallel for
      for (long test = 0; test < ITERS; test += 16 * 4)
        {
          sum_v0 = _mm256_add_epi8 (addthis_v, sum_v0);
          sum_v1 = _mm256_add_epi8 (addthis_v, sum_v1);
          sum_v2 = _mm256_add_epi8 (addthis_v, sum_v2);
        }

      gettimeofday (&end_time, NULL);

      double time_in_sec
          = (end_time.tv_sec + end_time.tv_usec / 1000000.0)
            - (start_time.tv_sec + start_time.tv_usec / 1000000.0);
      double GOPS = (ITERS / time_in_sec) / 1000000000 * 3 * 32;
      for (int i = 0; i < 16; i++)
        sum += ((char *)(&sum_v0))[i] + ((char *)(&sum_v1))[i]
               + ((char *)(&sum_v2))[i];
      printf ("sum: %d\n", sum);
      printf ("Completed %d adds in %g seconds for %g GOPS.\n", ITERS,
              time_in_sec, GOPS);
    }
}

void
cpu_FPAdd ()
{

  float sum = 1.0;

  __m256 sum_v0 = _mm256_set_ps (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  __m256 sum_v1 = _mm256_set_ps (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  __m256 sum_v2 = _mm256_set_ps (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  __m256 addthis_v = _mm256_set_ps (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

  for (int run = 0; run < NUMBER_OF_RUNS; run++)
    {

      gettimeofday (&start_time, NULL);

#pragma omp parallel for
      for (long test = 0; test < ITERS; test += 16 * 4)
        {
          sum_v0 = _mm256_add_ps (addthis_v, sum_v0);
          sum_v1 = _mm256_add_ps (addthis_v, sum_v1);
          sum_v2 = _mm256_add_ps (addthis_v, sum_v2);
        }

      gettimeofday (&end_time, NULL);

      double time_in_sec
          = (end_time.tv_sec + end_time.tv_usec / 1000000.0)
            - (start_time.tv_sec + start_time.tv_usec / 1000000.0);
      double GOPS = (ITERS / time_in_sec) / 1000000000 * 3 * 8;
      for (int i = 0; i < 16; i++)
        sum += ((float *)(&sum_v0))[i] + ((float *)(&sum_v1))[i]
               + ((float *)(&sum_v2))[i];
      printf ("sum: %f\n", sum);
      printf ("Completed %d adds in %g seconds for %g GOPS.\n", ITERS,
              time_in_sec, GOPS);
    }
}

void
run_cpu_seqLoad ()
{
  for (int i = 0; i < numArraySizes; i++)
    {
      int arraySize = arraySizes[i];
      int *array = malloc (arraySize * sizeof (int));
      printf ("Sequential load with ArraySize %d\n", arraySize);
      cpu_seqLoad (array, arraySize);
      free (array);
    }
}

void
cpu_seqLoad (int *array, int numElems)
{

  int elem;

  for (int run = 0; run < NUMBER_OF_RUNS; run++)
    {

      gettimeofday (&start_time, NULL);

      for (size_t j = 0; j < ITERS; j += numElems)
        {
          for (size_t i = 0; i < numElems; ++i)
            {
              elem = array[i];
            }
        }
      size_t reminder = ITERS % numElems;
      for (size_t i = 0; i < reminder; ++i)
        {
          elem = array[i];
        }

      gettimeofday (&end_time, NULL);

      double time_in_sec
          = (end_time.tv_sec + end_time.tv_usec / 1000000.0)
            - (start_time.tv_sec + start_time.tv_usec / 1000000.0);
      double GOPS = (ITERS / time_in_sec) / 1000000000;
      printf ("Completed %d loads in %g seconds for %g GOPS.\n", ITERS,
              time_in_sec, GOPS);
    }
}

void
run_cpu_randLoad ()
{
  int tmp;
  int r;

  for (int i = 0; i < numArraySizes; i++)
    {

      int arraySize = arraySizes[i];
      int *array = malloc (arraySize * sizeof (int));

      // set each element to its index
      for (int k = 0; k < arraySize; k++)
        {
          array[k] = k;
        }

      // for each element, swap with a random element
      for (int j = 0; j < arraySize; j++)
        {
          r = j + (rand () % (arraySize - j));
          tmp = array[r];
          array[r] = array[j];
          array[j] = tmp;
        }

      printf ("Random load with ArraySize %d\n", arraySize);
      cpu_randLoad (array);
      free (array);
    }
}
void
cpu_randLoad (int *array)
{

  register int elem = 0;

  for (int run = 0; run < NUMBER_OF_RUNS; run++)
    {

      gettimeofday (&start_time, NULL);

      for (long test = 0; test < ITERS; test++)
        {
          elem = array[elem];
        }

      gettimeofday (&end_time, NULL);

      double time_in_sec
          = (end_time.tv_sec + end_time.tv_usec / 1000000.0)
            - (start_time.tv_sec + start_time.tv_usec / 1000000.0);
      double GOPS = (ITERS / time_in_sec) / 1000000000;
      printf ("Completed %d loads in %g seconds for %g GOPS.\n", ITERS,
              time_in_sec, GOPS);
    }
}

int
main ()
{
  cpu_8Add ();
  cpu_FPAdd ();
  run_cpu_seqLoad ();
  run_cpu_randLoad ();
}
