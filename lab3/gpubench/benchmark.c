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

#include "benchmark.h"

#include <CL/cl.h>
#include <errno.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

extern int errno;
static cl_int cl_err;

const char *const cl_err_str[]
    = { "CL_SUCCESS",
        "CL_DEVICE_NOT_FOUND",
        "CL_DEVICE_NOT_AVAILABLE",
        "CL_COMPILER_NOT_AVAILABLE",
        "CL_MEM_OBJECT_ALLOCATION_FAILURE",
        "CL_OUT_OF_RESOURCES",
        "CL_OUT_OF_HOST_MEMORY",
        "CL_PROFILING_INFO_NOT_AVAILABLE",
        "CL_MEM_COPY_OVERLAP",
        "CL_IMAGE_FORMAT_MISMATCH",
        "CL_IMAGE_FORMAT_NOT_SUPPORTED",
        "CL_BUILD_PROGRAM_FAILURE",
        "CL_MAP_FAILURE",
        "CL_MISALIGNED_SUB_BUFFER_OFFSET",
        "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST",
        "CL_COMPILE_PROGRAM_FAILURE",
        "CL_LINKER_NOT_AVAILABLE",
        "CL_LINK_PROGRAM_FAILURE",
        "CL_DEVICE_PARTITION_FAILED",
        "CL_KERNEL_ARG_INFO_NOT_AVAILABLE",
        "CL_INVALID_VALUE",
        "CL_INVALID_DEVICE_TYPE",
        "CL_INVALID_PLATFORM",
        "CL_INVALID_DEVICE",
        "CL_INVALID_CONTEXT",
        "CL_INVALID_QUEUE_PROPERTIES",
        "CL_INVALID_COMMAND_QUEUE",
        "CL_INVALID_HOST_PTR",
        "CL_INVALID_MEM_OBJECT",
        "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
        "CL_INVALID_IMAGE_SIZE",
        "CL_INVALID_SAMPLER",
        "CL_INVALID_BINARY",
        "CL_INVALID_BUILD_OPTIONS",
        "CL_INVALID_PROGRAM",
        "CL_INVALID_PROGRAM_EXECUTABLE",
        "CL_INVALID_KERNEL_NAME",
        "CL_INVALID_KERNEL_DEFINITION",
        "CL_INVALID_KERNEL",
        "CL_INVALID_ARG_INDEX",
        "CL_INVALID_ARG_VALUE",
        "CL_INVALID_ARG_SIZE",
        "CL_INVALID_KERNEL_ARGS",
        "CL_INVALID_WORK_DIMENSION",
        "CL_INVALID_WORK_GROUP_SIZE",
        "CL_INVALID_WORK_ITEM_SIZE",
        "CL_INVALID_GLOBAL_OFFSET",
        "CL_INVALID_EVENT_WAIT_LIST",
        "CL_INVALID_EVENT",
        "CL_INVALID_OPERATION",
        "CL_INVALID_GL_OBJECT",
        "CL_INVALID_BUFFER_SIZE",
        "CL_INVALID_MIP_LEVEL",
        "CL_INVALID_GLOBAL_WORK_SIZE",
        "CL_INVALID_PROPERTY",
        "CL_INVALID_IMAGE_DESCRIPTOR",
        "CL_INVALID_COMPILER_OPTIONS",
        "CL_INVALID_LINKER_OPTIONS",
        "CL_INVALID_DEVICE_PARTITION_COUNT" };

const char *
cl_error_str (cl_int err)
{
  int ind = err >= -19 ? -err : -err - 10;
  return cl_err_str[ind];
}

#define CL_PRINT_ERROR(err, func)                                             \
  do                                                                          \
    {                                                                         \
      const char *err_str = cl_error_str (err);                               \
      fprintf (stderr, "%s: When calling %s:\n", __FILE__, #func);            \
      fprintf (stderr, "%s:%lu: CL error: %s (%d)\n", __FILE__,               \
               __LINE__ + 0UL, err_str, err);                                 \
    }                                                                         \
  while (0)

#define CL_CHECK(x)                                                           \
  do                                                                          \
    {                                                                         \
      cl_int err = (x);                                                       \
      if (err)                                                                \
        {                                                                     \
          const char *err_str = cl_error_str (err);                           \
          CL_PRINT_ERROR (err, x);                                            \
          exit (EXIT_FAILURE);                                                \
        }                                                                     \
    }                                                                         \
  while (0)

#define CL_CHECK_R(x)                                                         \
  x;                                                                          \
  if (cl_err)                                                                 \
    {                                                                         \
      const char *err_str = cl_error_str (cl_err);                            \
      CL_PRINT_ERROR (cl_err, x);                                             \
      exit (EXIT_FAILURE);                                                    \
    }                                                                         \
  while (0)

#define KERNELS                                                               \
  X (add_char)                                                                \
  X (add_float)                                                               \
  X (load_seq)                                                                \
  X (load_rand)

#define X(name) name##_kernel,
enum Kernel
{
  KERNELS kernel_count
};
#undef X

#define X(name) [name##_kernel] = #name,
static const char *const kernel_names[kernel_count] = { KERNELS };
#undef X

static const char kernel_file[] = "kernel.cl";
static const size_t MAX_SOURCE_SIZE = 0x100000;

static cl_device_id device_id;
static cl_context context;
static cl_program program;
static cl_kernel kernels[kernel_count];
static cl_command_queue cmd_queue;

void
init ()
{
  cl_platform_id platform_id;
  CL_CHECK (clGetPlatformIDs (1, &platform_id, 0));
  CL_CHECK (
      clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id, 0));

  char device_name[0x100];
  CL_CHECK (
      clGetDeviceInfo (device_id, CL_DEVICE_NAME, 0x100, device_name, 0));
  printf ("Using GPU: %s\n", device_name);

  context = CL_CHECK_R (clCreateContext (0, 1, &device_id, 0, 0, &cl_err));

  /* TODO(Xiaoyue Chen): error handling file reading */
  FILE *fp = fopen (kernel_file, "r");
  char *src = malloc (MAX_SOURCE_SIZE);
  size_t size = fread (src, 1, MAX_SOURCE_SIZE, fp);
  fclose (fp);

  program = CL_CHECK_R (clCreateProgramWithSource (
      context, 1, (const char **)&src, &size, &cl_err));
  free (src);

  CL_CHECK (clBuildProgram (program, 0, 0, 0, 0, 0));

  for (size_t i = 0; i < kernel_count; ++i)
    {
      kernels[i]
          = CL_CHECK_R (clCreateKernel (program, kernel_names[i], &cl_err));
    }

  cmd_queue = CL_CHECK_R (
      clCreateCommandQueueWithProperties (context, device_id, 0, &cl_err));
}

#define START_TIMER(timer) clock_t timer##_start = clock ()

#define END_TIMER(timer)                                                      \
  clock_t timer##_end = clock ();                                             \
  double timer = ((double)(timer##_end - timer##_start)) / CLOCKS_PER_SEC

void
add_char (size_t size)
{
  cl_kernel kernel = kernels[add_char_kernel];

  char scale = 7;
  int load_per_thd = 10000;
  size_t dim = size / load_per_thd;
  cl_mem buf = CL_CHECK_R (clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                           dim * sizeof (char), 0, &cl_err));
  CL_CHECK (clSetKernelArg (kernel, 0, sizeof (int), &load_per_thd));
  CL_CHECK (clSetKernelArg (kernel, 1, sizeof (char), &scale));
  CL_CHECK (clSetKernelArg (kernel, 2, sizeof (cl_mem), &buf));
  CL_CHECK (clFinish (cmd_queue));

  START_TIMER (timer);
  CL_CHECK (
      clEnqueueNDRangeKernel (cmd_queue, kernel, 1, 0, &dim, 0, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  END_TIMER (timer);

  printf ("%s %zu takes %u us or %g GOPS\n", __func__, size,
          (uint32_t)(timer * 1E6), size / 1E9 / timer);
}

void
add_float (size_t size)
{
  cl_kernel kernel = kernels[add_float_kernel];

  float scale = 7.0f;
  int load_per_thd = 10000;
  size_t dim = size / load_per_thd;
  cl_mem buf = CL_CHECK_R (clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                           dim * sizeof (float), 0, &cl_err));
  CL_CHECK (clSetKernelArg (kernel, 0, sizeof (int), &load_per_thd));
  CL_CHECK (clSetKernelArg (kernel, 1, sizeof (float), &scale));
  CL_CHECK (clSetKernelArg (kernel, 2, sizeof (cl_mem), &buf));
  CL_CHECK (clFinish (cmd_queue));

  START_TIMER (timer);
  CL_CHECK (
      clEnqueueNDRangeKernel (cmd_queue, kernel, 1, 0, &dim, 0, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  END_TIMER (timer);

  printf ("%s %zu takes %u us or %g GOPS\n", __func__, size,
          (uint32_t)(timer * 1E6), size / 1E9 / timer);
}

void
load_int_seq (size_t size)
{
  cl_mem buf = CL_CHECK_R (clCreateBuffer (context, CL_MEM_READ_ONLY,
                                           sizeof (int) * size, 0, &cl_err));
  int out = 0;
  cl_mem bufout = CL_CHECK_R (
      clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof (int), &out, &cl_err));

  int size_int = size;

  CL_CHECK (
      clSetKernelArg (kernels[load_seq_kernel], 0, sizeof (int), &size_int));
  CL_CHECK (
      clSetKernelArg (kernels[load_seq_kernel], 1, sizeof (cl_mem), &buf));
  CL_CHECK (
      clSetKernelArg (kernels[load_seq_kernel], 2, sizeof (cl_mem), &bufout));
  CL_CHECK (clFinish (cmd_queue));

  size_t one = 1;

  START_TIMER (timer);
  CL_CHECK (clEnqueueNDRangeKernel (cmd_queue, kernels[load_seq_kernel], 1, 0,
                                    &one, 0, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  END_TIMER (timer);

  printf ("%s %zu takes %u us or %g GOPS\n", __func__, size,
          (uint32_t)(timer * 1E6), size / 1E9 / timer);

  clReleaseMemObject (buf);
}

void
load_int_rand (size_t size)
{
  cl_kernel kernel = kernels[load_rand_kernel];

  int *array = malloc (sizeof (int) * size);
  int out = 0;

  for (int j = 0; j < size; j++)
    {
      int r = j + (rand () % (size - j));
      int tmp = array[r];
      array[r] = array[j];
      array[j] = tmp;
    }

  cl_mem buf = CL_CHECK_R (
      clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof (int) * size, array, &cl_err));


  cl_mem bufout = CL_CHECK_R (
      clCreateBuffer (context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                      sizeof (int), &out, &cl_err));


  int size_int = size;
  CL_CHECK (clSetKernelArg (kernel, 0, sizeof (int), &size_int));
  CL_CHECK (clSetKernelArg (kernel, 1, sizeof (cl_mem), &buf));
  CL_CHECK (clSetKernelArg (kernel, 2, sizeof (cl_mem), &bufout));
  CL_CHECK (clFinish (cmd_queue));

  START_TIMER (timer);
  size_t one = 1;
  CL_CHECK (
      clEnqueueNDRangeKernel (cmd_queue, kernel, 1, 0, &one, 0, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  END_TIMER (timer);

  printf ("%s %zu takes %u us or %g GOPS\n", __func__, size,
          (uint32_t)(timer * 1E6), size / 1E9 / timer);

  clReleaseMemObject (buf);
  free (array);
}

void
transfer_data (size_t size)
{
  char *array = malloc (sizeof (char) * size);
  cl_mem buf = CL_CHECK_R (clCreateBuffer (context, CL_MEM_READ_ONLY,
                                           sizeof (char) * size, 0, &cl_err));
  CL_CHECK (clFinish (cmd_queue));

  START_TIMER (timer);
  CL_CHECK (clEnqueueWriteBuffer (cmd_queue, buf, 0, 0, size, array, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  END_TIMER (timer);
  printf ("%s %zu takes %u us or %g GB/s\n", __func__, size,
          (uint32_t)(timer * 1E6), size / 1E9 / timer);
}
