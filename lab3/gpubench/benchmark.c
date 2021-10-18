/*
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

#define CL_TARGET_OPENCL_VERSION 300

#include <CL/cl.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

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
GetCLErrorStr (cl_int err)
{
  int ind = err >= -19 ? -err : -err - 10;
  return cl_err_str[ind];
}

#define CL_PRINT_ERROR(err, func)                                             \
  do                                                                          \
    {                                                                         \
      const char *err_str = GetCLErrorStr (err);                              \
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
          const char *err_str = GetCLErrorStr (err);                          \
          CL_PRINT_ERROR (err, x);                                            \
          exit (EXIT_FAILURE);                                                \
        }                                                                     \
    }                                                                         \
  while (0)

#define CL_CHECK_EA(x)                                                        \
  do                                                                          \
    {                                                                         \
      x;                                                                      \
      if (cl_err)                                                             \
        {                                                                     \
          const char *err_str = GetCLErrorStr (cl_err);                       \
          CL_PRINT_ERROR (cl_err, x);                                         \
          exit (EXIT_FAILURE);                                                \
        }                                                                     \
    }                                                                         \
  while (0)

void
Error (const char *restrict format, ...)
{
  va_list args;
  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fprintf (stderr, "\n");
  exit (EXIT_FAILURE);
}

cl_context
CreateContext (cl_device_id *device_id)
{
  // Get platform and device information
  cl_platform_id platform_id;
  CL_CHECK (clGetPlatformIDs (1, &platform_id, 0));
  CL_CHECK (clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, device_id, 0));

  char device_name[0x100];
  CL_CHECK (
      clGetDeviceInfo (*device_id, CL_DEVICE_NAME, 0x100, device_name, 0));
  printf ("Using GPU: %s\n", device_name);

  // Create an OpenCL context
  cl_context context = clCreateContext (0, 1, device_id, 0, 0, &cl_err);

  return context;
}

cl_program
CreateProgram (const char *file_name, cl_context context)
{
  const size_t MAX_SOURCE_SIZE = 0x100000;
  FILE *fp = fopen (file_name, "r");
  char *src = malloc (MAX_SOURCE_SIZE);
  size_t size = fread (src, 1, MAX_SOURCE_SIZE, fp);
  fclose (fp);

  cl_program prg = clCreateProgramWithSource (context, 1, (const char **)&src,
                                              &size, &cl_err);
  free (src);

  CL_CHECK (clBuildProgram (prg, 0, 0, 0, 0, 0));
  return prg;
}

cl_kernel
CreateKernel (cl_program prg, const char *kernel_name)
{
  cl_kernel kernel = clCreateKernel (prg, kernel_name, &cl_err);
  return kernel;
}

cl_command_queue
CreateCommandQueue (cl_context context, cl_device_id device)
{
  cl_command_queue command_queue
      = clCreateCommandQueueWithProperties (context, device, 0, &cl_err);
  return command_queue;
}

cl_mem
CreateBuffer (cl_context context, size_t size)
{
  cl_mem mem = clCreateBuffer (context, CL_MEM_READ_WRITE, size, 0, &cl_err);
  return mem;
}

uint32_t
GetTimevalMicroSeconds (const struct timeval *start,
                        const struct timeval *stop)
{
  return ((stop->tv_sec - start->tv_sec) * 1000000 + stop->tv_usec
          - start->tv_usec);
}

static cl_device_id device_id;
static cl_context context;
static cl_program program;
static cl_kernel convert_kernel;
static cl_command_queue cmd_queue;
static float *pinned_mem[3];
static cl_mem pinned_buf[3];
static cl_mem buf[3];
static int width, height;
static FILE *output_file;

void
init ()
{
  context = CreateContext (&device_id);
  program = CreateProgram ("kernel.cl", context);
  convert_kernel = CreateKernel (program, "add_char");
  cmd_queue = CreateCommandQueue (context, device_id);
}