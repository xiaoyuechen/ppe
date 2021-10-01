#define CL_TARGET_OPENCL_VERSION 300

#include "opt_opencl.h"
#include <CL/cl.h>
#include <errno.h>
#include <stdarg.h>
#include <stddef.h>
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

void
CL_Error (const char *restrict format, ...)
{
  va_list args;
  va_start (args, format);
  vfprintf (stderr, format, args);
  va_end (args);
  fprintf (stderr, "\nErrorCode: %d %s\n", cl_err, GetCLErrorStr (cl_err));
  exit (EXIT_FAILURE);
}

cl_context
CreateContext (cl_device_id *device_id)
{
  // Get platform and device information
  cl_platform_id platform_id;
  cl_err = clGetPlatformIDs (1, &platform_id, 0);
  if (cl_err)
    CL_Error ("Error getting platform ID");

  cl_err = clGetDeviceIDs (platform_id, CL_DEVICE_TYPE_GPU, 1, device_id, 0);
  if (cl_err)
    CL_Error ("Error getting device ID");

  char device_name[0x100];
  clGetDeviceInfo (*device_id, CL_DEVICE_NAME, 0x100, device_name, 0);
  printf ("Using GPU: %s\n", device_name);

  // Create an OpenCL context
  cl_context context = clCreateContext (0, 1, device_id, 0, 0, &cl_err);
  if (cl_err)
    CL_Error ("Error creating cl context");

  return context;
}

cl_program
CreateProgram (const char *file_name, cl_context context)
{
  const size_t MAX_SOURCE_SIZE = 0x100000;
  FILE *fp = fopen (file_name, "r");
  if (!fp)
    CL_Error ("Error opening cl kernel file: %s", strerror (errno));
  char *src = malloc (MAX_SOURCE_SIZE);
  size_t size = fread (src, 1, MAX_SOURCE_SIZE, fp);
  fclose (fp);

  cl_program prg = clCreateProgramWithSource (context, 1, (const char **)&src,
                                              &size, &cl_err);
  if (cl_err)
    CL_Error ("Error creating cl program");
  free (src);

  cl_err = clBuildProgram (prg, 0, 0, 0, 0, 0);
  if (cl_err)
    CL_Error ("Error building cl program");

  return prg;
}

cl_kernel
CreateKernel (cl_program prg, const char *kernel_name)
{
  cl_kernel kernel = clCreateKernel (prg, kernel_name, &cl_err);
  if (cl_err)
    CL_Error ("Error creating kernel %s: %d", kernel_name, cl_err);
  return kernel;
}

cl_command_queue
CreateCommandQueue (cl_context context, cl_device_id device)
{
  cl_command_queue command_queue
      = clCreateCommandQueueWithProperties (context, device, 0, &cl_err);
  if (cl_err)
    CL_Error ("Error creating command queue");
  return command_queue;
}

cl_mem
CreateBuffer (cl_context context, size_t size)
{
  cl_mem mem = clCreateBuffer (context, CL_MEM_READ_WRITE, size, 0, &cl_err);
  if (cl_err)
    CL_Error ("Error creating buffer");
  return mem;
}

void *
AllocPinnedMem (cl_mem *pinned_buf, cl_context context, cl_command_queue cmdq,
                cl_map_flags map_flags, size_t size)
{
  *pinned_buf = clCreateBuffer (
      context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, size, 0, &cl_err);
  if (cl_err)
    CL_Error ("Error creating pinned buffer");
  void *pinned_mem = clEnqueueMapBuffer (cmdq, *pinned_buf, CL_FALSE,
                                         map_flags, 0, size, 0, 0, 0, &cl_err);
  if (cl_err)
    CL_Error ("Error mapping pinned buffer");
  return pinned_mem;
}

void
ReleasePinnedMem (cl_mem pinned_buf, void *pinned_mem, cl_command_queue cmdq)
{
  cl_err = clEnqueueUnmapMemObject (cmdq, pinned_buf, pinned_mem, 0, 0, 0);
  if (cl_err)
    CL_Error ("Error unmapping pinned buffer");
  clReleaseMemObject (pinned_mem);
}

void
EnqueueWriteBuffer (cl_command_queue cmdq, cl_mem buff, size_t size,
                    const void *data)
{
  cl_err = clEnqueueWriteBuffer (cmdq, buff, CL_FALSE, 0, size, data, 0, 0, 0);
  if (cl_err)
    CL_Error ("Error enqueueing write buffer");
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
initCL (int pwidth, int pheight, FILE *fd)
{
  struct timeval start, stop;
  gettimeofday (&start, 0);

  output_file = fd;
  width = pwidth;
  height = pheight;

  context = CreateContext (&device_id);
  program = CreateProgram ("kernel.cl", context);
  convert_kernel = CreateKernel (program, "convert");
  cmd_queue = CreateCommandQueue (context, device_id);

  // Create memory buffers on the device for each channel
  for (int i = 0; i < 3; ++i)
    {
      size_t size = width * height * sizeof (float);
      buf[i] = CreateBuffer (context, size);
    }
  clFinish (cmd_queue);

  gettimeofday (&stop, 0);
  uint32_t t = GetTimevalMicroSeconds (&start, &stop);
  fprintf (output_file, "CL init time: %u\n", t);
}

void
convertCL (int size, float *in[3], float *out[3], size_t num_thd)
{
  struct timeval start, stop;
  gettimeofday (&start, 0);

  for (size_t c = 0; c < 3; ++c)
    {
      EnqueueWriteBuffer (cmd_queue, buf[c], size * sizeof (float), in[c]);
      cl_err = clSetKernelArg (convert_kernel, c, sizeof (cl_mem), &buf[c]);
      if (cl_err)
        CL_Error ("Error setting kernel arg");
    }
  clFinish (cmd_queue);
  gettimeofday (&stop, 0);
  fprintf (output_file, "CL copy h2d time: %u\n",
           GetTimevalMicroSeconds (&start, &stop));

  gettimeofday (&start, 0);
  size_t global_item_size = num_thd;
  for (size_t offset = 0; offset < width * height; offset += num_thd)
    {
      cl_err = clEnqueueNDRangeKernel (cmd_queue, convert_kernel, 1, &offset,
                                       &global_item_size, 0, 0, 0, 0);
      if (cl_err)
        CL_Error ("Error enqueueing range kernel");

      clFinish (cmd_queue);
    }

  gettimeofday (&stop, 0);
  fprintf (output_file, "CL kernel execution time: %u\n",
           GetTimevalMicroSeconds (&start, &stop));

  gettimeofday (&start, 0);
  for (size_t c = 0; c < 3; ++c)
    {
      cl_err = clEnqueueReadBuffer (cmd_queue, buf[c], CL_FALSE, 0,
                                    size * sizeof (float), out[c], 0, 0, 0);
      if (cl_err)
        CL_Error ("Error enqueueing write buffer");
    }

  cl_err = clFinish (cmd_queue);
  if (cl_err)
    CL_Error ("Error finishing convert command queue");

  gettimeofday (&stop, 0);
  fprintf (output_file, "CL copy d2h time: %u\n",
           GetTimevalMicroSeconds (&start, &stop));
}
