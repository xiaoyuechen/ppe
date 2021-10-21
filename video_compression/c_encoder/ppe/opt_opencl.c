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

void initMotionKernel ();

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
          CL_PRINT_ERROR (err, x);                                            \
          exit (EXIT_FAILURE);                                                \
        }                                                                     \
    }                                                                         \
  while (0)

#define CL_CHECK_R(x)                                                         \
  x;                                                                          \
  if (cl_err)                                                                 \
    {                                                                         \
      CL_PRINT_ERROR (cl_err, x);                                             \
      exit (EXIT_FAILURE);                                                    \
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

void
EnqueueWriteBuffer (cl_command_queue cmdq, cl_mem buff, size_t size,
                    const void *data)
{
  cl_err = clEnqueueWriteBuffer (cmdq, buff, CL_FALSE, 0, size, data, 0, 0, 0);
  if (cl_err)
    CL_Error ("Error enqueueing write buffer");
}

void
EnqueueReadBuffer (cl_command_queue cmdq, cl_mem buff, size_t size, void *data)
{
  cl_err = clEnqueueReadBuffer (cmdq, buff, CL_FALSE, 0, size, data, 0, 0, 0);
  if (cl_err)
    CL_Error ("Error enqueueing read buffer");
}

uint32_t
GetTimevalMicroSeconds (const struct timeval *start,
                        const struct timeval *stop)
{
  return ((stop->tv_sec - start->tv_sec) * 1000000 + stop->tv_usec
          - start->tv_usec);
}

static int width, height;
static FILE *output_file;

static cl_device_id device_id;
static cl_context context;
static cl_program program;
static cl_command_queue cmd_queue;

/* CL objects for convert kernel */
static cl_kernel convert_kernel;
static cl_mem buf[3];

/* CL objects for motion kernel */
static cl_kernel motion_kernel;
static cl_mem sbuf[3];
static cl_mem mbuf[3];
static cl_mem motion_buf;

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
  motion_kernel = CreateKernel (program, "motionVectorSearch");
  cmd_queue = CreateCommandQueue (context, device_id);

  // Create memory buffers on the device for each channel
  /* for (int i = 0; i < 3; ++i) */
  /*   { */
  /*     size_t size = width * height * sizeof (float); */
  /*     buf[i] = CreateBuffer (context, size); */
  /*   } */
  initMotionKernel ();

  clFinish (cmd_queue);

  gettimeofday (&stop, 0);
  uint32_t t = GetTimevalMicroSeconds (&start, &stop);
  fprintf (output_file, "CL init time: %u\n", t);
}

void
convertCL (size_t size, const float *R, const float *G, const float *B,
           float *Y, float *Cb, float *Cr, size_t num_thd)
{
  struct timeval start, stop;
  gettimeofday (&start, 0);

  const float *in[3] = { R, G, B };
  float *out[3] = { Y, Cb, Cr };

  for (size_t c = 0; c < 3; ++c)
    EnqueueWriteBuffer (cmd_queue, buf[c], size * sizeof (float), in[c]);

  for (size_t c = 0; c < 3; ++c)
    {
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
  for (size_t offset = 0; offset < size; offset += num_thd)
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
      EnqueueReadBuffer (cmd_queue, buf[c], size * sizeof (float), out[c]);
    }

  cl_err = clFinish (cmd_queue);
  if (cl_err)
    CL_Error ("Error finishing convert command queue");

  gettimeofday (&stop, 0);
  fprintf (output_file, "CL copy d2h time: %u\n",
           GetTimevalMicroSeconds (&start, &stop));
}

void
initMotionKernel ()
{
  size_t size[] = { width, height };
  size_t block_size = 16;
  for (size_t c = 0; c < 3; ++c)
    {
      size_t buff_size = size[0] * size[1] * sizeof (float);
      sbuf[c] = CL_CHECK_R (
          clCreateBuffer (context, CL_MEM_READ_ONLY, buff_size, 0, &cl_err));
      mbuf[c] = CL_CHECK_R (
          clCreateBuffer (context, CL_MEM_READ_ONLY, buff_size, 0, &cl_err));
    }

  size_t motion_buf_size = (size[0] / block_size - 2)
                           * (size[1] / block_size - 2) * sizeof (int) * 2;
  motion_buf = CL_CHECK_R (clCreateBuffer (context, CL_MEM_WRITE_ONLY,
                                           motion_buf_size, 0, &cl_err));

  for (size_t c = 0; c < 3; ++c)
    {
      CL_CHECK (clSetKernelArg (motion_kernel, c, sizeof (cl_mem), &sbuf[c]));
      CL_CHECK (
          clSetKernelArg (motion_kernel, 3 + c, sizeof (cl_mem), &mbuf[c]));
    }

  CL_CHECK (clSetKernelArg (motion_kernel, 6, sizeof (cl_mem), &motion_buf));
}

void
motionCL (size_t size[2], size_t block_size, const float *s[3],
          const float *m[3], int *out_motion_vector)
{
  size_t local_work_size[2] = { block_size, block_size };
  for (size_t c = 0; c < 3; ++c)
    {
      size_t buff_size = size[0] * size[1] * sizeof (float);
      CL_CHECK (clEnqueueWriteBuffer (cmd_queue, sbuf[c], 0, 0, buff_size,
                                      s[c], 0, 0, 0));
      CL_CHECK (clEnqueueWriteBuffer (cmd_queue, mbuf[c], 0, 0, buff_size,
                                      m[c], 0, 0, 0));
    }
  CL_CHECK (clEnqueueNDRangeKernel (cmd_queue, motion_kernel, 2, 0, size,
                                    local_work_size, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
  size_t motion_buf_size = (size[0] / block_size - 2)
                           * (size[1] / block_size - 2) * sizeof (int) * 2;
  CL_CHECK (clEnqueueReadBuffer (cmd_queue, motion_buf, 0, 0, motion_buf_size,
                                 out_motion_vector, 0, 0, 0));
  CL_CHECK (clFinish (cmd_queue));
}
