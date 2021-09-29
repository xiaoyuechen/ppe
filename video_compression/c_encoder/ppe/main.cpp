#include "cmd_args.h"
#include "config.h"
#include "custom_types.h"
#include "dct8x8_block.h"
#include "xml_aux.h"
#include <iostream>
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <sys/time.h>
#include <tiff.h>
#include <tiffio.h>
#include <vector>
#include <CL/cl.h>

using namespace std;

#define MAX_SOURCE_SIZE (0x100000)
Args args;

void
loadImage (int number, string path, Image **photo)
{
  string filename;
  TIFFRGBAImage img;
  char emsg[1024];

  filename = path + to_string (number) + ".tiff";
  TIFF *tif = TIFFOpen (filename.c_str (), "r");
  if (tif == NULL)
    fprintf (stderr, "Failed opening image: %s\n", filename.c_str ());
  if (!(TIFFRGBAImageBegin (&img, tif, 0, emsg)))
    TIFFError (filename.c_str (), "%s", emsg);

  uint32_t w, h;
  size_t npixels;
  uint32_t *raster;

  TIFFGetField (tif, TIFFTAG_IMAGEWIDTH, &w);
  TIFFGetField (tif, TIFFTAG_IMAGELENGTH, &h);
  npixels = w * h;
  raster = (uint32_t *)_TIFFmalloc (npixels * sizeof (uint32_t));

  TIFFReadRGBAImage (tif, w, h, raster, 0);

  if (*photo == NULL)
    *photo = new Image ((int)w, (int)h, FULLSIZE);

  // Matlab and LibTIFF store the image diferently.
  // Necessary to mirror the image horizonatly to be consistent
  for (int j = 0; j < (int)w; j++)
    {
      for (int i = 0; i < (int)h; i++)
        {
          // The inversion is ON PURPOSE
          (*photo)->rc->data[(h - 1 - i) * w + j]
              = (float)TIFFGetR (raster[i * w + j]);
          (*photo)->gc->data[(h - 1 - i) * w + j]
              = (float)TIFFGetG (raster[i * w + j]);
          (*photo)->bc->data[(h - 1 - i) * w + j]
              = (float)TIFFGetB (raster[i * w + j]);
        }
    }

  _TIFFfree (raster);
  TIFFRGBAImageEnd (&img);
  TIFFClose (tif);
}



void
convertRGBtoYCbCr (Image *in, Image *out)
{
  int width = in->width;
  int height = in->height;

  // Load the kernel source code into the array source_str
  FILE *fp;
  char *source_str;
  size_t source_size;

  // open kernel file
  fp = fopen("kernel.cl", "r");
  if (!fp) {
      fprintf(stderr, "Failed to load kernel.\n");
      exit(1);
  }
  source_str = (char*)malloc(MAX_SOURCE_SIZE);
  source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
  fclose( fp );

  // Get platform and device information
  cl_platform_id platform_id = NULL;
  cl_device_id device_id = NULL;   
  cl_uint ret_num_devices;
  cl_uint ret_num_platforms;
  cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
  ret = clGetDeviceIDs( platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

  // Create an OpenCL context
  cl_context context = clCreateContext( NULL, 1, &device_id, NULL, NULL, &ret);

  // Create a command queue
  cl_command_queue command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

  // Create memory buffers on the device for each vector 
  cl_mem r_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, width * sizeof(float), NULL, &ret);
  cl_mem g_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, width * sizeof(float), NULL, &ret);
  cl_mem b_mem_obj = clCreateBuffer(context, CL_MEM_READ_ONLY, width * sizeof(float), NULL, &ret);
  
  cl_mem rc_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * sizeof(float), NULL, &ret);
  cl_mem gc_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * sizeof(float), NULL, &ret);
  cl_mem bc_mem_obj = clCreateBuffer(context, CL_MEM_WRITE_ONLY, width * sizeof(float), NULL, &ret);

  // Create a program from the kernel source
  cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source_str, (const size_t *)&source_size, &ret);

  // Build the program
  ret = clBuildProgram(program, 1, &device_id, NULL, NULL, NULL);

  // Create the OpenCL kernel
  cl_kernel kernel = clCreateKernel(program, "convertRGBtoYCbCr", &ret);

  for (int row = 0; row < height; row++)
  {
    // Copy data to GPU
    ret = clEnqueueWriteBuffer(command_queue, r_mem_obj, CL_TRUE, 0, width * sizeof(float), in->rc->data[row], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, g_mem_obj, CL_TRUE, 0, width * sizeof(float), in->gc->data[row], 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(command_queue, b_mem_obj, CL_TRUE, 0, width * sizeof(float), in->bc->data[row], 0, NULL, NULL);

    // Set the arguments of the kernel
    ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&r_mem_obj);
    ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&g_mem_obj);
    ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&b_mem_obj);

    // Execute the OpenCL kernel
    size_t global_item_size = width; // Process the entire row
    size_t local_item_size = 64; // Divide work items into groups of 64
    ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &global_item_size, &local_item_size, 0, NULL, NULL);
    
    // Copy data back from the GPU
    ret = clEnqueueReadBuffer(command_queue, rc_mem_obj, CL_TRUE, 0, width * sizeof(float), out->rc->data[row], 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, gc_mem_obj, CL_TRUE, 0, width * sizeof(float), out->gc->data[row], 0, NULL, NULL);
    ret = clEnqueueReadBuffer(command_queue, bc_mem_obj, CL_TRUE, 0, width * sizeof(float), out->bc->data[row], 0, NULL, NULL);

    // Clean up
    ret = clFlush(command_queue);
    ret = clFinish(command_queue);
    ret = clReleaseKernel(kernel);
    ret = clReleaseProgram(program);
    ret = clReleaseMemObject(r_mem_obj);
    ret = clReleaseMemObject(b_mem_obj);
    ret = clReleaseMemObject(g_mem_obj);
    ret = clReleaseMemObject(rc_mem_obj);
    ret = clReleaseMemObject(bc_mem_obj);
    ret = clReleaseMemObject(gc_mem_obj);
    ret = clReleaseCommandQueue(command_queue);
    ret = clReleaseContext(context);
    return 0;
  }

}





Channel *
lowPass (Channel *in, Channel *out)
{
  // Applies a simple 3-tap low-pass filter in the X- and Y- dimensions.
  // E.g., blur
  // weights for neighboring pixels
  const float a = 0.25;
  const float b = 0.5;
  const float c = 0.25;

  int width = in->width;
  int height = in->height;

// out = in; TODO Is this necessary?

  for (int i = 0; i < width * height; i++)
    out->data[i] = in->data[i];

    // In X

  {
    int columns_per_thread
        = (width + omp_get_num_threads () - 1) / omp_get_num_threads ();
    int col_begin = 1 + omp_get_thread_num () * columns_per_thread;
    int col_end = col_begin + columns_per_thread;

    for (int row = 1; row < height - 1; row++)
      for (int col = col_begin; col < std::min (col_end, width - 1); col++)
        {
          out->data[row * width + col]
              = a * in->data[(row - 1) * width + col]
                + b * in->data[row * width + col]
                + c * in->data[(row + 1) * width + col];
        }
  }


  for (int row = 1; row < (height - 1); row++)
    for (int col = 1; col < (width - 1); col++)
      {
        out->data[row * width + col]
            = a * out->data[row * width + (col - 1)]
              + b * out->data[row * width + col]
              + c * out->data[row * width + (col + 1)];
      }
  // In Y

  return out;
}

std::vector<mVector> *
motionVectorSearch (Frame *source, Frame *match, int width, int height)
{
  std::vector<mVector> *motion_vectors
      = new std::vector<mVector> (); // empty list of ints

  float Y_weight = 0.5;
  float Cr_weight = 0.25;
  float Cb_weight = 0.25;

  // Window size is how much on each side of the block we search
  int window_size = 16;
  int block_size = 16;

  // How far from the edge we can go since we don't special case the edges
  int inset = (int)max ((float)window_size, (float)block_size);
  int iter = 0;

  for (int my = inset; my < height - (inset + window_size) + 1;
       my += block_size)
    {
      for (int mx = inset; mx < width - (inset + window_size) + 1;
           mx += block_size)
        {

          float best_match_sad = 1e10;
          int best_match_location[2] = { 0, 0 };

          for (int sy = my - window_size; sy < my + window_size; sy++)
            {
              for (int sx = mx - window_size; sx < mx + window_size; sx++)
                {
                  float current_match_sad = 0;
                  // Do the SAD
                  for (int y = 0; y < block_size; y++)
                    {
                      for (int x = 0; x < block_size; x++)
                        {
                          int match_x = mx + x;
                          int match_y = my + y;
                          int search_x = sx + x;
                          int search_y = sy + y;
                          float diff_Y = abs (
                              match->Y->data[match_x * width + match_y]
                              - source->Y->data[search_x * width + search_y]);
                          float diff_Cb = abs (
                              match->Cb->data[match_x * width + match_y]
                              - source->Cb->data[search_x * width + search_y]);
                          float diff_Cr = abs (
                              match->Cr->data[match_x * width + match_y]
                              - source->Cr->data[search_x * width + search_y]);

                          float diff_total = Y_weight * diff_Y
                                             + Cb_weight * diff_Cb
                                             + Cr_weight * diff_Cr;
                          current_match_sad = current_match_sad + diff_total;
                        }
                    } // end SAD

                  if (current_match_sad <= best_match_sad)
                    {
                      best_match_sad = current_match_sad;
                      best_match_location[0] = sx - mx;
                      best_match_location[1] = sy - my;
                    }
                }
            }

          mVector v;
          v.a = best_match_location[0];
          v.b = best_match_location[1];
          motion_vectors->push_back (v);
        }
    }

  return motion_vectors;
}

Frame *
computeDelta (Frame *i_frame_ycbcr, Frame *p_frame_ycbcr,
              std::vector<mVector> *motion_vectors)
{
  Frame *delta = new Frame (p_frame_ycbcr);

  int width = i_frame_ycbcr->width;
  int height = i_frame_ycbcr->height;
  int window_size = 16;
  int block_size = 16;
  // How far from the edge we can go since we don't special case the edges
  int inset = (int)max ((float)window_size, (float)block_size);

  int current_block = 0;
  for (int my = inset; my < width - (inset + window_size) + 1;
       my += block_size)
    {
      for (int mx = inset; mx < height - (inset + window_size) + 1;
           mx += block_size)
        {
          int vector[2];
          vector[0] = (int)motion_vectors->at (current_block).a;
          vector[1] = (int)motion_vectors->at (current_block).b;

          // copy the block
          for (int y = 0; y < block_size; y++)
            {
              for (int x = 0; x < block_size; x++)
                {

                  int src_x = mx + vector[0] + x;
                  int src_y = my + vector[1] + y;
                  int dst_x = mx + x;
                  int dst_y = my + y;
                  delta->Y->data[dst_x * width + dst_y]
                      = delta->Y->data[dst_x * width + dst_y]
                        - i_frame_ycbcr->Y->data[src_x * width + src_y];
                  delta->Cb->data[dst_x * width + dst_y]
                      = delta->Cb->data[dst_x * width + dst_y]
                        - i_frame_ycbcr->Cb->data[src_x * width + src_y];
                  delta->Cr->data[dst_x * width + dst_y]
                      = delta->Cr->data[dst_x * width + dst_y]
                        - i_frame_ycbcr->Cr->data[src_x * width + src_y];
                }
            }

          current_block = current_block + 1;
        }
    }
  return delta;
}

Channel *
downSample (Channel *in)
{
  int width = in->width;
  int height = in->height;
  int w2 = width / 2;
  int h2 = height / 2;

  Channel *out = new Channel ((width / 2), (height / 2));

  for (int y2 = 0, y = 0; y2 < w2; y2++)
    {
      for (int x2 = 0, x = 0; x2 < h2; x2++)
        {

          out->data[x2 * w2 + y2] = in->data[x * width + y];
          x += 2;
        }
      y += 2;
    }

  return out;
}

void
dct8x8 (Channel *in, Channel *out)
{
  int width = in->width;
  int height = in->height;

  // 8x8 block dct on each block
  for (int i = 0; i < width * height; i++)
    {
      in->data[i] -= 128;
      out->data[i] = 0; // zeros
    }

  for (int y = 0; y < width; y += 8)
    {
      for (int x = 0; x < height; x += 8)
        {
          dct8x8_block (&(in->data[x * width + y]),
                        &(out->data[x * width + y]), width);
        }
    }
}

void
round_block (float *in, float *out, int stride)
{
  float quantMatrix[8][8] = {
    { 16, 11, 10, 16, 24, 40, 51, 61 },
    { 12, 12, 14, 19, 26, 58, 60, 55 },
    { 14, 13, 16, 24, 40, 57, 69, 56 },
    { 14, 17, 22, 29, 51, 87, 80, 62 },
    { 18, 22, 37, 56, 68, 109, 103, 77 },
    { 24, 35, 55, 64, 81, 104, 113, 92 },
    { 49, 64, 78, 87, 103, 121, 120, 101 },
    { 72, 92, 95, 98, 112, 100, 103, 99 },
  };

  for (int y = 0; y < 8; y++)
    {
      for (int x = 0; x < 8; x++)
        {
          quantMatrix[x][y] = ceil (quantMatrix[x][y] / QUALITY);
          out[x * stride + y]
              = (float)round (in[x * stride + y] / quantMatrix[x][y]);
        }
    }
}

void
quant8x8 (Channel *in, Channel *out)
{
  int width = in->width;
  int height = in->height;

  for (int i = 0; i < width * height; i++)
    {
      out->data[i] = 0; // zeros
    }

  for (int y = 0; y < width; y += 8)
    {
      for (int x = 0; x < height; x += 8)
        {
          round_block (&(in->data[x * width + y]), &(out->data[x * width + y]),
                       width);
        }
    }
}

void
dcDiff (Channel *in, Channel *out)
{
  int width = in->width;
  int height = in->height;

  int number_of_dc = width * height / 64;
  double *dc_values_transposed = new double[number_of_dc];
  double *dc_values = new double[number_of_dc];

  int iter = 0;
  for (int j = 0; j < width; j += 8)
    {
      for (int i = 0; i < height; i += 8)
        {
          dc_values_transposed[iter] = in->data[i * width + j];
          dc_values[iter] = in->data[i * width + j];
          iter++;
        }
    }

  int new_w = std::max (width / 8, 1);
  int new_h = std::max (height / 8, 1);

  out->data[0] = (float)dc_values[0];

  double prev = 0.;
  iter = 0;
  for (int j = 0; j < new_w; j++)
    {
      for (int i = 0; i < new_h; i++)
        {
          out->data[iter] = (float)(dc_values[i * new_w + j] - prev);
          prev = dc_values[i * new_w + j];
          iter++;
        }
    }
  delete[] dc_values_transposed;
  delete[] dc_values;
}

void
cpyBlock (float *in, float *out, int blocksize, int stride)
{
  for (int j = 0; j < blocksize; j++)
    {
      for (int i = 0; i < blocksize; i++)
        {
          out[i * blocksize + j] = in[i * stride + j];
        }
    }
}

void
zigZagOrder (Channel *in, Channel *ordered)
{
  int width = in->width;
  int height = in->height;
  int zigZagIndex[64]
      = { 0,  1,  8,  16, 9,  2,  3,  10, 17, 24, 32, 25, 18, 11, 4,  5,
          12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,  7,  14, 21, 28,
          35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51,
          58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63 };

  int blockNumber = 0;
  float _block[MPEG_CONSTANT];

  for (int x = 0; x < height; x += 8)
    {
      for (int y = 0; y < width; y += 8)
        {
          cpyBlock (&(in->data[x * width + y]), _block, 8,
                    width); // block = in(x:x+7,y:y+7);
          // Put the coefficients in zig-zag order
          float zigZagOrdered[MPEG_CONSTANT] = { 0 };
          for (int index = 0; index < MPEG_CONSTANT; index++)
            {
              zigZagOrdered[index] = _block[zigZagIndex[index]];
            }
          for (int i = 0; i < MPEG_CONSTANT; i++)
            ordered->data[blockNumber * MPEG_CONSTANT + i] = zigZagOrdered[i];
          blockNumber++;
        }
    }
}

void
encode8x8 (Channel *ordered, SMatrix *encoded)
{
  int width = encoded->height;
  int height = encoded->width;
  int num_blocks = height;

  for (int i = 0; i < num_blocks; i++)
    {
      std::string block_encode[MPEG_CONSTANT];
      for (int j = 0; j < MPEG_CONSTANT; j++)
        {
          block_encode[j] = "\0"; // necessary to initialize every string
                                  // position to empty string
        }

      double *block = new double[width];
      for (int y = 0; y < width; y++)
        block[y] = ordered->data[i * width + y];
      int num_coeff = MPEG_CONSTANT; // width
      int encoded_index = 0;
      int in_zero_run = 0;
      int zero_count = 0;

      // Skip DC coefficient
      for (int c = 1; c < num_coeff; c++)
        {
          double coeff = block[c];
          if (coeff == 0)
            {
              if (in_zero_run == 0)
                {
                  zero_count = 0;
                  in_zero_run = 1;
                }
              zero_count = zero_count + 1;
            }
          else
            {
              if (in_zero_run == 1)
                {
                  in_zero_run = 0;
                  block_encode[encoded_index]
                      = "Z" + std::to_string (zero_count);
                  encoded_index = encoded_index + 1;
                }
              block_encode[encoded_index] = std::to_string ((int)coeff);
              encoded_index = encoded_index + 1;
            }
        }

      // If we were in a zero run at the end attach it as well.
      if (in_zero_run == 1)
        {
          if (zero_count > 1)
            {
              block_encode[encoded_index] = "Z" + std::to_string (zero_count);
            }
          else
            {
              block_encode[encoded_index] = "0";
            }
        }

      for (int it = 0; it < MPEG_CONSTANT; it++)
        {
          if (block_encode[it].length () > 0)
            encoded->data[i * width + it] = new std::string (block_encode[it]);
          else
            it = MPEG_CONSTANT;
        }
      delete[] block;
    }
}

int
encode ()
{
  int end_frame = int (N_FRAMES);
  int i_frame_frequency = int (I_FRAME_FREQ);
  struct timeval starttime, endtime;
  double runtime[10] = { 0 };

  // Hardcoded paths
  string image_path
      = "../../inputs/" + string (image_name) + "/" + image_name + ".";
  string stream_path
      = "../../outputs/stream_c_" + string (image_name) + ".xml";

  xmlDocPtr stream = NULL;

  Image *frame_rgb = NULL;
  Image *previous_frame_rgb = NULL;
  Frame *previous_frame_lowpassed = NULL;

  loadImage (0, image_path, &frame_rgb);

  int width = frame_rgb->width;
  int height = frame_rgb->height;
  int npixels = width * height;

  delete frame_rgb;

  createStatsFile ();
  stream = create_xml_stream (width, height, QUALITY, WINDOW_SIZE, BLOCK_SIZE);
  vector<mVector> *motion_vectors = NULL;

  for (int frame_number = 0; frame_number < end_frame; frame_number++)
    {
      frame_rgb = NULL;
      loadImage (frame_number, image_path, &frame_rgb);

      //  Convert to YCbCr
      print ("Covert to YCbCr...");

      Image *frame_ycbcr = new Image (width, height, FULLSIZE);
      gettimeofday (&starttime, NULL);
      convertRGBtoYCbCr (frame_rgb, frame_ycbcr);
      gettimeofday (&endtime, NULL);
      runtime[0] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_image (frame_ycbcr, "frame_ycbcr", frame_number);
      delete frame_rgb;

      // We low pass filter Cb and Cr channesl
      print ("Low pass filter...");

      gettimeofday (&starttime, NULL);
      Channel *frame_blur_cb = new Channel (width, height);
      Channel *frame_blur_cr = new Channel (width, height);
      Frame *frame_lowpassed = new Frame (width, height, FULLSIZE);

      lowPass (frame_ycbcr->gc, frame_blur_cb);
      lowPass (frame_ycbcr->bc, frame_blur_cr);

      frame_lowpassed->Y->copy (frame_ycbcr->rc);
      frame_lowpassed->Cb->copy (frame_blur_cb);
      frame_lowpassed->Cr->copy (frame_blur_cr);
      gettimeofday (&endtime, NULL);
      runtime[1] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_frame (frame_lowpassed, "frame_ycbcr_lowpass", frame_number);
      delete frame_ycbcr;
      delete frame_blur_cb;
      delete frame_blur_cr;

      Frame *frame_lowpassed_final = NULL;

      if (frame_number % i_frame_frequency != 0)
        {
          // We have a P frame
          // Note that in the first iteration we don't enter this branch!

          // Compute the motion vectors
          print ("Motion Vector Search...");

          gettimeofday (&starttime, NULL);
          motion_vectors = motionVectorSearch (
              previous_frame_lowpassed, frame_lowpassed,
              frame_lowpassed->width, frame_lowpassed->height);
          gettimeofday (&endtime, NULL);
          runtime[2] = double (endtime.tv_sec) * 1000.0f
                       + double (endtime.tv_usec) / 1000.0f
                       - double (starttime.tv_sec) * 1000.0f
                       - double (starttime.tv_usec) / 1000.0f; // in ms

          print ("Compute Delta...");
          gettimeofday (&starttime, NULL);
          frame_lowpassed_final = computeDelta (
              previous_frame_lowpassed, frame_lowpassed, motion_vectors);
          gettimeofday (&endtime, NULL);
          runtime[3] = double (endtime.tv_sec) * 1000.0f
                       + double (endtime.tv_usec) / 1000.0f
                       - double (starttime.tv_sec) * 1000.0f
                       - double (starttime.tv_usec) / 1000.0f; // in ms
        }
      else
        {
          // We have a I frame
          motion_vectors = NULL;
          frame_lowpassed_final = new Frame (frame_lowpassed);
        }
      delete frame_lowpassed;
      frame_lowpassed = NULL;

      if (frame_number > 0)
        delete previous_frame_lowpassed;
      previous_frame_lowpassed = new Frame (frame_lowpassed_final);

      // Downsample the difference
      print ("Downsample...");

      gettimeofday (&starttime, NULL);
      Frame *frame_downsampled = new Frame (width, height, DOWNSAMPLE);

      // We don't touch the Y frame
      frame_downsampled->Y->copy (frame_lowpassed_final->Y);
      Channel *frame_downsampled_cb = downSample (frame_lowpassed_final->Cb);
      frame_downsampled->Cb->copy (frame_downsampled_cb);
      Channel *frame_downsampled_cr = downSample (frame_lowpassed_final->Cr);
      frame_downsampled->Cr->copy (frame_downsampled_cr);
      gettimeofday (&endtime, NULL);
      runtime[4] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_frame (frame_downsampled, "frame_downsampled", frame_number);
      delete frame_lowpassed_final;
      delete frame_downsampled_cb;
      delete frame_downsampled_cr;

      // Convert to frequency domain
      print ("Convert to frequency domain...");

      gettimeofday (&starttime, NULL);
      Frame *frame_dct = new Frame (width, height, DOWNSAMPLE);

      dct8x8 (frame_downsampled->Y, frame_dct->Y);
      dct8x8 (frame_downsampled->Cb, frame_dct->Cb);
      dct8x8 (frame_downsampled->Cr, frame_dct->Cr);
      gettimeofday (&endtime, NULL);
      runtime[5] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_frame (frame_dct, "frame_dct", frame_number);
      delete frame_downsampled;

      // Quantize the data
      print ("Quantize...");

      gettimeofday (&starttime, NULL);
      Frame *frame_quant = new Frame (width, height, DOWNSAMPLE);

      quant8x8 (frame_dct->Y, frame_quant->Y);
      quant8x8 (frame_dct->Cb, frame_quant->Cb);
      quant8x8 (frame_dct->Cr, frame_quant->Cr);
      gettimeofday (&endtime, NULL);
      runtime[6] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_frame (frame_quant, "frame_quant", frame_number);
      delete frame_dct;

      // Extract the DC components and compute the differences
      print ("Compute DC differences...");

      gettimeofday (&starttime, NULL);
      Frame *frame_dc_diff = new Frame (1, (width / 8) * (height / 8),
                                        DCDIFF); // dealocate later

      dcDiff (frame_quant->Y, frame_dc_diff->Y);
      dcDiff (frame_quant->Cb, frame_dc_diff->Cb);
      dcDiff (frame_quant->Cr, frame_dc_diff->Cr);
      gettimeofday (&endtime, NULL);
      runtime[7] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_dc_diff (frame_dc_diff, "frame_dc_diff", frame_number);

      // Zig-zag order for zero-counting
      print ("Zig-zag order...");
      gettimeofday (&starttime, NULL);

      Frame *frame_zigzag
          = new Frame (MPEG_CONSTANT, width * height / MPEG_CONSTANT, ZIGZAG);

      zigZagOrder (frame_quant->Y, frame_zigzag->Y);
      zigZagOrder (frame_quant->Cb, frame_zigzag->Cb);
      zigZagOrder (frame_quant->Cr, frame_zigzag->Cr);
      gettimeofday (&endtime, NULL);
      runtime[8] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      dump_zigzag (frame_zigzag, "frame_zigzag", frame_number);
      delete frame_quant;

      // Encode coefficients
      print ("Encode coefficients...");

      gettimeofday (&starttime, NULL);
      FrameEncode *frame_encode
          = new FrameEncode (width, height, MPEG_CONSTANT);

      encode8x8 (frame_zigzag->Y, frame_encode->Y);
      encode8x8 (frame_zigzag->Cb, frame_encode->Cb);
      encode8x8 (frame_zigzag->Cr, frame_encode->Cr);
      gettimeofday (&endtime, NULL);
      runtime[9] = double (endtime.tv_sec) * 1000.0f
                   + double (endtime.tv_usec) / 1000.0f
                   - double (starttime.tv_sec) * 1000.0f
                   - double (starttime.tv_usec) / 1000.0f; // in ms

      delete frame_zigzag;

      stream_frame (stream, frame_number, motion_vectors, frame_number - 1,
                    frame_dc_diff, frame_encode);
      write_stream (stream_path, stream);

      delete frame_dc_diff;
      delete frame_encode;

      if (motion_vectors != NULL)
        {
          free (motion_vectors);
          motion_vectors = NULL;
        }

      writestats (frame_number, frame_number % i_frame_frequency, runtime);
    }

  closeStats ();
  /* Uncoment to prevent visual studio output window from closing */
  // system("pause");

  return 0;
}

int
main (int argc, char *argv[])
{
  args = parseArgs (argc, argv);
  encode ();
  return 0;
}
