kernel void
convertRGBtoYCbCr (float *in, float *out)
{
  int width = get_global_size (0);
  int height = get_global_size (1);
  int row = get_global_id (0);
  int col = get_global_id (1);

  float R = in[row * width + col];
  float G = in[row * width + col];
  float B = in[row * width + col];

  float Y = 0.299f * R + 0.587f * G + 0.113f * B;
  float Cb = 128 - 0.168736f * R - 0.331264f * G + 0.5f * B;
  float Cr = 128 + 0.5f * R - 0.418688f * G - 0.081312f * B;

  out[row * width + col] = Y;
  out[row * width + col] = Cb;
  out[row * width + col] = Cr;
}
