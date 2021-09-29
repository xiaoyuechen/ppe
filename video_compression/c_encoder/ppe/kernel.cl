kernel void
convertRGBtoYCbCr (float *R, float *G, float *B, float *Y, float *Cb, float *Cr)
{
  // Get the index of the current element to be processed
  int i = get_global_id(0);

  Y[i] = 0.299f * R[i] + 0.587f * G[i] + 0.113f * B[i];
  Cb[i] = 128 - 0.168736f * R[i] - 0.331264f * G[i] + 0.5f * B[i];
  Cr[i] = 128 + 0.5f * R[i] - 0.418688f * G[i] - 0.081312f * B[i];
}
