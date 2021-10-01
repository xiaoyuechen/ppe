kernel void
convert (global float *R, global float *G, global float *B)
{
  // Get the index of the current element to be processed
  int i = get_global_id(0);

  float r = 0.299f * R[i] + 0.587f * G[i] + 0.113f * B[i];
  float g = 128 - 0.168736f * R[i] - 0.331264f * G[i] + 0.5f * B[i];
  float b = 128 + 0.5f * R[i] - 0.418688f * G[i] - 0.081312f * B[i];
  R[i] = r;
  G[i] = g;
  B[i] = b;
}
