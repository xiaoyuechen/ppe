kernel void
convert (global float *R, global float *G, global float *B)
{
  int i = get_global_id (0);
  float y = 0.299f * R[i] + 0.587f * G[i] + 0.113f * B[i];
  float cb = 128 - 0.168736f * R[i] - 0.331264f * G[i] + 0.5f * B[i];
  float cr = 128 + 0.5f * R[i] - 0.418688f * G[i] - 0.081312f * B[i];
  R[i] = y;
  G[i] = cb;
  B[i] = cr;
}

#define BLKSIZE 16

kernel void
motionVectorSearch (global const float *sy, global const float *scb,
                    global const float *scr, global const float *my,
                    global const float *mcb, global const float *mcr,
                    global int *out)
{
  if (get_group_id (0) == 0 || get_group_id (0) == get_num_groups (0) - 1
      || get_group_id (1) == 0 || get_group_id (1) == get_num_groups (1) - 1)
    {
      return;
    }

  local float3 search[BLKSIZE * 3][BLKSIZE * 3];
  local float3 match[BLKSIZE][BLKSIZE];
  local float sad[BLKSIZE][BLKSIZE];
  local int2 motion[BLKSIZE][BLKSIZE];

  /* Load match block to local memory */
  {
    size_t global_index
        = get_global_id (0) * get_global_size (0) + get_global_id (1);
    match[get_local_id (0)][get_local_id (1)]
        = (float3)(my[global_index], mcb[global_index], mcr[global_index]);
  }

  /* Load search blocks to local memory */
  {
    size_t local_index = get_local_id (0) * BLKSIZE + get_local_id (1);
    for (size_t i = 0; i < 9; ++i)
      {
        size_t srow = (local_index * 9 + i) / (3 * BLKSIZE);
        size_t scol = (local_index * 9 + i) - srow * 3 * BLKSIZE;
        size_t global_srow = (get_group_id (0) - 1) * BLKSIZE + srow;
        size_t global_scol = (get_group_id (1) - 1) * BLKSIZE + scol;
        size_t global_index = global_srow * get_global_size (0) + global_scol;
        search[srow][scol]
            = (float3)(sy[global_index], scb[global_index], scr[global_index]);
      }
  }

  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  /* Compute sad for different search blocks */
  {
    size_t search_size = BLKSIZE * 2 + 1;
    size_t n_blocks
        = ceil (search_size * search_size
                / (float)(get_local_size (0) * get_local_size (1)));
    float thread_best_match_sad = 1E10;
    int2 thread_best_match_location;
    size_t local_index = get_local_id (0) * BLKSIZE + get_local_id (1);
    for (size_t i = local_index * n_blocks;
         i < (local_index + 1) * n_blocks && i < search_size * search_size;
         ++i)
      {
        float current_match_sad = 0;
        size_t srow0 = i / search_size;
        size_t scol0 = i - srow0 * search_size;
        for (size_t mrow = 0; mrow < BLKSIZE; ++mrow)
          {
            for (size_t mcol = 0; mcol < BLKSIZE; ++mcol)
              {
                size_t srow = srow0 + mrow;
                size_t scol = scol0 + mcol;
                float3 diff = fabs (match[mrow][mcol] - search[srow][scol]);
                current_match_sad
                    += 0.5f * diff.r + 0.25f * diff.g + 0.25f * diff.b;
              }
          }
        if (current_match_sad < thread_best_match_sad)
          {
            thread_best_match_sad = current_match_sad;
            thread_best_match_location
                = (int2)(-BLKSIZE + srow0, -BLKSIZE + scol0);
          }
      }
    sad[get_local_id (0)][get_local_id (1)] = thread_best_match_sad;
    motion[get_local_id (0)][get_local_id (1)] = thread_best_match_location;
  }

  work_group_barrier (CLK_LOCAL_MEM_FENCE);

  if (get_local_id (0) == 0 && get_local_id (1) == 0)
    {
      uint2 group_best_match = (uint2)(0, 0);
      for (size_t i = 0; i < get_local_size (0); ++i)
        {
          for (size_t j = 0; j < get_local_size (1); ++j)
            {
              if (sad[i][j] < sad[group_best_match.x][group_best_match.y])
                {
                  group_best_match = (uint2)(i, j);
                }
            }
        }
      vstore2 (motion[group_best_match.x][group_best_match.y], 0,
               &out[((get_group_id (1) - 1) * (get_num_groups (1) - 2)
                     + get_group_id (0) - 1)
                    * 2]);
    }
}
