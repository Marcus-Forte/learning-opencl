

kernel void vecsum_reduce(global const float *A, local float *A_local,
                          global float *result) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  size_t gsize = get_global_size(0);
  size_t lsize = get_local_size(0);
  int wgid = get_group_id(0);

  if (gid == 0) {
    printf("local size: %d\n", lsize);
    printf("global size: %d\n", gsize);
    // printf("local id: %d\n", gid);
    // printf("global id: %d\n", lid);
    printf("num wgs: %d\n", get_num_groups(0));
    printf("wg id: %d\n", wgid);
    result[gid] =
        get_num_groups(0);  // convention : number of elements to be summed up
  }
  // fetch from global memory to local memory
  A_local[lid] = A[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  /* INTERLEAVED ADDRESSING */
  // for (unsigned int stride = 1; stride < lsize; stride *= 2) {
  //   if (lid % (2 * stride) == 0) {
  //     A_local[lid] += A_local[lid + stride];
  //   }
  //   barrier(CLK_LOCAL_MEM_FENCE);
  // }

  /* INTERLEAVED ADDRESSING NON-DIVERGENT */
  for (unsigned int stride = 1; stride < lsize; stride *= 2) {
    int index = 2 * stride * lid;
    if (index < lsize) {
      A_local[index] += A_local[index + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  /* SEQUENTIAL ADDRESSING */
  // for (unsigned int stride = lsize / 2; stride > 0; stride >>= 1) {
  //   if (lid < stride) {
  //     A_local[lid] += A_local[lid + stride];
  //   }
  //   barrier(CLK_LOCAL_MEM_FENCE);
  // }

  if (lid == 0) {
    result[1 + wgid] = A_local[0];
  }
}
