// runs a model
void print_data(global const float *data, const unsigned int elements) {
  int gid = get_global_id(0);
  if (gid < elements) printf("data[%d] = %f\n", gid, data[gid]);
}

kernel void model(global const float *x0, global const float *data_x,
                  global const float *data_y, global float *residual) {
  int gid = get_global_id(0);

  if (gid == 0) {
    printf("local size: %d\n", get_local_size(0));
    printf("global size: %d\n", get_global_size(0));
    // printf("local id: %d\n", gid);
    // printf("global id: %d\n", lid);
    printf("num wgs: %d\n", get_num_groups(0));
  }

  residual[gid] = data_y[gid] - (x0[0] * data_x[gid]) / (x0[1] + data_x[gid]);

  //   print_data(residual, get_global_size(0));
}

kernel void model_reduce(global const float *residuals,
                         local float *local_residuals, global float *result) {
  int gid = get_global_id(0);
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  local_residuals[lid] = residuals[gid];
  barrier(CLK_LOCAL_MEM_FENCE);

  for (unsigned int stride = 1; stride < lsize; stride *= 2) {
    int index = 2 * stride * lid;
    if (index < lsize) {
      local_residuals[index] += local_residuals[index + stride];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  // ONLY WORK FOR SINGLE WORK GROUP..
  if (lid == 0) result[0] = local_residuals[lid];
}
