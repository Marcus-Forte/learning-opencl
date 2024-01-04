
void vecreduce(global const float *A);

kernel void vecsum_single_kernel(global const float *A, global const float *B,
                                 global float *C) {
  if (get_global_id(0) == 0) {
    printf("single GPU thread doing all the work... D:\n");
    printf("N elements: %d\n", get_global_size(0));
    for (int i = 0; i < get_global_size(0); ++i) {
      C[i] = A[i] + B[i];
    }
  }
}
kernel void vecsum(global const float *A, global const float *B,
                   global float *C) {
  int id = get_global_id(0);

  C[id] = A[id] + B[id];
}

kernel void vecmult(global const float *A, global const float *B,
                    global float *C) {
  int id = get_global_id(0);

  C[id] = A[id] * B[id];
}