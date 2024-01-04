#pragma once

#define CheckResult(x)                                                 \
  if (x != CL_SUCCESS) {                                               \
    printf("Error in %s:%d\nError code: %d\n", __FILE__, __LINE__, x); \
    exit(-1);                                                          \
  }

inline void generate_random_data(float *data, size_t length,
                                 float max_val = 10.0) {
  for (int i = 0; i < length; ++i) {
    data[i] = ((float)std::rand() * max_val / (float)RAND_MAX);
    if (length < 10) printf("data[%d] = %f\n", i, data[i]);
  }
}