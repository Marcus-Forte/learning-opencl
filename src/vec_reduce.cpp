
#include "vec_reduce.h"

#include <filesystem>
#include <iostream>

#include "programFactory.h"
#include "utils.h"

float vec_reduce_host(const float *A, int N_ELEMENTS);

void vec_reduce(const cl::Context &context, const cl::Device &selected_device,
                int N_ELEMENTS) {
  programFactory factory(context);

  auto cwd = std::filesystem::current_path();
  auto program = factory.fromSource(cwd / "reduce.cl");

  auto build_result = program.build("-I .");

  if (build_result != CL_SUCCESS) {
    std::cerr << "Build failure:\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device)
              << std::endl;
    exit(-1);
  }

  cl::CommandQueue queue(context, cl::QueueProperties::Profiling);

  std::cout << "Computing reduction of: " << N_ELEMENTS << " elements.\n";

  std::cout << "Allocating data...\n";

  float *h_a = new float[N_ELEMENTS];
  float *h_c_host_calc = new float[N_ELEMENTS];

  generate_random_data(h_a, N_ELEMENTS, 10);

  cl::Buffer d_a(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
  cl::Buffer d_c(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));

  CheckResult(queue.enqueueWriteBuffer(d_a, CL_TRUE, 0,
                                       sizeof(float) * N_ELEMENTS, h_a));

  cl::Kernel kernel(program, "vecsum_reduce");

  // MAX Nvidia work group size: 1024
  // MAX int
  const int _local_size = 128;
  const int _num_wg = N_ELEMENTS / _local_size;

  CheckResult(kernel.setArg(0, d_a));
  CheckResult(kernel.setArg(1, cl::Local(sizeof(float) * 1024)));
  CheckResult(kernel.setArg(2, d_c));
  // std::cout << "Num of work groups: " << _num_wg << std::endl;
  cl::Event event;

  // If local is unsed, implementation decides
  CheckResult(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                         cl::NDRange(N_ELEMENTS), cl::NullRange,
                                         NULL, &event));
  event.wait();
  auto event_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto event_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  std::cout << "Normal GPU Execution took: "
            << (double)(event_end - event_start) * 1e-9 << " s\n";

  CheckResult(queue.enqueueReadBuffer(
      d_c, CL_TRUE, 0, N_ELEMENTS * sizeof(float), h_c_host_calc));
  std::cout << "leftovers(h_c_host_calc[0]): " << h_c_host_calc[0] << std::endl;
  queue.finish();

  float total_gpu = 0.0f;
  for (int i = 0; i < h_c_host_calc[0]; ++i) {
    total_gpu += h_c_host_calc[1 + i];
    // std::cout << h_c_host_calc[1 + i] << std::endl;
  }

  auto start = std::chrono::high_resolution_clock::now();
  float total_host = vec_reduce_host(h_a, N_ELEMENTS);
  auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() *
               1e-9;
  std::cout << "Normal CPU Execution took: " << delta << " s\n";

  std::cout << "GPU total sum: " << total_gpu << std::endl;
  std::cout << "CPU total sum: " << total_host << std::endl;

  // Timings...
  // 200M GPU -> 0.014 s
  // 200M CPU ->
}

float vec_reduce_host(const float *A, int N_ELEMENTS) {
  float sum = 0.0f;
  for (int i = 0; i < N_ELEMENTS; ++i) {
    sum += A[i];
  }
  return sum;
}
