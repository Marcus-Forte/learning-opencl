#include <tbb/parallel_for.h>

#include <filesystem>
#include <iostream>

#include "programFactory.h"
#include "utils.h"

void vec_sum_host(const float *a, const float *b, float *c, int N_ELEMENTS);

void vec_sum(const cl::Context &context, const cl::Device &selected_device,
             int N_ELEMENTS) {
  programFactory factory(context);

  auto cwd = std::filesystem::current_path();
  auto program = factory.fromSource(cwd / "kernels.cl");

  auto build_result = program.build("-I .");

  if (build_result != CL_SUCCESS) {
    std::cerr << "Build failure:\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device)
              << std::endl;
    exit(-1);
  }

  cl::CommandQueue queue(context, cl::QueueProperties::Profiling);

  std::cout << "Computing sum of: " << N_ELEMENTS << " elements.\n";

  std::cout << "Allocating data...\n";

  float *h_a = new float[N_ELEMENTS];
  float *h_b = new float[N_ELEMENTS];
  float *h_c = new float[N_ELEMENTS];
  float *h_c_host_calc = new float[N_ELEMENTS];

  std::cout << "Allocated data bytes: "
            << N_ELEMENTS * sizeof(float) * 4 / (1024 * 1024) << " MB\n";

  std::cout << "Generating values...\n";

  generate_random_data(h_a, N_ELEMENTS);
  generate_random_data(h_b, N_ELEMENTS);

  std::cout << "Allocating GPU data...\n";

  cl::Buffer d_a(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
  cl::Buffer d_b(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));
  cl::Buffer d_c(context, CL_MEM_READ_WRITE, N_ELEMENTS * sizeof(float));

  CheckResult(queue.enqueueWriteBuffer(d_a, CL_TRUE, 0,
                                       sizeof(float) * N_ELEMENTS, h_a));
  CheckResult(queue.enqueueWriteBuffer(d_b, CL_TRUE, 0,
                                       sizeof(float) * N_ELEMENTS, h_b));

  cl::Kernel kernel(program, "vecsum");

  CheckResult(kernel.setArg(0, d_a));
  CheckResult(kernel.setArg(1, d_b));
  CheckResult(kernel.setArg(2, d_c));

  cl::Kernel single_kernel(program, "vecsum_single_kernel");
  CheckResult(single_kernel.setArg(0, d_a));
  CheckResult(single_kernel.setArg(1, d_b));
  CheckResult(single_kernel.setArg(2, d_c));

  std::cout << "Executing queue...\n";
  cl::Event event;
  CheckResult(queue.enqueueNDRangeKernel(kernel, cl::NullRange,
                                         cl::NDRange(N_ELEMENTS), cl::NullRange,
                                         NULL, &event));
  event.wait();
  auto event_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  auto event_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();

  std::cout << "Normal GPU Execution took: "
            << (double)(event_end - event_start) * 1e-9 << " s\n";

  CheckResult(queue.enqueueNDRangeKernel(single_kernel, cl::NullRange,
                                         cl::NDRange(N_ELEMENTS), cl::NullRange,
                                         NULL, &event));
  event.wait();
  event_start = event.getProfilingInfo<CL_PROFILING_COMMAND_START>();
  event_end = event.getProfilingInfo<CL_PROFILING_COMMAND_END>();
  std::cout << "Single Kernel GPU Execution took: "
            << (double)(event_end - event_start) * 1e-9 << " s\n";

  CheckResult(queue.enqueueReadBuffer(d_c, CL_TRUE, 0,
                                      sizeof(float) * N_ELEMENTS, h_c));

  // CheckResult(queue.finish());
  // event.wait();

  std::cout << "done.\n";

  std::cout << "Executing on host...\n";
  auto start = std::chrono::high_resolution_clock::now();
  vec_sum_host(h_a, h_b, h_c_host_calc, N_ELEMENTS);
  auto delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count();

  std::cout << "Host execution (sequential) took: " << (double)delta * 1e-9
            << std::endl;

  start = std::chrono::high_resolution_clock::now();
  auto tbb_range = tbb::blocked_range<int>(0, N_ELEMENTS);
  tbb::parallel_for(tbb_range, [&](tbb::blocked_range<int> r) {
    for (int i = r.begin(); i != r.end(); ++i) {
      h_c_host_calc[i] = h_a[i] + h_b[i];
    }
  });
  delta = std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now() - start)
              .count();
  std::cout << "Host execution (TBB) took: " << (double)delta * 1e-9
            << std::endl;

  std::cout << "checking results...\n";
  for (int i = 0; i < N_ELEMENTS; ++i) {
    if (h_c[i] != h_c_host_calc[i]) {
      std::cerr << "Wrong results: " << std::endl;
      std::cerr << h_c[i] << " != " << h_a[i] << " * " << h_b[i] << std::endl;
      break;
    }
  }
  std::cout << "done.\n";
}

void vec_sum_host(const float *a, const float *b, float *c, int N_ELEMENTS) {
  for (int i = 0; i < N_ELEMENTS; ++i) {
    c[i] = a[i] + b[i];
  }
}