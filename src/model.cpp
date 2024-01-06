
#include "test/model.h"

#include <filesystem>
#include <iostream>

#include "programFactory.h"
#include "utils.h"
#include "vec_reduce.h"

void model(const cl::Context &context, const cl::Device &selected_device) {
  programFactory factory(context);

  auto cwd = std::filesystem::current_path();
  auto program = factory.fromSource(cwd / "model.cl");

  auto build_result = program.build("-I .");

  if (build_result != CL_SUCCESS) {
    std::cerr << "Build failure:\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device)
              << std::endl;
    exit(-1);
  }

  cl::CommandQueue queue(context);

  std::cout << "Allocating data...\n";

  float x_data[7] = {0.038, 0.194, 0.425, 0.626, 1.253, 2.5, 3.70};
  float y_data[7] = {0.05, 0.127, 0.094, 0.2122, 0.2729, 0.2665, 0.3317};
  float residuals[7];

  float x0[] = {0.9, 0.2};
  Model<float> model(x_data, y_data);
  float sum = 0.0f;
  for (int i = 0; i < 7; ++i) {
    model.f(x0, &residuals[i], i);
    std::cout << residuals[i] << std::endl;

    sum += residuals[i];
  }

  cl::Buffer d_x0(context, CL_MEM_READ_ONLY, 2 * sizeof(float));
  cl::Buffer d_x(context, CL_MEM_READ_ONLY, 7 * sizeof(float));
  cl::Buffer d_y(context, CL_MEM_READ_ONLY, 7 * sizeof(float));
  cl::Buffer d_residuals(context, CL_MEM_READ_WRITE, 7 * sizeof(float));

  CheckResult(
      queue.enqueueWriteBuffer(d_x0, CL_TRUE, 0, sizeof(float) * 2, x0));
  CheckResult(
      queue.enqueueWriteBuffer(d_x, CL_TRUE, 0, sizeof(float) * 7, x_data));
  CheckResult(
      queue.enqueueWriteBuffer(d_y, CL_TRUE, 0, sizeof(float) * 7, y_data));

  cl::Kernel kernel(program, "model");

  CheckResult(kernel.setArg(0, d_x0));
  CheckResult(kernel.setArg(1, d_x));
  CheckResult(kernel.setArg(2, d_y));
  CheckResult(kernel.setArg(3, d_residuals));

  cl::Kernel reduce_kernel(program, "model_reduce");
  float result;
  cl::Buffer d_result(context, CL_MEM_WRITE_ONLY, 1 * sizeof(float));
  CheckResult(reduce_kernel.setArg(0, d_residuals));
  CheckResult(reduce_kernel.setArg(1, cl::Local(sizeof(float) * 7)));
  CheckResult(reduce_kernel.setArg(2, d_result));

  CheckResult(queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(7),
                                         cl::NullRange, NULL, NULL));
  CheckResult(queue.enqueueNDRangeKernel(
      reduce_kernel, cl::NullRange, cl::NDRange(7), cl::NullRange, NULL, NULL));
  CheckResult(queue.enqueueReadBuffer(
      d_residuals, CL_TRUE, 0, sizeof(float) * 7, residuals, NULL, NULL));
  CheckResult(queue.enqueueReadBuffer(d_result, CL_TRUE, 0, sizeof(float) * 1,
                                      &result, NULL, NULL));
  CheckResult(queue.finish());

  std::cout << "GPU sum result: " << result << std::endl;
  std::cout << "CPU sum result: " << sum << std::endl;
}
