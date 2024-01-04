#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <chrono>
#include <filesystem>
#include <iostream>

#include "model.h"
#include "printDeviceInfo.h"
#include "utils.h"
#include "vec_reduce.h"
#include "vec_sum.h"

// 200000000 //  ~ (800 / array ) * 3 arrays = 2400 MB
#define N_ELEMENTS 100000  //  ~ (800 / array ) * 3 arrays = 2400 MB

void printUsage() { std::cout << "./main [platform] [device] \n"; }

int main(int argc, char **argv) {
  if (argc < 3) {
    printUsage();
    exit(0);
  }

  const int selected_platform_id = atoi(argv[1]);
  const int selected_device_id = atoi(argv[2]);

  cl::vector<cl::Platform> platforms;
  CheckResult(cl::Platform::get(&platforms));

  std::cout << "Num platforms: " << platforms.size() << std::endl;

  if (selected_platform_id >= platforms.size())
    throw std::runtime_error("Invalid selected platform.\n");

  const auto &selected_platform = platforms[selected_platform_id];
  auto platform_name = selected_platform.getInfo<CL_PLATFORM_NAME>();

  std::cout << "Selected platform: " << platform_name << std::endl;

  cl::vector<cl::Device> devices;

  selected_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);

  std::cout << "Num devices: " << devices.size() << std::endl;

  if (selected_device_id >= devices.size())
    throw std::runtime_error("Invalid selected device.\n");

  const auto &selected_device = devices[selected_device_id];

  printDeviceInfo(selected_device);

  cl::Context context(selected_device);

  // std::cout << "#------ VEC SUM START ------#\n";
  // vec_sum(context, selected_device, N_ELEMENTS);
  // std::cout << "#------ VEC SUM END ------#\n";

  // std::cout << "#------ VEC REDUCE START ------#\n";
  // vec_reduce(context, selected_device, N_ELEMENTS);
  // std::cout << "#------ VEC REDUCE END ------#\n";

  std::cout << "#------ MODEL COST START ------#\n";
  model(context, selected_device);
  std::cout << "#------ MODEL COST END ------#\n";
}
