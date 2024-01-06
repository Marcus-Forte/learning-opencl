#define CL_HPP_TARGET_OPENCL_VERSION 300

#include <CL/opencl.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>

#include "programFactory.h"
#include "utils.h"

static void printUsage() {
  std::cout
      << "Usage: ./offline_compier <platform #> <device #> <program>.cl\n";
}

int main(int argc, char **argv) {
  if (argc < 4) {
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

  std::cout << "CL_DEVICE_NAME: " << selected_device.getInfo<CL_DEVICE_NAME>()
            << std::endl;

  const std::string source(argv[3]);

  cl::Context context(selected_device);

  cl::Program program = programFactory::fromSource(context, source);

  auto build_result = program.build();

  if (build_result != CL_SUCCESS) {
    std::cerr << "Build failure:\n";
    std::cerr << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device)
              << std::endl;
    exit(-1);
  }

  auto size = program.getInfo<CL_PROGRAM_BINARY_SIZES>();

  std::cout << "program size: " << size[0] << " bytes\n";
  // writting to file.
  cl::Program::Binaries program_binaries =
      program.getInfo<CL_PROGRAM_BINARIES>();

  std::cout << "# Programs: " << program_binaries.size() << " bytes\n";
  auto out_filename = std::filesystem::path(source).stem().string() + ".clbin";

  std::ofstream program_output_file(out_filename,
                                    std::ios::out | std::ios::binary);

  program_output_file.write((const char *)(program_binaries[0].data()),
                            size[0]);

  std::cout << "compile successful to: " << out_filename << std::endl;

  //  std::cout << program_binaries[0] << std::endl;
}