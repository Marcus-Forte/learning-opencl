#include "programFactory.h"

#include <filesystem>
#include <fstream>
#include <iostream>
cl::Program programFactory::fromSource(const std::string &source_path) const {
  if (!std::filesystem::exists(source_path))
    throw std::runtime_error("Kernel source file `" + source_path +
                             "` does not exist.\n");

  std::ifstream file(source_path);
  std::string content;

  if (!file.is_open())
    throw std::runtime_error("Unable to open kernel file: " + source_path);

  std::string line;
  while (std::getline(file, line)) {
    content += line + "\n";
  }

  return cl::Program(context_, content);
}

cl::Program programFactory::fromBinarySource(
    const cl::Device &device, const std::string &binary_path) const {
  std::ifstream file(binary_path, std::ios::binary);

  if (!file.is_open())
    throw std::runtime_error("Unable to open kernel file: " + binary_path);

  // std::filesystem::path path(binary_path);

  auto file_size = std::filesystem::file_size(binary_path);
  std::vector<unsigned char> binary;
  binary.reserve(file_size);

  char byte;
  while (file.get(byte)) {
    binary.emplace_back(byte);
  }

  file.close();
  cl::Program::Binaries binaries{binary};

  const cl::vector<cl::Device> devices{device};

  return cl::Program(context_, devices, binaries);
}

cl::Program programFactory::fromSource(const cl::Context &context,

                                       const std::string &source_path) {
  programFactory factory(context);

  return factory.fromSource(source_path);
}

cl::Program programFactory::fromBinarySource(const cl::Context &context,
                                             const cl::Device &device,
                                             const std::string &binary_path) {
  programFactory factory(context);

  return factory.fromBinarySource(device, binary_path);
}