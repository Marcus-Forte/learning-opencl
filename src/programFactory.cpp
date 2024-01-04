#include "programFactory.h"

#include <filesystem>
#include <fstream>

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