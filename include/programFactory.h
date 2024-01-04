#pragma once

#include "CL/opencl.hpp"

/// @brief creates openCL programs
class programFactory {
 public:
  programFactory(const cl::Context& context) : context_(context) {}
  virtual ~programFactory() = default;

  /// @brief
  /// @param source_path
  /// @return CL program;
  cl::Program fromSource(const std::string& source_path) const;

 private:
  const cl::Context& context_;
};