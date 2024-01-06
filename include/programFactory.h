#pragma once
#define CL_HPP_TARGET_OPENCL_VERSION 300
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
  static cl::Program fromSource(const cl::Context& context,
                                const std::string& source_path);

  cl::Program fromBinarySource(const cl::Device& device,
                               const std::string& binary_path) const;
  static cl::Program fromBinarySource(const cl::Context& context,
                                      const cl::Device& device,
                                      const std::string& binary_path);

 private:
  const cl::Context& context_;
};