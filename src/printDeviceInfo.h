#pragma once

#include <CL/opencl.hpp>
#include <iostream>

/// @brief Prints some useful information about the selected device.
/// @param device
void printDeviceInfo(const cl::Device& device) {
  std::cout << "CL_DEVICE_NAME: " << device.getInfo<CL_DEVICE_NAME>()
            << std::endl;
  std::cout << "CL_DEVICE_GLOBAL_MEM_SIZE: "
            << device.getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>() / 1024 << " KB"
            << std::endl;
  std::cout << "CL_DEVICE_LOCAL_MEM_SIZE: "
            << device.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>() / 1024 << " KB"
            << std::endl;
  std::cout << "CL_DEVICE_MAX_COMPUTE_UNITS: "
            << device.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << std::endl;
  std::cout << "CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS: "
            << device.getInfo<CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS>()
            << std::endl;
  std::cout << "CL_DEVICE_MAX_WORK_GROUP_SIZE: "
            << device.getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << std::endl;
  std::cout << "CL_DEVICE_MAX_WORK_ITEM_SIZES: " << std::endl;
  [&device] {
    auto work_item_sizes = device.getInfo<CL_DEVICE_MAX_WORK_ITEM_SIZES>();
    std::cout << "  - work_item_sizes: " << work_item_sizes.size() << "\n    ";
    std::for_each(work_item_sizes.cbegin(), work_item_sizes.cend(),
                  [](const auto& element) { std::cout << element << " "; });
    std::cout << std::endl;
  }();
  std::cout << "CL_DEVICE_MAX_CLOCK_FREQUENCY: "
            << device.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << std::endl;
}