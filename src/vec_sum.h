#pragma once

#include <CL/opencl.hpp>

void vec_sum(const cl::Context& context, const cl::Device& selected_device,
             int N_ELEMENTS);