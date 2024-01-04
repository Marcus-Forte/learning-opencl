#pragma once

#include <CL/opencl.hpp>

void vec_reduce(const cl::Context& context, const cl::Device& selected_device,
                int N_ELEMENTS);