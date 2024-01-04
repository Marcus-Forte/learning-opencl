cmake_minimum_required(VERSION 3.10)
project(opencl-hello)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCL REQUIRED)
find_package(TBB REQUIRED)

add_executable(benchmark 
src/benchmark.cpp
src/programFactory.cpp
src/vec_sum.cpp
src/vec_reduce.cpp
src/model.cpp
)
    if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 8.0)
    target_link_libraries(benchmark OpenCL::OpenCL TBB::tbb)
else()
    message(WARNING "Old GCC version detected.")
    target_link_libraries(benchmark OpenCL::OpenCL pthread stdc++fs)
endif()


target_include_directories(benchmark PUBLIC include)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/vec_sum.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/reduce.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/model.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)