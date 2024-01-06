cmake_minimum_required(VERSION 3.10)
project(opencl-hello)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
find_package(OpenCL REQUIRED)

add_executable(benchmark 
src/benchmark.cpp
src/programFactory.cpp
src/vec_sum.cpp
src/vec_reduce.cpp
src/model.cpp
)

target_link_libraries(benchmark OpenCL::OpenCL)

target_include_directories(benchmark PUBLIC include)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/vec_sum.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/reduce.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/kernel/model.cl ${CMAKE_CURRENT_BINARY_DIR} COPYONLY)


## Offline compiler
add_executable(offline_compiler
src/offline_compiler.cpp
src/programFactory.cpp)
target_link_libraries(offline_compiler OpenCL::OpenCL)
target_include_directories(offline_compiler PUBLIC include)