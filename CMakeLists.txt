cmake_minimum_required(VERSION 3.10)

project(GrayScott)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -mavx2 -mfma -funroll-loops -fpeel-loops")


find_package(OpenMP REQUIRED)

add_executable(grayscott src/main.cpp)

target_link_libraries(grayscott PUBLIC OpenMP::OpenMP_CXX)

