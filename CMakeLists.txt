cmake_minimum_required(VERSION 3.10)

project(GrayScott)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenMP REQUIRED)

add_executable(grayscott src/main.cpp)

target_link_libraries(grayscott PUBLIC OpenMP::OpenMP_CXX)

