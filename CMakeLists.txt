cmake_minimum_required(VERSION 3.10)
project(GrayScott)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -mavx2 -mfma -funroll-loops -fpeel-loops")


option(USE_OPENMP "Enable OpenMP Backend" ON)
option(USE_CUDA "Enable CUDA Backend" OFF)

if(USE_OPENMP AND USE_CUDA)
	message(FATAL_ERROR "Cannot enable both OpenMP and CUDA backends")
endif()

if(USE_OPENMP)
	find_package(OpenMP REQUIRED)
	message(STATUS "OpenMP found ! --> ${OpenMP_CXX_VERSION}")
endif()
if(USE_CUDA)
	find_package(CUDA REQUIRED)
	message(STATUS "Cuda found ! --> ${CUDA_VERSION}")
	set(CUDA_BACKEND_SRC
		src/kernel.cu
	)
	cuda_add_library(cuda_backend ${CUDA_BECKEND_SRC})
	target_link_libraries(grayscott cuda_backend)
endif()


set(SRC
        src/main.cpp
)

if(USE_CUDA)
	list(APPEND SRC ${CUDA_BACKEND_SRC})
endif()


add_executable(grayscott ${SRC})

if(USE_OPENMP)
	target_link_libraries(grayscott PUBLIC OpenMP::OpenMP_CXX)
endif()
if(USE_CUDA)
	target_link_libraries(grayscott cuda_backend)
endif()

if(USE_OPENMP)
	target_compile_definitions(grayscott PRIVATE USE_OPENMP)
endif()
if(USE_CUDA)
        target_compile_definitions(grayscott PRIVATE USE_CUDA)
endif()




