cmake_minimum_required(VERSION 3.14)
project(gemm_naive)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(NVCC_FLAGS "-O3")

set(CMAKE_CUDA_COMPILER  /usr/local/cuda/bin/nvcc)
set(CUDA_ARCHITECTURES 86 89)
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
# Check for CUDA
# find_package(CUDAToolkit REQUIRED)
# include_directories(${CUDAToolkit_INCLUDE_DIRS})
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-extended-lambda;-G)

#include
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(/usr/local/cuda/include)

# Enable CUDA for the project
enable_language(CUDA)

# Set source files
set(SOURCE_FILES
  src/gemm_test.cpp 
  src/cuda/cuda_gemm.cu
  # Add other .cpp and .cu files here

  src/cpu/cpu_naive.cpp
)

# Add the executable
add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_link_libraries(${PROJECT_NAME} cublas)
target_compile_options(${PROJECT_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${NVCC_FLAGS}>)

# #假设你想为CUDA目标添加编译选项
# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_NVCC_FLAGS "--ptxas-options=-v"
# )

# target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart CUDA::cublas)

# set_target_properties(${PROJECT_NAME} PROPERTIES
#     CUDA_SEPARABLE_COMPILATION ON # 允许分离编译
#     CUDA_ARCHITECTURES 86
# )
# find_package(OpenMP)

# # 如果 OpenMP 可用，设置编译选项
# if(OPENMP_FOUND)
#     # 设置 OpenMP 编译选项
#     set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#     set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#     set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")

#     target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_C)
#     target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)
# endif()