cmake_minimum_required(VERSION 3.18)
project(ft_sgemm LANGUAGES CXX CUDA)


set(CMAKE_CUDA_ARCHITECTURES 87)


set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 -lineinfo")

find_package(CUDAToolkit REQUIRED)

# 包含头文件路径
include_directories(
    ${CUDAToolkit_INCLUDE_DIRS}
    cuda-samples/Common/
    ${CMAKE_SOURCE_DIR}
)


set(KERNEL_PATH "kernel/ft_sgemm")


set(SRC_FILES
    ${KERNEL_PATH}/sgemm.cu
    utils/utils.cu
)

set(SRC_FILES_RUN_KERNEL_NO_CHECK
    ${KERNEL_PATH}/run_kernel_no_check.cu
    utils/utils.cu
)

set(SRC_FILES_GENERATE_MATRICES
    ${KERNEL_PATH}/generate_matrices.cu
    utils/utils.cu
)

add_executable(ft_sgemm ${SRC_FILES})
add_executable(run_kernel_no_check ${SRC_FILES_RUN_KERNEL_NO_CHECK})
add_executable(generate_matrices ${SRC_FILES_GENERATE_MATRICES})


target_link_libraries(ft_sgemm PRIVATE CUDA::cudart CUDA::cublas jetgpio)
target_link_libraries(run_kernel_no_check PRIVATE CUDA::cudart CUDA::cublas jetgpio)
target_link_libraries(generate_matrices PRIVATE CUDA::cudart CUDA::cublas)

target_compile_definitions(run_kernel_no_check PUBLIC DISABLE_ERROR_CORRECTION)
