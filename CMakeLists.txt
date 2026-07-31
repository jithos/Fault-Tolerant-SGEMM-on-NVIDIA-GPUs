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

set(SRC_FILES_RUN_KERNEL_SEU_COUNT
    ${KERNEL_PATH}/run_kernel_no_check.cu
    utils/utils.cu
)

set(SRC_FILES_RUN_KERNEL_SEU_DATA
    ${KERNEL_PATH}/run_kernel_no_check.cu
    utils/utils.cu
)

set(SRC_FILES_GENERATE_MATRICES
    ${KERNEL_PATH}/generate_matrices.cu
    utils/utils.cu
)

set(SRC_FILES_RUN_BEAM_TIMESTAMP
    ${KERNEL_PATH}/run_beam_timestamp.cu
    utils/utils.cu
)

add_executable(ft_sgemm ${SRC_FILES})
add_executable(run_kernel_seu_count ${SRC_FILES_RUN_KERNEL_SEU_COUNT})
add_executable(run_kernel_seu_data ${SRC_FILES_RUN_KERNEL_SEU_DATA})
add_executable(generate_matrices ${SRC_FILES_GENERATE_MATRICES})
add_executable(run_beam_timestamp ${SRC_FILES_RUN_BEAM_TIMESTAMP})


target_link_libraries(ft_sgemm PRIVATE CUDA::cudart CUDA::cublas jetgpio)
target_link_libraries(run_kernel_seu_count PRIVATE CUDA::cudart CUDA::cublas jetgpio)
target_link_libraries(run_kernel_seu_data PRIVATE CUDA::cudart CUDA::cublas jetgpio)
target_link_libraries(generate_matrices PRIVATE CUDA::cudart CUDA::cublas)
target_link_libraries(run_beam_timestamp PRIVATE jetgpio)

target_compile_definitions(run_kernel_seu_count PUBLIC DISABLE_ERROR_CORRECTION)
target_compile_definitions(run_kernel_seu_data PUBLIC DISABLE_ERROR_CORRECTION ENABLE_SEU_DATA_LOGGING)
