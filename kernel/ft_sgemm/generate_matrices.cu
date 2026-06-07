#include <stdio.h>     
#include <cublas_v2.h>        
#include "utils/utils.cuh"            
#define PPP 1
#include <cuda_runtime.h> 
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "kernels.cuh"
#include <ctime>
#include <jetgpio.h>
#include <fstream>
#define multi 20

/* Global variables */
float *A = NULL, *B = NULL, *C = NULL;
float *dA = NULL,*dB = NULL, *dC = NULL;
const char* folder_name;
char file_A_name[256];
char file_B_name[256];
char file_C_name[256];

void read_matrix_from_file(const char* filename, float* matrix, int matrix_size) {
    std::fstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        file.close();
        exit(EXIT_FAILURE);
    }
    file.read((char *) matrix, sizeof(float) * matrix_size * matrix_size);
    file.close();
}

void write_matrix_to_file(const char* filename, float* matrix, int matrix_size) {
    std::fstream file(filename, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename);
        file.close();
        exit(EXIT_FAILURE);
    }
    file.write((char *) matrix, sizeof(float) * matrix_size * matrix_size);
    file.close();
}

void atexit_handler() {
    printf("Cleaning up resources...\n");

    // Free up memories
    free(A);
    free(B);
    free(C);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
}

int main(int argc, char **argv){
    // Register atexit handler to ensure it gets called on normal exit
    const int result = std::atexit(atexit_handler); // Handler will be called
  
    if (result != 0)
    {
        std::cerr << "atexit registration failed\n";
        return EXIT_FAILURE;
    }

    if (argc < 2 + 1)
    {
        printf("Expected 2 arguments: Matrix size, output folder. Some arguments are missing! Exiting...\n");
        exit(-1);
    }

    // Iinitialization   
    srand(10);   
    int size =  atoi(argv[1]);
    int MAX_SIZE = size;
    folder_name = argv[2];
    sprintf(file_A_name, "%sA_%d.bin", folder_name, size);
    sprintf(file_B_name, "%sB_%d.bin", folder_name, size);
    sprintf(file_C_name, "%sC_%d.bin", folder_name, size);
    float alpha = 1.0;
    float beta = 0.0;
    int M, N, K;
    M = MAX_SIZE; N = MAX_SIZE;  K = MAX_SIZE;

    A = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    B = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    C = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));

    // -- Generarte random matrices (Very slow !!)
    printf("Generating random matrices...\n");
    generate_random_matrix(A, MAX_SIZE);
    generate_random_matrix(B, MAX_SIZE);
    // memset(check_C_col, 0.0, sizeof(check_C_col));
    // memset(check_C_row, 0.0, sizeof(check_C_row));

    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(float) * MAX_SIZE * MAX_SIZE));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(float) * MAX_SIZE * MAX_SIZE));  
    CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * MAX_SIZE * MAX_SIZE));
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));     
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));

    // Verification
    cublasHandle_t handle;                  
    cublasCreate(&handle);                 
    cudaDeviceSynchronize();

    // Use CUBLAS sgemm to generate output matrix C for sanity check
    // printf("Start cublas sgemm\n");
    // cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K, &alpha, dA, M, dB, N, &beta, dC, M);

    // Use ABFT kernel to generate output matrix C for sanity check
    printf("Start ABFT kernel ft_sgemm_medium\n");
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    cudaDeviceSynchronize();  
    ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);

    // Check launch error
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "[Kernel Launch Error] %s\n", cudaGetErrorString(launchErr));
        exit(EXIT_FAILURE);
    }

    // Check execution error
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[Kernel Execution Error] %s\n", cudaGetErrorString(syncErr));
        exit(EXIT_FAILURE);
    }

    printf("[Kernel Completed Successfully]\n");

    cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);

    // Print values from B
    printf("-------------------------------------------\n");
    printf("Values from B:\n");
    for (int j=0; j<5; j++)
    {
        for (int i=0; i<5; i++)
        {
            printf("%e, ", B[i * MAX_SIZE + j]);
        }
        printf("\n");
    }
    printf("-------------------------------------------\n");
    
    // Store generated input matrices to file
    write_matrix_to_file(file_A_name, A, MAX_SIZE);
    write_matrix_to_file(file_B_name, B, MAX_SIZE);
    write_matrix_to_file(file_C_name, C, MAX_SIZE);

    // Sanity check
    read_matrix_from_file(file_B_name, B, MAX_SIZE);

    // Print values from B
    printf("-------------------------------------------\n");
    printf("Values of B read from file:\n");
    for (int j=0; j<5; j++)
    {
        for (int i=0; i<5; i++)
        {
            printf("%e, ", B[i * MAX_SIZE + j]);
        }
        printf("\n");
    }
    printf("-------------------------------------------\n");

    printf("Saved %s\n", file_A_name);
    printf("Saved %s\n", file_B_name);
    printf("Saved %s\n", file_C_name);

    exit(0); // Exit normally to ensure atexit handler is called
}
