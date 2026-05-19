#include <stdio.h>     
#include <cublas_v2.h>        
#include "utils/utils.cuh"            
#define PPP 1
#include <cuda_runtime.h> 
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "kernels.cuh"
#include <ctime>
#define multi 20   
int main(int argc, char **argv){
// Print start timestamp
time_t timestamp;
time(&timestamp);
printf("Start timestamp: %ld, %s", timestamp, ctime(&timestamp));

// Iinitialization   
srand(10);      
int start_size = atoi(argv[1]);        
int end_size =  atoi(argv[2]);           
int test_size = end_size   ;
int MAX_SIZE = end_size;
int gap_size =  atoi(argv[3]);        
int st_kernel = atoi(argv[4]);  
int end_kernel = atoi(argv[5]);     
int kernel_number;
int num_tests = 5;                            
float alpha = 1.0;                      
float negative_1 = -1.0;                           
float beta = 0.0;
int max_size = max(end_size,test_size);        
int M, N, K;    
M = test_size; N = test_size;  K = test_size;
float *A = NULL, *B = NULL, *C_ref = NULL, *C = NULL, *E = NULL, *E_ = NULL, *Res=NULL, *error_injec = NULL;
float *check_A_col = NULL, *check_B_row = NULL, *check_C_col = NULL, *check_C_row = NULL, *check_A_row_mul_C=NULL, *check_B_row_mul_C=NULL;
float *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC = NULL, *dE=NULL, *dE_ = NULL, *dRes =NULL, *derror_injec=NULL;
float *dcheck_A_col = NULL, *dcheck_B_row = NULL, *dcheck_C_col = NULL, *dcheck_C_row = NULL, *dcheck_A_col_mul_B=NULL, *dcheck_B_row_mul_A=NULL;
int size = max_size * sizeof (int);               
int deviceId;          
cudaGetDevice(&deviceId);            
cudaDeviceProp props = getDetails(deviceId);          

A = (float *)malloc(sizeof(float) * MAX_SIZE * MAX_SIZE);
error_injec = (float *)malloc(sizeof(float) * MAX_SIZE);
B = (float *)malloc(sizeof(float) * MAX_SIZE * MAX_SIZE);    
C = (float *)malloc(sizeof(float) * MAX_SIZE * MAX_SIZE);                                   
E = (float *)malloc(sizeof(float) * MAX_SIZE);  
E_ = (float *)malloc(sizeof(float) * MAX_SIZE);   
Res = (float *)malloc(sizeof(float) * 1);
check_A_col = (float *)malloc(sizeof(float) * MAX_SIZE); 
check_B_row = (float *)malloc(sizeof(float) * MAX_SIZE); 
check_C_col = (float *)malloc(sizeof(float) * MAX_SIZE);     
check_C_row = (float *)malloc(sizeof(float) * MAX_SIZE);
check_A_row_mul_C = (float *)malloc(sizeof(float) * MAX_SIZE);
check_B_row_mul_C = (float  *)malloc(sizeof(float) * MAX_SIZE);
            
C_ref = (float *)malloc(sizeof(float) * MAX_SIZE * MAX_SIZE);    
generate_random_matrix(A, MAX_SIZE);
generate_random_matrix(B, MAX_SIZE);                                                           
generate_random_matrix(C, MAX_SIZE);
fill_vector(Res, 0.0, 1);       
fill_vector(C, 0.0, MAX_SIZE * MAX_SIZE);          
fill_vector(error_injec, 1.0, MAX_SIZE);        
fill_vector(E, 1.0, MAX_SIZE);      
fill_vector(check_A_col, 0.0, MAX_SIZE);  
fill_vector(check_B_row, 0.0, MAX_SIZE); 
fill_vector(check_C_col, 0.0, MAX_SIZE);                                                    
fill_vector(check_C_row, 0.0, MAX_SIZE);  
fill_vector(check_A_row_mul_C, 0.0, MAX_SIZE);
fill_vector(check_B_row_mul_C, 0.0, MAX_SIZE);         
copy_matrix(C, C_ref, MAX_SIZE); 
for(int i = 1; i <= MAX_SIZE; ++i)E_[i] = (float)i; 

// Jithin - DEBUGGING START
int *d_debug_int = NULL, *debug_int = NULL;
debug_int = (int *)malloc(sizeof(int) * 10);
CUDA_CALLER(cudaMalloc((void**) &d_debug_int, sizeof(int)*10));
CUDA_CALLER(cudaMemcpy(d_debug_int, debug_int, sizeof(int)*10, cudaMemcpyHostToDevice));
// Fill matrices
fill_vector(A, 1.0, MAX_SIZE * MAX_SIZE);
fill_vector(B, 0.0, MAX_SIZE * MAX_SIZE);
fill_vector(C, 0.0, MAX_SIZE * MAX_SIZE);
// Make B an identity matrix
for (int i=0; i<MAX_SIZE; i++)
{
    B[i * MAX_SIZE + i] = 1.0;
}
// Set one element in A
int row_in = 0;
int column_in = 0;
A[column_in * MAX_SIZE + row_in] = 5.0;
// Jithin - DEBUGGING END
    
        
CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(float) * MAX_SIZE * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(float) * MAX_SIZE * MAX_SIZE));  
CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * MAX_SIZE * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(float) * MAX_SIZE * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &derror_injec, sizeof(float) * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dE, sizeof(float) * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dE_, sizeof(float) * MAX_SIZE)); 
CUDA_CALLER(cudaMalloc((void**) &dcheck_A_col, sizeof(float) * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dcheck_B_row, sizeof(float) * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dcheck_C_col, sizeof(float) * MAX_SIZE)); 
CUDA_CALLER(cudaMalloc((void**) &dcheck_C_row, sizeof(float) * MAX_SIZE));
CUDA_CALLER(cudaMalloc((void**) &dcheck_A_col_mul_B, sizeof(float) * MAX_SIZE)); 
CUDA_CALLER(cudaMalloc((void**) &dcheck_B_row_mul_A, sizeof(float) * MAX_SIZE));  
CUDA_CALLER(cudaMalloc((void**) &dRes, sizeof(float)));   
CUDA_CALLER(cudaMemcpy(dE, E, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));  
CUDA_CALLER(cudaMemcpy(dE_, E_, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice)); 
CUDA_CALLER(cudaMemcpy(dcheck_A_col, check_A_col, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dcheck_B_row, check_B_row, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dcheck_C_col, check_C_col, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dcheck_C_row, check_C_row, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dcheck_A_col_mul_B, check_A_row_mul_C, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dcheck_B_row_mul_A, check_B_row_mul_C, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dRes, Res, sizeof(float), cudaMemcpyHostToDevice));        
CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));     
CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));        
CUDA_CALLER(cudaMemcpy(dC_ref, C, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
CUDA_CALLER(cudaMemcpy(derror_injec, error_injec, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
        
    
// Verification           
printf("Start verification!\n");
cublasHandle_t handle;                  
cublasCreate(&handle);                 
cudaDeviceSynchronize(); 
beta = 0;                      
for(int i = st_kernel; i <= end_kernel; ++i){
    kernel_number = i;
for (int iter = 0; iter < 1 ; iter++){ 
cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC_ref, M);
}
if(kernel_number == 1){                             
    dim3 blockDim(64);                        
    dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));      
    cudaDeviceSynchronize(); 
    sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
}  
else if(kernel_number == 2){        
    dim3 blockDim(64);      
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));     
    cudaDeviceSynchronize(); 
    sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
}         
else if(kernel_number == 3){         
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64)); 
    cudaDeviceSynchronize();  
    sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
}  
else if(kernel_number == 4){ 
    dim3 blockDim(128);      
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32));
    cudaDeviceSynchronize();            
    sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
}                 
else if(kernel_number == 5){                                   
    dim3 blockDim(128);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 128)); 
    cudaDeviceSynchronize();   
    sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
}                 
else if(kernel_number == 6){  
    for (int iter = 0; iter < 1; iter++){
    dim3 blockDim(256);    
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    printf("%d, %d, %d, %d, %d\n", gridDim.x, gridDim.y, M, N, K);
    cudaDeviceSynchronize(); 
    sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }      
}         
else if(kernel_number == 11){        
    for (int iter = 0; iter < 1; iter++){ 
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
    cudaDeviceSynchronize(); 
    ft_sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }
}  
else if(kernel_number == 12){
    for (int iter = 0; iter < 1; iter++){
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    cudaDeviceSynchronize();  
    ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta, dcheck_A_col, dcheck_B_row, d_debug_int);  
    }                
}      
else if(kernel_number == 13){   
    for (int iter = 0; iter < 1; iter++){  
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
    cudaDeviceSynchronize();       
    ft_sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }
} 
else if(kernel_number == 14){
    for (int iter = 0; iter < 1; iter++){     
    dim3 blockDim(128);                           
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32));
    cudaDeviceSynchronize(); 
    ft_sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }   
}              
else if(kernel_number == 15){
    for (int iter = 0; iter < 1; iter++){
    dim3 blockDim(128);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,  128));  
    cudaDeviceSynchronize(); 
    ft_sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }  
}  
else if(kernel_number == 16){
    for (int iter = 0; iter < 1; iter++){
    dim3 blockDim(256);  
    dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
    cudaDeviceSynchronize(); 
    ft_sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }
}   
else if(kernel_number == 17){
    for (int iter = 0; iter < 1; iter++){
    dim3 blockDim(96);  
    dim3 gridDim(CEIL_DIV(M, 48), CEIL_DIV(N, 48));
    cudaDeviceSynchronize(); 
    ft_sgemm_medium_96<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
    }
}  
else{
    cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K, &alpha, dA, M, dB, N, &beta, dC, M);
} 
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

cudaDeviceSynchronize();    
cudaMemcpy(C, dC, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
cudaMemcpy(C_ref, dC_ref, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();                                                                    

// Jithin - DEBUGGING START
cudaMemcpy(check_A_col, dcheck_A_col, sizeof(float) * MAX_SIZE, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();
// cudaMemcpy(check_B_row, dcheck_B_row, sizeof(float) * MAX_SIZE, cudaMemcpyDeviceToHost);
// cudaDeviceSynchronize();
cudaMemcpy(debug_int, d_debug_int, sizeof(int)*10, cudaMemcpyDeviceToHost);
cudaDeviceSynchronize();

int rows = 5;
int columns = 10;
float values_ref[rows*columns];
float values[rows*columns];
int row_start = 0;
int column_start = 0;
for (int i=0; i<columns; i++)
{
    for (int j=0; j<rows; j++)
    {
        values_ref[i * rows + j] = C_ref[(i+column_start) * M + (j+row_start)];
        values[i * rows + j] = C[(i+column_start) * M + (j+row_start)];
    }
}
printf("--- C_ref -----------------------------\n");
// Print values from C_ref
for (int j=0; j<rows; j++)
{
    for (int i=0; i<columns; i++)
    {
        printf("%f, ", values_ref[i * rows + j]);
    }
    printf("\n");
}
printf("--- C result -------------------------------\n");
// Print values from C
for (int j=0; j<rows; j++)
{
    for (int i=0; i<columns; i++)
    {
        printf("%f, ", values[i * rows + j]);
    }
    printf("\n");
}
printf("--- A column checksums -------------------------------\n");
// Print checksum values
// for (int i=0; i<columns; i++)
// {
//     printf("%f, ", check_A_col[i + column_start]);
// }
int check_count = 0;
int failed_columns[MAX_SIZE];
int failed_values[MAX_SIZE];
for (int i=0; i<MAX_SIZE; i++)
{
    if (check_A_col[i] != MAX_SIZE)
    {
        failed_columns[check_count] = i;
        failed_values[check_count] = check_A_col[i];
        check_count++;
    }
}
printf("Number of checksum failures: %d\n", check_count);
printf("Failed columns: ");
for (int i=0; i<check_count; i++)
{
    printf("%d (%d), ", failed_columns[i], failed_values[i]);
}
printf("\n");
// printf("--- debug int -------------------------------\n");
// printf("clock: %d\n", debug_int[0]);
// printf("clock64: %d\n", debug_int[1]);
// printf("clock: %d\n", debug_int[2]);
// printf("clock64: %d\n", debug_int[3]);
printf("--- END -----------------------------\n");
// Jithin - DEBUGGING END

if (!verify_matrix(C_ref, C, M, N)) { 
    printf("kernel %d failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n", kernel_number);
    // exit(-3);  
}    
fflush(stdout);              
printf("kernel %d finish verified!\n", kernel_number);      
cudaDeviceSynchronize();  
}
// Performance Profiling    
printf("################## Performance (GFLOPS) ########################\n");
// printf("##################### kernel %d #########################\n", kernel_number);
// return 0; 
beta=-1.5;
int list[15] = {0, 1, 2, 3, 4, 5, 6, 10, 11, 12, 13, 14, 15, 16, 17};
char arr[15][24] = {"cublas", "kernel_sgemm_small", "kernel_sgemm_medium", "kernel_sgemm_large", "kernel_sgemm_tall", "kernel_sgemm_wide", "kernel_sgemm_huge",
                    "abft_baseline", "abft_kernel_small", "abft_kernel_medium", "abft_kernel_large", "abft_kernel_tall", "abft_kernel_wide", "abft_kernel_huge", "abft_kernel_medium_96"};
// return 0;  
printf("Matrix Size         |");
for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
printf("%8d|", max_size);
}    
printf("\n");
for(int jj = 0; jj < 15; ++jj){
    kernel_number = list[jj];       
    if(kernel_number < st_kernel)continue;
    if(kernel_number > end_kernel) break;                                                
    printf("%-20s|", arr[jj]);
    // CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
    for(int max_size = start_size; max_size <= end_size; max_size += gap_size){
    N = K = M = max_size;                                     
    cudaEvent_t beg, end; 
    cudaEventCreate(&beg);                        
    cudaEventCreate(&end); 
    float elapsed = 0;       
    if (kernel_number == 0){     
        cudaEventRecord(beg);                      
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();    
            cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_N, M, N, K, &alpha, dA, M, dB, K, &beta, dC, M);
            cudaDeviceSynchronize();  
        }
        cudaEventRecord(end);
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);  
    }    
    else if (kernel_number == 1){
        cudaEventRecord(beg);
        dim3 blockDim(64);
        dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
        
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);  
    }  
    else if (kernel_number == 2){ 
        cudaEventRecord(beg);
        dim3 blockDim(64);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        } 
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);  
    }  
    else if (kernel_number == 3){     
        cudaEventRecord(beg);                
        dim3 blockDim(64);                           
        dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize(); 
            sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize(); 
        }
        cudaEventRecord(end);      
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 4){
        cudaEventRecord(beg); 
        dim3 blockDim(128);
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32)); 
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }   
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);                  
    } 
    else if (kernel_number == 5){
        cudaEventRecord(beg); 
        dim3 blockDim(128);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 128));
        for(int ii = 0; ii < num_tests; ++ii){         
            cudaDeviceSynchronize();
            sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 6){
        cudaEventRecord(beg);
        dim3 blockDim(256);
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 10){
        cudaEventRecord(beg);
        baseline_ft_sgemm(num_tests, M,N, K, handle, dA, dB, dC, dE, dRes, dcheck_C_row, dcheck_C_col, dcheck_A_col_mul_B, dcheck_B_row_mul_A, dcheck_A_col, dcheck_B_row,  alpha, beta, negative_1);
        cudaEventRecord(end);
        cudaEventSynchronize(beg);  
        cudaEventSynchronize(end); 
    }
    else if (kernel_number == 11){ 
        cudaEventRecord(beg);
        dim3 blockDim(64);
        dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            ft_sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize(); 
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    }  
    else if (kernel_number == 12){ 
        cudaEventRecord(beg);
        dim3 blockDim(64); 
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta, dcheck_A_col, dcheck_B_row, d_debug_int);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end);  
    }  
    else if (kernel_number == 13){                                      
        cudaEventRecord(beg);                                  
        dim3 blockDim(64);                       
        dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            ft_sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 14){
        cudaEventRecord(beg);             
        dim3 blockDim(128);
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();
            ft_sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);             
        cudaEventSynchronize(end); 
    }  
    else if (kernel_number == 15){
        cudaEventRecord(beg);
        dim3 blockDim(128);
        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 128));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();        
            ft_sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);      
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 16){
        cudaEventRecord(beg);
        dim3 blockDim(256);   
        dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();   
                ft_sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC,  alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    else if (kernel_number == 17){
        cudaEventRecord(beg);
        dim3 blockDim(96);   
        dim3 gridDim(CEIL_DIV(M, 48), CEIL_DIV(N, 48));
        for(int ii = 0; ii < num_tests; ++ii){
            cudaDeviceSynchronize();   
                ft_sgemm_medium_96<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC,  alpha, beta);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(end);     
        cudaEventSynchronize(beg);
        cudaEventSynchronize(end); 
    } 
    cudaEventElapsedTime(&elapsed, beg, end);                     
    double gflops  = 0.;
    gflops = double(2 * num_tests * double(M) * double(N) * double(K)) / (1e9);
    double perf = gflops / (elapsed / 1e3);
    printf("%8.0f|", perf);
    fflush(stdout);
}
printf("\n");
}
time(&timestamp);
printf("End timestamp: %ld, %s", timestamp, ctime(&timestamp));
}
