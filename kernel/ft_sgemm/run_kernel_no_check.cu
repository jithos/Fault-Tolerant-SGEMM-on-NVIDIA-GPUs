#include <stdio.h>     
#include <cublas_v2.h>        
#include "utils/utils.cuh"            
#define PPP 1
#include <cuda_runtime.h> 
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "kernels.cuh"
// #include <ctime>
#include <jetgpio.h>
#include <fstream>
#include <filesystem>
#include <chrono>
#define multi 20

/* TODO: REMOVE after DEBUGGING */
#include <thread>

#define RESULTS_FILE "_results.csv"
#define ERROR_INDICES_FILE "_indices.csv"

// #define DO_CUBLAS_VERIFICATION

// /* Global variable to interrupt the loop later on */
static volatile int wait_trigger = 1;
unsigned long trigger_timestamp;

// /* Variable for trigger GPIO pin */
int trigger_gpio = 7;

/* Global variables */
time_t time_convert;
uint64_t start_time, end_time;
float *A = NULL, *B = NULL, *C_ref = NULL, *C_BLAS = NULL, *C = NULL;
float *check_C_col = NULL, *check_C_row = NULL;
float *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC_BLAS = NULL, *dC = NULL;
float *dcheck_C_col = NULL, *dcheck_C_row = NULL;
const char* folder_name;
const char* output_folder;
const char* experiment_name;
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
    printf("Read matrix from %s\n", filename);
}

void write_header_to_results_file(const char* folder, const char* experiment_name){

    /* ----------------------------- */
    /* Write header for results file */
    /* ----------------------------- */

    // Define the filename for results file
    std::string results_filename = std::string(folder) + std::string("exp_") + std::string(experiment_name) + std::string(RESULTS_FILE);

    // Open the file in output mode to write the header
    std::fstream results_fd(results_filename, std::ios::out | std::ios::binary);
    if (!results_fd.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", results_filename.c_str());
        results_fd.close();
        exit(EXIT_FAILURE);
    }
    
    // Write header for results file
    std::stringstream results_header;
    results_header << "matrix_size" << ","
        << "experiment_name" << ","
        << "file_A" << ","
        << "file_B" << ","
        << "file_C" << ","
        << "start_time" << ","
        << "end_time" << ","
        << "trigger_timestamp" << ","
        << "trigger_signal_enabled" << ","
        << "seu_count_total" << ","
        << "kernel_number" << "\n";
    // printf(results_header.str().c_str());
    results_fd << results_header.str();

    // Close the file after writing
    results_fd.close();
    printf("Wrote header to %s\n", results_filename.c_str());
}

void write_header_to_indices_file(const char* folder, const char* experiment_name) {
    
    /* ----------------------------- */
    /* Write header for indices file */
    /* ----------------------------- */

    // Define the filename for indices file
    std::string indices_filename = std::string(folder) + std::string("exp_") + std::string(experiment_name) + std::string(ERROR_INDICES_FILE);

    // Open the file in output mode to write the header
    std::fstream indices_fd(indices_filename, std::ios::out | std::ios::binary);
    if (!indices_fd.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", indices_filename.c_str());
        indices_fd.close();
        exit(EXIT_FAILURE);
    }
    
    // Write header for indices file
    std::stringstream indices_header;
    indices_header << "sanity_error_row_index" << ","
        << "sanity_error_col_index" << ","
        << "error_row_index" << ","
        << "error_col_index" << ","
        << "error_diff" << ","
        << "seu_count" << ","
        << "repetition" << "\n";
    // printf(indices_header.str().c_str());
    indices_fd << indices_header.str();

    // Close the file after writing
    indices_fd.close();
    printf("Wrote header to %s\n", indices_filename.c_str());
}

void write_results_to_file(
        const char* folder, 
        int matrix_size,
        const char* experiment_name,
        char* file_A,
        char* file_B,
        char* file_C,
        uint64_t start_time,
        uint64_t end_time,
        unsigned long trigger_timestamp,
        bool trigger_signal_enabled,
        unsigned int seu_count_total,
        int kernel_number
    ) {
    
    // Define the filename for results file
    std::string filename = std::string(folder) + std::string("exp_") + std::string(experiment_name) + std::string(RESULTS_FILE);

    // Open the file in append mode to add results
    std::fstream file(filename, std::ios::out | std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename.c_str());
        file.close();
        exit(EXIT_FAILURE);
    }

    // Write the results as a new line in the CSV file
    std::stringstream result_line;
    result_line << matrix_size << ","
        << experiment_name << ","
        << file_A << ","
        << file_B << ","
        << file_C << ","
        << start_time << ","
        << end_time << ","
        << trigger_timestamp << ","
        << trigger_signal_enabled << ","
        << seu_count_total << ","
        << kernel_number << "\n";

    file << result_line.str();

    // Close the file after writing
    file.close();
    printf("Saved results to %s\n", filename.c_str());
}

void write_indices_to_file(
        const char* folder,
        int* sanity_error_row_index,
        int* sanity_error_col_index,
        int* error_row_index,
        int* error_col_index,
        float* error_diff,
        unsigned int seu_count,
        int repetition,
        const char* experiment_name
    ) {

    // Define the filename for indices file
    std::string filename = std::string(folder) + std::string("exp_") + std::string(experiment_name) + std::string(ERROR_INDICES_FILE);

    // Open the file in append mode to add results
    std::fstream file(filename, std::ios::out | std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", filename.c_str());
        file.close();
        exit(EXIT_FAILURE);
    }

    // Write the error indices as a new line in the CSV file
    for (int i = 0; i < seu_count; i++) {
        // Replace NULL pointers with default placeholder value
        int sanity_row_idx = sanity_error_row_index != NULL ? sanity_error_row_index[i] : -1;
        int sanity_col_idx = sanity_error_col_index != NULL ? sanity_error_col_index[i] : -1;
        int error_row_idx = error_row_index != NULL ? error_row_index[i] : -1;
        int error_col_idx = error_col_index != NULL ? error_col_index[i] : -1;

        std::stringstream result_line;
        result_line << sanity_row_idx << ","
            << sanity_col_idx << ","
            << error_row_idx << ","
            << error_col_idx << ","
            << seu_count << ","
            << error_diff[i] << ","
            << repetition << "\n";
        file << result_line.str();
    }

    // Close the file after writing
    file.close();
}

void count_seu_errors(float* C, float* C_ref, int M, int N, unsigned int* seu_count)
{
    *seu_count = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            (*seu_count)++;
            // if (*seu_count <= 10) { // Print details for the first 10 errors
            //     printf("Mismatch at index %d: GPU result = %e, Reference = %e\n", i, C[i], C_ref[i]);
            // }
        }
    }
}

void get_seu_indices(float* C, float* C_ref, int M, int N, int* error_row_idx, int* error_col_idx, float* error_diff)
{
    int error_idx = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            error_row_idx[error_idx] = i % M;
            error_col_idx[error_idx] = i / M;
            error_diff[error_idx] = C[i] - C_ref[i];
            error_idx++;
        }
    }
}

// /* Function to handle GPIO interrupt */
void beam_line_trigger_handler() {
    time_t time = static_cast<time_t>(trigger_timestamp/1e9); // Convert nanoseconds to seconds
    printf("Trigger signal timestamp: %lu [us], %s\n", (unsigned long)(trigger_timestamp/1e3), ctime(&time));
    wait_trigger = 0;
}

// // Signal handler for Ctrl+C
// void signal_handler(int signum) {
//     printf("User pressed Ctrl+C. Stopping wait for beam line trigger if waiting...\n");
//     wait_trigger = 0;
// }

void atexit_handler() {
    printf("Cleaning up resources...\n");

    // Free up memories
    free(A);
    free(B);
    free(C);
    free(C_ref);
    free(check_C_col);
    free(check_C_row);
    #ifdef DO_CUBLAS_VERIFICATION
    free(C_BLAS);
    #endif

    // Release pheripherals
    gpioTerminate();

    uint64_t exit_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(exit_time/1e6); // Convert microseconds to seconds
    printf("Exit timestamp: %lu [us], %s\n", exit_time, ctime(&time_convert));
}

int main(int argc, char **argv){
    // Print start timestamp
    start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(start_time/1e6); // Convert microseconds to seconds
    printf("Start timestamp: %lu [us], %s\n", start_time, ctime(&time_convert));

    // Register atexit handler to ensure it gets called on normal exit
    const int result = std::atexit(atexit_handler); // Handler will be called
  
    if (result != 0)
    {
        std::cerr << "atexit registration failed\n";
        return EXIT_FAILURE;
    }

    // Inform user if ABFT error correction is disabled
    #ifdef DISABLE_ERROR_CORRECTION
    printf("\n----------------------------------------\n");
    printf("!! ABFT error correction is DISABLED !!");
    printf("\n----------------------------------------\n\n");
    #endif

    if (argc < 10 + 1)
    {
        printf("Expected 10 arguments: Matrix start size, end size, step size, kernel start number, end number, repetitions, enable_trigger_signal, input folder, output folder, experiment number. Some arguments are missing! Exiting...\n");
        exit(-1);
    }

    /* -------------------------- */
    /* Get command line arguments */
    /* -------------------------- */

    // int start_size = atoi(argv[1]);        
    int end_size =  atoi(argv[2]);
    int MAX_SIZE = end_size;
    // int gap_size =  atoi(argv[3]);        
    // int st_kernel = atoi(argv[4]);  
    int end_kernel = atoi(argv[5]);     
    int kernel_number = end_kernel;
    int repeat_kernel = atoi(argv[6]);
    bool enable_trigger_signal = atoi(argv[7]);
    folder_name = argv[8];
    sprintf(file_A_name, "%sA_%d.bin", folder_name, MAX_SIZE);
    sprintf(file_B_name, "%sB_%d.bin", folder_name, MAX_SIZE);
    sprintf(file_C_name, "%sC_%d.bin", folder_name, MAX_SIZE);
    output_folder = argv[9];
    experiment_name = argv[10];

    // Initialization   
    srand(10);
    float alpha = 1.0;
    float beta = 0.0;       
    int M, N, K;    
    M = MAX_SIZE; N = MAX_SIZE;  K = MAX_SIZE;
    int size = MAX_SIZE * sizeof (int);               
    int deviceId;          
    cudaGetDevice(&deviceId);            
    cudaDeviceProp props = getDetails(deviceId);

    printf("Allocating host memory...\n");
    A = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    B = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    C = (float *)calloc(MAX_SIZE * MAX_SIZE * repeat_kernel, sizeof(float));
    C_ref = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    check_C_col = (float *)calloc(MAX_SIZE, sizeof(float));
    check_C_row = (float *)calloc(MAX_SIZE, sizeof(float));

    // Initialize arrays with random values (or read from file)
    // -- Generarte random matrices (Very slow !! Read from file instead !!)
    // Use the generate matrix utility -> see generate_matrices.cu

    // -- Read input matrices from file
    printf("Reading input matrices from input files...\n");
    read_matrix_from_file(file_A_name, A, MAX_SIZE);
    read_matrix_from_file(file_B_name, B, MAX_SIZE);
    read_matrix_from_file(file_C_name, C_ref, MAX_SIZE);

    #ifdef DO_CUBLAS_VERIFICATION
    C_BLAS = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    // memset(C_BLAS, 0.0, sizeof(C_BLAS));
    #endif

    printf("Allocating device memory...\n");
    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(float) * MAX_SIZE * MAX_SIZE));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(float) * MAX_SIZE * MAX_SIZE));  
    CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * MAX_SIZE * MAX_SIZE * repeat_kernel));
    CUDA_CALLER(cudaMalloc((void**) &dC_ref, sizeof(float) * MAX_SIZE * MAX_SIZE));
    // CUDA_CALLER(cudaMalloc((void**) &dcheck_C_col, sizeof(float) * MAX_SIZE)); 
    // CUDA_CALLER(cudaMalloc((void**) &dcheck_C_row, sizeof(float) * MAX_SIZE));
    // CUDA_CALLER(cudaMemcpy(dcheck_C_col, check_C_col, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
    // CUDA_CALLER(cudaMemcpy(dcheck_C_row, check_C_row, sizeof(float) * MAX_SIZE, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));     
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC, C, sizeof(float) * MAX_SIZE * MAX_SIZE * repeat_kernel, cudaMemcpyHostToDevice));
    CUDA_CALLER(cudaMemcpy(dC_ref, C_ref, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));

    #ifdef DO_CUBLAS_VERIFICATION
    CUDA_CALLER(cudaMalloc((void**) &dC_BLAS, sizeof(float) * MAX_SIZE * MAX_SIZE));
    CUDA_CALLER(cudaMemcpy(dC_BLAS, C_BLAS, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));
    #endif

    /* -------------------- */
    /* Prepare output files */
    /* -------------------- */

    printf("Writing header to output files...\n");
    write_header_to_results_file(output_folder, experiment_name);
    write_header_to_indices_file(output_folder, experiment_name);

    /* ------------------------------ */
    /* Preapre cuda handle for CUBLAS */
    /* ------------------------------ */

    cublasHandle_t handle;                  
    cublasCreate(&handle);                 
    cudaDeviceSynchronize();
    printf("Intialized CUDA handle.\n");

    /* ---------------------------- */
    /* Run sanity check on matrices */
    /* ---------------------------- */

    // Sanity check with CUBLAS
    // cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC, M);

    // Sanity check with ABFT kernel
    printf("Starting sanity check kernel...\n");
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    cudaDeviceSynchronize();  
    ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);

    cudaDeviceSynchronize();
    printf("Copying sanity check results to host memory...\n");
    CUDA_CALLER(cudaMemcpy(C, dC, sizeof(float) * MAX_SIZE * MAX_SIZE * repeat_kernel, cudaMemcpyDeviceToHost));

    // Count SEU errors by comparing C with C_ref
    unsigned int seu_count = 0;
    printf("Calcuating SEU count for sanity check...\n");
    count_seu_errors(C, C_ref, M, N, &seu_count);

    // Get SEU error locations (row and column indices)
    int error_row_idx[seu_count];
    int error_col_idx[seu_count];
    float error_diff[seu_count];
    get_seu_indices(C, C_ref, M, N, error_row_idx, error_col_idx, error_diff);

    write_indices_to_file(
        output_folder,
        error_row_idx,
        error_col_idx,
        NULL, // (not applicable here)
        NULL, // (not applicable here)
        error_diff,
        seu_count,
        -1, // repetition (not applicable here)
        experiment_name
    );

    printf("Sanity check completed. SEUs prior to beam detected: %u\n", seu_count);

    /* -------------------------------------- */
    /* Wait for trigger signal from beam line */
    /* -------------------------------------- */

    if (enable_trigger_signal)
    {
        int Init = gpioInitialise();
        if (Init < 0) {
            printf("Jetgpio initialisation failed. Error code:  %d\n", Init);
            exit(1);
        }
        int stat = gpioSetMode(trigger_gpio, JET_INPUT); // Set GPIO pin as input
        if (stat < 0)
        {
            printf("Failed to set GPIO pin mode. Error code: %d\n", stat);
            exit(1);
        }
        // Set up interrupt handler for falling edge on the trigger GPIO pin
        stat = gpioSetISRFunc(trigger_gpio, FALLING_EDGE, 10 /* us */, &trigger_timestamp, &beam_line_trigger_handler);
        if (stat < 0)
        {
            printf("Failed to set GPIO alert function. Error code: %d\n", stat);
            exit(1);
        }
        printf("Waiting for trigger signal from beam line...\n");
        while (wait_trigger) {
            // Wait for the GPIO pin to go high
            if (!enable_trigger_signal)
            {
                break; // Continue without waiting for trigger signal if it's disabled
            }
        }
        printf("Trigger received!\n");
    }
    else
    {
        printf("Trigger signal reception is DISABLED.\n");
    }
    printf("Starting kernel execution...\n");

    /* -------------------- */
    /* Run selected kernels */
    /* -------------------- */

    #ifdef DO_CUBLAS_VERIFICATION
    printf("Start cublas sgemm\n");
    cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC_BLAS, M);
    #endif

    // Run the selected kernel(s) for the specified number of repetitions
    for (int repetition = 0; repetition < repeat_kernel; repetition++){
        if(kernel_number == 1){                             
            dim3 blockDim(64);                        
            dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));      
            cudaDeviceSynchronize(); 
            sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }  
        else if(kernel_number == 2){        
            dim3 blockDim(64);      
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));     
            cudaDeviceSynchronize(); 
            // sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
            sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta);  
        }         
        else if(kernel_number == 3){         
            dim3 blockDim(64);  
            dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64)); 
            cudaDeviceSynchronize();  
            sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }  
        else if(kernel_number == 4){ 
            dim3 blockDim(128);      
            dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32));
            cudaDeviceSynchronize();            
            sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }                 
        else if(kernel_number == 5){                                   
            dim3 blockDim(128);  
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 128)); 
            cudaDeviceSynchronize();   
            sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }                 
        else if(kernel_number == 6){  
            dim3 blockDim(256);    
            dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
            printf("%d, %d, %d, %d, %d\n", gridDim.x, gridDim.y, M, N, K);
            cudaDeviceSynchronize(); 
            sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);
        }         
        else if(kernel_number == 11){
            dim3 blockDim(64);  
            dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
            cudaDeviceSynchronize(); 
            ft_sgemm_small<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);
        }  
        else if(kernel_number == 12){
            dim3 blockDim(64);  
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
            cudaDeviceSynchronize();  
            ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);  
        }      
        else if(kernel_number == 13){   
            dim3 blockDim(64);  
            dim3 gridDim(CEIL_DIV(M, 64), CEIL_DIV(N, 64));
            cudaDeviceSynchronize();       
            ft_sgemm_large<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        } 
        else if(kernel_number == 14){
            dim3 blockDim(128);                           
            dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 32));
            cudaDeviceSynchronize(); 
            ft_sgemm_tall<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }              
        else if(kernel_number == 15){
            dim3 blockDim(128);  
            dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N,  128));  
            cudaDeviceSynchronize(); 
            ft_sgemm_wide<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }  
        else if(kernel_number == 16){
            dim3 blockDim(256);  
            dim3 gridDim(CEIL_DIV(M, 128), CEIL_DIV(N, 128));
            cudaDeviceSynchronize(); 
            ft_sgemm_huge<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }   
        else if(kernel_number == 17){
            dim3 blockDim(96);  
            dim3 gridDim(CEIL_DIV(M, 48), CEIL_DIV(N, 48));
            cudaDeviceSynchronize(); 
            ft_sgemm_medium_96<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);  
        }  
        else{
            cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K, &alpha, dA, M, dB, N, &beta, dC+repetition*MAX_SIZE*MAX_SIZE, M);
            printf("Invalid kernel number provided. Running cuBLAS sgemm by default.\n");
        } 

        /* Pause between kernels (used for debugging & current measurements) - START */
        // cudaDeviceSynchronize();
        // time_t kernel_timestamp = time(nullptr);
        // printf("[%s] Repetition %d finisched.\n", ctime(&kernel_timestamp), repetition);
        // std::this_thread::sleep_for(std::chrono::seconds(1));
        /* Pause between kernels (used for debugging & current measurements) - END */
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

    cudaMemcpy(C, dC, sizeof(float) * M * N * repeat_kernel, cudaMemcpyDeviceToHost);

    #ifdef DO_CUBLAS_VERIFICATION
    cudaMemcpy(C_BLAS, dC_BLAS, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    if (!verify_matrix(C_BLAS, C, M, N)) { 
        printf("kernel %d failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n", kernel_number);
        // exit(-3);  
    }    
    fflush(stdout);              
    printf("kernel %d finish verified!\n", kernel_number);      
    cudaDeviceSynchronize();
    #endif

    /* ---------------- */
    /* Count SEU errors */
    /* ---------------- */

    unsigned int seu_count_total = 0;
    for (int repetition = 0; repetition < repeat_kernel; repetition++) {
        printf("Repetition %d - ", repetition + 1);

        // Count SEU errors by comparing C with C_ref
        unsigned int seu_count = 0;
        count_seu_errors(C + repetition * M * N, C_ref, M, N, &seu_count);
        printf("SEU errors detected: %u\n", seu_count);

        // Get SEU error locations (row and column indices)
        int error_row_idx[seu_count];
        int error_col_idx[seu_count];
        float error_diff[seu_count];
        get_seu_indices(C + repetition * M * N, C_ref, M, N, error_row_idx, error_col_idx, error_diff);

        write_indices_to_file(
            output_folder,
            NULL, // sanity_error_row_index (not applicable here)
            NULL, // sanity_error_col_index (not applicable here)
            error_row_idx,
            error_col_idx,
            error_diff,
            seu_count,
            repetition,
            experiment_name
        );

        // Accumulate SEU counts over all kernel repetitions
        seu_count_total += seu_count;
    }

    printf("Total SEU errors across all repetitions: %d\n", seu_count_total);

    end_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(end_time/1e6); // Convert microseconds to seconds
    printf("End timestamp: %lu [us], %s\n", end_time, ctime(&time_convert));

    write_results_to_file(
        output_folder,
        MAX_SIZE,
        experiment_name,
        file_A_name,
        file_B_name,
        file_C_name,
        start_time,
        end_time,
        trigger_timestamp,
        enable_trigger_signal,
        seu_count_total,
        kernel_number
    );

    exit(0); // Exit normally to ensure atexit handler is called
}
