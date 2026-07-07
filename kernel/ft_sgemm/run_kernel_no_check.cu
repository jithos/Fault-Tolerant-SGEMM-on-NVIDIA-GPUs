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
#include <unistd.h>

#define RESULTS_FILE "results.csv"
#define EVENTS_FILE "events.csv"
#define SEU_DATA_FILE "seu_data.bin"
#define STDOUT_FILE "stdout.txt"
#define STDERR_FILE "stderr.txt"
// #define SYNC_BETWEEN_KERNELS
#define MAX_CONCURRENT_KERNELS 4

// #define DO_CUBLAS_VERIFICATION

// /* Global variable to interrupt the loop later on */
static volatile int wait_trigger = 1;
unsigned long trigger_timestamp;

// /* Variable for trigger GPIO pin */
int trigger_gpio = 7;

/* Global variables */
time_t time_convert;
uint64_t app_start_time, app_end_time, kernel_launch_start_time, kernel_launch_end_time;
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
std::string file_timestamp;
std::string results_file_path;
std::string events_file_path;
std::string seu_data_file_path;

/* Global variables for CUDA events */
float sanity_time_ms = 0.0f;
float* kernel_time_ms = NULL;

/* ------------------------ */
/* File Operation Functions */
/* ------------------------ */

void read_matrix_from_file(const char* filename, float* matrix, int matrix_size) {
    std::fstream file(filename, std::ios::in | std::ios::binary);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file: %s\n", filename);
        file.close();
        exit(EXIT_FAILURE);
    }
    file.read((char *) matrix, sizeof(float) * matrix_size * matrix_size);
    file.close();
    fprintf(stdout,"Read matrix from %s\n", filename);
}

void write_header_to_results_file(const std::string& file_path) {

    /* ----------------------------- */
    /* Write header for results file */
    /* ----------------------------- */

    // Open the file in output mode to write the header
    std::fstream results_fd(file_path, std::ios::out | std::ios::binary);
    if (!results_fd.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", file_path.c_str());
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
        << "app_start_time" << ","
        << "app_end_time" << ","
        // << "kernel_launch_start_time" << ","
        // << "kernel_launch_end_time" << ","
        << "trigger_timestamp" << ","
        << "trigger_signal_enabled" << ","
        << "seu_count_total" << ","
        << "kernel_number" << "\n";
    // fprintf(stdout,results_header.str().c_str());
    results_fd << results_header.str();

    // Close the file after writing
    results_fd.close();
    fprintf(stdout,"Wrote header to %s\n", file_path.c_str());
}

void write_header_to_events_file(const std::string& file_path) {
    
    /* ----------------------------- */
    /* Write header for events file */
    /* ----------------------------- */

    // Open the file in output mode to write the header
    std::fstream events_fd(file_path, std::ios::out | std::ios::binary);
    if (!events_fd.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", file_path.c_str());
        events_fd.close();
        exit(EXIT_FAILURE);
    }
    
    // Write header for events file
    std::stringstream events_header;
    events_header
        << "repetition" << ","
        << "seu_count" << ","
        << "kernel_duration" << ","
        << "kernel_start_timestamp" << ","
        << "kernel_end_timestamp" << ","
        << "trigger_rise_timestamp" << ","
        << "trigger_fall_timestamp" << "\n";
    // fprintf(stdout,events_header.str().c_str());
    events_fd << events_header.str();

    // Close the file after writing
    events_fd.close();
    fprintf(stdout,"Wrote header to %s\n", file_path.c_str());
}

void write_results_to_file(
        const std::string& file_path,
        int matrix_size,
        char* file_A,
        char* file_B,
        char* file_C,
        uint64_t app_start_time,
        uint64_t app_end_time,
        // uint64_t kernel_launch_start_time,
        // uint64_t kernel_launch_end_time,
        unsigned long trigger_timestamp,
        bool trigger_signal_enabled,
        unsigned int seu_count_total,
        int kernel_number
    ) {

    // Open the file in append mode to add results
    std::fstream file(file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", file_path.c_str());
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
        << app_start_time << ","
        << app_end_time << ","
        // << kernel_launch_start_time << ","
        // << kernel_launch_end_time << ","
        << trigger_timestamp << ","
        << trigger_signal_enabled << ","
        << seu_count_total << ","
        << kernel_number << "\n";

    file << result_line.str();

    // Close the file after writing
    file.close();
    fprintf(stdout,"Saved results to %s\n", file_path.c_str());
}

void write_events_to_file(
        const std::string& file_path,
        int repetition, // kernel info
        unsigned int seu_count, // kernel info
        uint64_t kernel_duration, // kernel info
        uint64_t kernel_start_timestamp, // kernel info
        uint64_t kernel_end_timestamp, // kernel info
        unsigned long trigger_rise_timestamp, // trigger info
        unsigned long trigger_fall_timestamp // trigger info
    ) {

    // Open the file in append mode to add results
    std::fstream file(file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", file_path.c_str());
        file.close();
        exit(EXIT_FAILURE);
    }

    // Write the events to csv file
    std::stringstream result_line;
    result_line
        << repetition << ","
        << seu_count << ","
        << kernel_duration << ","
        << kernel_start_timestamp << ","
        << kernel_end_timestamp << ","
        << trigger_rise_timestamp << ","
        << trigger_fall_timestamp << "\n";
    file << result_line.str();

    // Close the file after writing
    file.close();
}

void write_seu_data_to_file(
    const std::string& file_path,
    int repetition,
    int seu_count,
    int* error_col_index,
    int* error_row_index,
    float* error_value
)
{

    // Open the file in append mode to add results
    std::fstream file(file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!file.is_open()) {
        fprintf(stderr, "Error opening file for writing: %s\n", file_path.c_str());
        file.close();
        exit(EXIT_FAILURE);
    }

    // Write the error indices and values
    file.write(reinterpret_cast<const char*>(&repetition), sizeof(int));
    file.write(",", sizeof(char)); // Separator
    file.write(reinterpret_cast<const char*>(&seu_count), sizeof(unsigned int));
    file.write(",", sizeof(char)); // Separator
    if (seu_count > 0)
    {
        file.write(reinterpret_cast<const char*>(error_col_index), sizeof(int) * seu_count);
        file.write(reinterpret_cast<const char*>(error_row_index), sizeof(int) * seu_count);
        file.write(reinterpret_cast<const char*>(error_value), sizeof(float) * seu_count);
    }

    // Close the file after writing
    file.close();
}

/* ------------------- */
/* Timestamp Functions */
/* ------------------- */

void timestamp_kernel(void* data)
{
    uint64_t* timestamp = static_cast<uint64_t*>(data);
    *(timestamp) = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

/* ------------------------ */
/* Error Counting Functions */
/* ------------------------ */

void count_seu_errors(float* C, float* C_ref, int M, int N, unsigned int* seu_count)
{
    *seu_count = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            (*seu_count)++;
            // if (*seu_count <= 10) { // Print details for the first 10 errors
            //     fprintf(stdout,"Mismatch at index %d: GPU result = %e, Reference = %e\n", i, C[i], C_ref[i]);
            // }
        }
    }
}

void get_seu_indices(float* C, float* C_ref, int M, int N, int* error_row_idx, int* error_col_idx, float* error_value)
{
    int error_idx = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            error_row_idx[error_idx] = i % M;
            error_col_idx[error_idx] = i / M;
            error_value[error_idx] = C[i];
            error_idx++;
        }
    }
}

/* --------------------------- */
/* Beam Line Trigger Functions */
/* --------------------------- */

// /* Function to handle GPIO interrupt */
void beam_line_trigger_handler() {
    time_t time = static_cast<time_t>(trigger_timestamp/1e9); // Convert nanoseconds to seconds
    fprintf(stdout,"Trigger signal timestamp: %lu [us], %s\n", (unsigned long)(trigger_timestamp/1e3), ctime(&time));
    wait_trigger = 0;
}

/* ---------------------- */
/* Exit Handler Functions */
/* ---------------------- */

// // Signal handler for Ctrl+C
// void signal_handler(int signum) {
//     fprintf(stdout,"User pressed Ctrl+C. Stopping wait for beam line trigger if waiting...\n");
//     wait_trigger = 0;
// }

void atexit_handler() {
    fprintf(stdout,"Cleaning up resources...\n");

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
    fprintf(stdout,"\nProgram exit timestamp: %lu [us], %s", exit_time, ctime(&time_convert));
    fprintf(stdout, "\n--------------------------------------------------------------------------\n");

    exit(0);
}

/* ---------------- */
/* Logger Functions */
/* ---------------- */

// void logger_info(char* message)
// {
//     // Redirect stdout and stderr to files
//     fprintf(stdout,"%s", message);
//     fprintf(STDOUT_FILE, "%s", message);
// }

// void logger_err(char* message)
// {
//     fprintf(stderr,"%s", message);
//     fprintf(STDERR_FILE, "%s", message);
// }

/* ------------- */
/* Main Function */
/* ------------- */

int main(int argc, char **argv){
    fprintf(stdout, "\n--------------------------------------------------------------------------\n");
    // Print start timestamp
    app_start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(app_start_time/1e6); // Convert microseconds to seconds
    fprintf(stdout,"Program start timestamp: %lu [us], %s", app_start_time, ctime(&time_convert));

    // Register atexit handler to ensure it gets called on normal exit
    const int result = std::atexit(atexit_handler); // Handler will be called
  
    if (result != 0)
    {
        std::cerr << "atexit registration failed\n";
        return EXIT_FAILURE;
    }

    // Inform user if ABFT error correction is disabled
    #ifdef DISABLE_ERROR_CORRECTION
    fprintf(stdout,"\n----------------------------------------\n");
    fprintf(stdout,"!! ABFT error correction is DISABLED !!");
    fprintf(stdout,"\n----------------------------------------\n\n");
    #endif

    if (argc < 10 + 1)
    {
        fprintf(stdout,"Expected 10 arguments: Matrix start size, end size, step size, kernel start number, end number, repetitions, enable_trigger_signal, input folder, output folder, experiment number. Some arguments are missing! Exiting...\n");
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

    // Prepare file names and paths
    std::tm* f_ts = std::localtime(&time_convert);
    std::ostringstream ss;
    ss << std::put_time(f_ts, "%Y-%m-%d_%a_%H:%M:%S");
    file_timestamp = ss.str();
    results_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(RESULTS_FILE);
    events_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(EVENTS_FILE);
    seu_data_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(SEU_DATA_FILE);


    if (enable_trigger_signal and getuid() != 0)
    {
        fprintf(stdout,"Trigger signal reception is enabled, but the program is not running with root privileges. Please run as root or disable trigger signal reception. Exiting...\n");
        exit(-1);
    }

    // Initialization   
    srand(10);
    float alpha = 1.0;
    float beta = 0.0;       
    int M, N, K;    
    M = MAX_SIZE; N = MAX_SIZE;  K = MAX_SIZE;              
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
    write_header_to_results_file(results_file_path);
    write_header_to_events_file(events_file_path);

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

    // Timestamp start of kernel with cuda event
    cudaEvent_t sanity_start, sanity_stop;
    cudaEventCreate(&sanity_start);
    cudaEventCreate(&sanity_stop);

    // Sanity check with ABFT kernel
    printf("Starting sanity check kernel...\n");
    dim3 blockDim(64);  
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    cudaDeviceSynchronize();
    cudaEventRecord(sanity_start);
    ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);

    // Timestamp end of kernel with cuda event
    cudaEventRecord(sanity_stop);
    cudaEventSynchronize(sanity_stop);
    cudaEventElapsedTime(&sanity_time_ms, sanity_start, sanity_stop);
    fprintf(stdout,"Sanity check kernel execution time: %f ms\n", sanity_time_ms);

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
    float error_value[seu_count];
    get_seu_indices(C, C_ref, M, N, error_row_idx, error_col_idx, error_value);

    write_events_to_file(
        output_folder,
        app_start_time,
        error_row_idx,
        error_col_idx,
        NULL, // (not applicable here)
        NULL, // (not applicable here)
        error_value,
        seu_count,
        -1 // repetition (not applicable here)
    );

    fprintf(stdout,"Sanity check completed. SEUs prior to beam detected: %u\n", seu_count);

    /* -------------------------------------- */
    /* Wait for trigger signal from beam line */
    /* -------------------------------------- */

    if (enable_trigger_signal)
    {
        int Init = gpioInitialise();
        if (Init < 0) {
            fprintf(stdout,"Jetgpio initialisation failed. Error code:  %d\n", Init);
            exit(1);
        }
        int stat = gpioSetMode(trigger_gpio, JET_INPUT); // Set GPIO pin as input
        if (stat < 0)
        {
            fprintf(stdout,"Failed to set GPIO pin mode. Error code: %d\n", stat);
            exit(1);
        }
        // Set up interrupt handler for falling edge on the trigger GPIO pin
        stat = gpioSetISRFunc(trigger_gpio, FALLING_EDGE, 10 /* us */, &trigger_timestamp, &beam_line_trigger_handler);
        if (stat < 0)
        {
            fprintf(stdout,"Failed to set GPIO alert function. Error code: %d\n", stat);
            exit(1);
        }
        fprintf(stdout,"Waiting for trigger signal from beam line...\n");
        while (wait_trigger) {
            // Wait for the GPIO pin to go high
        }
        fprintf(stdout,"Trigger received!\n");
    }
    else
    {
        fprintf(stdout,"Trigger signal reception is DISABLED. Starting kernel execution immediately...\n");
    }
    
    fprintf(stdout,"Starting kernel execution...\n");
    fprintf(stdout,"Matrix size: %d, Kernel number: %d, Repetitions: %d\n", MAX_SIZE, kernel_number, repeat_kernel);
    // Print kernel start timestamp
    kernel_launch_start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(kernel_launch_start_time/1e6); // Convert microseconds to seconds
    fprintf(stdout,"Kernel launch start timestamp: %lu [us], %s", kernel_launch_start_time, ctime(&time_convert));

    /* -------------------- */
    /* Run selected kernels */
    /* -------------------- */

    #ifdef DO_CUBLAS_VERIFICATION
    fprintf(stdout,"Start cublas sgemm\n");
    cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC_BLAS, M);
    #endif

    // Setup cuda events for timing kernels
    cudaEvent_t kernel_start[repeat_kernel], kernel_stop[repeat_kernel];
    cudaStream_t kernel_stream[repeat_kernel];
    for (int i = 0; i < repeat_kernel; i++) {
        cudaEventCreate(&kernel_start[i]);
        cudaEventCreate(&kernel_stop[i]);
        cudaStreamCreateWithFlags(&kernel_stream[i], cudaStreamNonBlocking);
    }
    kernel_time_ms = (float*)malloc(sizeof(float) * repeat_kernel);

    // Run the selected kernel(s) for the specified number of repetitions
    // for (int repetition = 0; repetition < repeat_kernel; repetition++){
    //     cudaEventRecord(kernel_start[repetition], kernel_stream[repetition]);
    //     if(kernel_number == 11){
    //         dim3 blockDim(64);  
    //         dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
    //         #ifdef SYNC_BETWEEN_KERNELS
    //         cudaDeviceSynchronize();
    //         #endif
    //         ft_sgemm_small<<<gridDim, blockDim, 0, kernel_stream[repetition]>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta);
    //     }  
    //     else if(kernel_number == 12){
    //         dim3 blockDim(64);  
    //         dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    //         #ifdef SYNC_BETWEEN_KERNELS
    //         cudaDeviceSynchronize();
    //         #endif
    //         ft_sgemm_medium<<<gridDim, blockDim, 0, kernel_stream[repetition]>>>(M, N, K, dA, dB, dC+repetition*MAX_SIZE*MAX_SIZE, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);  
    //     }
    //     else
    //     {
    //         fprintf(stdout,"Kernel number %d is not implemented. Exiting...\n", kernel_number);
    //         exit(-1);
    //     }
    //     cudaEventRecord(kernel_stop[repetition], kernel_stream[repetition]);

    //     /* Pause between kernels (used for debugging & current measurements) - START */
    //     // cudaDeviceSynchronize();
    //     // time_t kernel_timestamp = time(nullptr);
    //     // fprintf(stdout,"[%s] Repetition %d finisched.\n", ctime(&kernel_timestamp), repetition);
    //     // std::this_thread::sleep_for(std::chrono::seconds(1));
    //     /* Pause between kernels (used for debugging & current measurements) - END */
    // }

    // CONCURRENT STREAM SCHEDULING - START

    /* --------------------------------------- */
    /* Concurrent streaming scheduling - SETUP */
    /* --------------------------------------- */
    int completed_streams = 0;
    bool stream_completed[repeat_kernel] = {false}; // Track completion status of each stream
    bool stream_running[repeat_kernel] = {false}; // Track running status of each stream
    uint64_t kernel_exec_start_time[repeat_kernel] = {0}; // Track start time of each kernel
    uint64_t kernel_exec_end_time[repeat_kernel] = {0}; // Track end time of each kernel
    int active_streams[MAX_CONCURRENT_KERNELS]; // Track active streams
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        active_streams[i] = -1;
    }

    /* -------------------------------------- */
    /* Concurrent streaming scheduling - LOOP */
    /* -------------------------------------- */
    while (completed_streams < repeat_kernel) {
    
        /* ------------------------- */
        /* Check for any CUDA errors */
        /* ------------------------- */
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stdout, "ERROR detected: (%s) %s\n", cudaGetErrorName(err), cudaGetErrorString(err));

            // Report which streams were active at the time of the error
            fprintf(stdout, "Current active streams: ");
            for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
                if (active_streams[i] != -1) {
                    fprintf(stdout,"%d,", active_streams[i] + 1);
                }
            }
            fprintf(stdout, "\n");

            // Save error to events file
            // TODO

            // Save which streams were cancelled to events file
            // TODO: Mark all active streams as cancelled

            // Cleanup and destroy all streams
            for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
                if (active_streams[i] != -1) {
                    fprintf(stdout,"Repetition %d cancelled due to some CUDA error.\n", active_streams[i] + 1);

                    // Cleanup the stream after cancellation
                    cudaEventDestroy(kernel_start[active_streams[i]]);
                    cudaEventDestroy(kernel_stop[active_streams[i]]);
                    cudaStreamDestroy(kernel_stream[active_streams[i]]);

                    // Update state variables for the cancelled stream
                    stream_running[active_streams[i]] = false; // Mark this stream as no longer running
                    stream_completed[active_streams[i]] = true; // Mark this stream as not completed
                    active_streams[i] = -1; // Mark this stream as inactive
                    completed_streams++;
                }
            }

            // Exit application for unrecoverable CUDA errors
            if (err == cudaErrorIllegalAddress)
            {
                fprintf(stdout,"NOTE: An illegal memory access is an unrecoverable error.\n");

                // Reset the device to recover from the error
                cudaDeviceReset();
                fprintf(stdout,"CUDA device was reset. Exiting application.\n");

                // Exit application
                exit(-1);
            }
        }

        /* ---------------------------------- */
        /* Check which streams have completed */
        /* ---------------------------------- */
        for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
            if (active_streams[i] == -1) continue; // Skip inactive streams
            if (cudaStreamQuery(kernel_stream[active_streams[i]]) == cudaSuccess) {
                // cudaStreamSynchronize(kernel_stream[active_streams[i]]);
                timestamp_kernel(&(kernel_exec_end_time[active_streams[i]])); // Record end time
                cudaEventElapsedTime(&kernel_time_ms[active_streams[i]], kernel_start[active_streams[i]], kernel_stop[active_streams[i]]);
                fprintf(stdout,"Repetition %d finished. Exec time - CPU: %lu us, \tCUDA: %d us\n", active_streams[i] + 1, kernel_exec_end_time[active_streams[i]] - kernel_exec_start_time[active_streams[i]], int(kernel_time_ms[active_streams[i]]*1000.0));

                // Copy results from device to host for this repetition
                cudaMemcpyAsync(C + active_streams[i] * M * N, dC + active_streams[i] * M * N, sizeof(float) * M * N, cudaMemcpyDeviceToHost, kernel_stream[active_streams[i]]);

                // Cleanup the stream after completion
                cudaEventDestroy(kernel_start[active_streams[i]]);
                cudaEventDestroy(kernel_stop[active_streams[i]]);
                cudaStreamDestroy(kernel_stream[active_streams[i]]);

                // Update state variables
                stream_completed[active_streams[i]] = true;
                stream_running[active_streams[i]] = false;
                active_streams[i] = -1; // Mark this stream as inactive
                completed_streams++;

                // Check for SEU errors for this repetition
                // TODO

                // Save results to file for this repetition
                // TODO

                // Save events to file for this repetition
                // TODO
            }
        }

        /* --------------------------------------------- */
        /* Schedule new kernels for any inactive streams */
        /* --------------------------------------------- */
        for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
            // Find an inactive stream slot
            if (active_streams[i] == -1) {
                int new_active_stream = -1;

                // Check which stream needs scheduling
                for (int j = 0; j < repeat_kernel; j++) {
                    if (!stream_completed[j] && !stream_running[j]) {
                        active_streams[i] = j;
                        new_active_stream = j;
                        break;
                    }
                }

                // Launch the kernel for the new active stream if found
                if (new_active_stream != -1)
                {
                    // Mark this stream as running
                    stream_running[new_active_stream] = true; // Mark this stream as running

                    // Prepare output matrix memory for this repetition
                    // TODO
                    // CUDA_CALLER(cudaMemcpy(dC + new_active_stream * MAX_SIZE * MAX_SIZE, C + new_active_stream * MAX_SIZE * MAX_SIZE, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));

                    // Launch the kernel for this repetition
                    timestamp_kernel(&(kernel_exec_start_time[new_active_stream])); // Record start time
                    cudaEventRecord(kernel_start[new_active_stream], kernel_stream[new_active_stream]);
                    if(kernel_number == 11){
                        dim3 blockDim(64);  
                        dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
                        #ifdef SYNC_BETWEEN_KERNELS
                        cudaDeviceSynchronize();
                        #endif
                        ft_sgemm_small<<<gridDim, blockDim, 0, kernel_stream[new_active_stream]>>>(M, N, K, dA, dB, dC+new_active_stream*MAX_SIZE*MAX_SIZE, alpha, beta);
                    }  
                    else if(kernel_number == 12){
                        dim3 blockDim(64);  
                        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
                        #ifdef SYNC_BETWEEN_KERNELS
                        cudaDeviceSynchronize();
                        #endif
                        ft_sgemm_medium<<<gridDim, blockDim, 0, kernel_stream[new_active_stream]>>>(M, N, K, dA, dB, dC+new_active_stream*MAX_SIZE*MAX_SIZE, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);  
                    }
                    else
                    {
                        fprintf(stdout,"Kernel number %d is not implemented. Exiting...\n", kernel_number);
                        exit(-1);
                    }
                    cudaEventRecord(kernel_stop[new_active_stream], kernel_stream[new_active_stream]);

                    fprintf(stdout,"Repetition %d started.\n", new_active_stream + 1);
                }
            }
        }
    }
    // CONCURRENT STREAM SCHEDULING - END


    // Check launch error
    cudaError_t launchErr = cudaGetLastError();
    if (launchErr != cudaSuccess) {
        fprintf(stderr, "[Kernel Launch Error] %s\n", cudaGetErrorString(launchErr));
    }

    // Check execution error
    cudaError_t syncErr = cudaDeviceSynchronize();
    if (syncErr != cudaSuccess) {
        fprintf(stderr, "[Kernel Execution Error] %s\n", cudaGetErrorString(syncErr));
    }

    fprintf(stdout,"[Kernel Launch Completed Successfully]\n");
    
    // Print start timestamp
    kernel_launch_end_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(kernel_launch_end_time/1e6); // Convert microseconds to seconds
    fprintf(stdout,"\nKernel launch end timestamp: %lu [us], %s", kernel_launch_end_time, ctime(&time_convert));
    fprintf(stdout,"Kernel launch duration: %lu [us]\n", kernel_launch_end_time - kernel_launch_start_time);

    // Copy results from device to host memory
    // cudaMemcpy(C, dC, sizeof(float) * M * N * repeat_kernel, cudaMemcpyDeviceToHost);

    #ifdef DO_CUBLAS_VERIFICATION
    cudaMemcpy(C_BLAS, dC_BLAS, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
    if (!verify_matrix(C_BLAS, C, M, N)) { 
        fprintf(stdout,"kernel %d failed to pass the correctness verification against NVIDIA cuBLAS. Exited.\n", kernel_number);
        // exit(-3);  
    }    
    fflush(stdout);              
    fprintf(stdout,"kernel %d finish verified!\n", kernel_number);      
    cudaDeviceSynchronize();
    #endif

    /* ---------------- */
    /* Count SEU errors */
    /* ---------------- */

    unsigned int seu_count_total = 0;
    for (int repetition = 0; repetition < repeat_kernel; repetition++) {
        fprintf(stdout,"Repetition %d - ", repetition + 1);

        // Count SEU errors by comparing C with C_ref
        unsigned int seu_count = 0;
        count_seu_errors(C + repetition * M * N, C_ref, M, N, &seu_count);
        fprintf(stdout,"SEU errors detected: %u \t(total: %u)\n", seu_count, seu_count_total + seu_count);

        // Get SEU error locations (row and column indices)
        int error_row_idx[seu_count];
        int error_col_idx[seu_count];
        float error_value[seu_count];
        get_seu_indices(C + repetition * M * N, C_ref, M, N, error_row_idx, error_col_idx, error_value);

        write_events_to_file(
            output_folder,
            app_start_time,
            NULL, // sanity_error_row_index (not applicable here)
            NULL, // sanity_error_col_index (not applicable here)
            error_row_idx,
            error_col_idx,
            error_value,
            seu_count,
            repetition
        );

        // Accumulate SEU counts over all kernel repetitions
        seu_count_total += seu_count;
    }

    fprintf(stdout,"Total SEU errors across all repetitions: %d\n", seu_count_total);

    app_end_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(app_end_time/1e6); // Convert microseconds to seconds
    fprintf(stdout,"\nProgram end timestamp: %lu [us], %s", app_end_time, ctime(&time_convert));
    fprintf(stdout,"Program duration: %lu [us]\n", app_end_time - app_start_time);

    write_results_to_file(
        results_file_path,
        MAX_SIZE,
        file_A_name,
        file_B_name,
        file_C_name,
        app_start_time,
        app_end_time,
        // kernel_launch_start_time,
        // kernel_launch_end_time,
        trigger_timestamp,
        enable_trigger_signal,
        seu_count_total,
        kernel_number
    );

    exit(0); // Exit normally to ensure atexit handler is called
}
