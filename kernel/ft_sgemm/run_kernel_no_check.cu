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

#include <thread>
#include <unistd.h>
#include <csignal>
#include <iomanip>

#define RESULTS_FILE "results.csv"
#define EVENTS_FILE "events.csv"
#define SEU_DATA_FILE "seu_data.bin"
#define STDOUT_FILE "stdout.txt"
#define STDERR_FILE "stderr.txt"
// #define SYNC_BETWEEN_KERNELS
#define MAX_CONCURRENT_KERNELS 2

// #define DO_CUBLAS_VERIFICATION

// /* Global variable to interrupt the loop later on */
static volatile int wait_trigger = 1;
unsigned long trigger_timestamp;

// /* Variable for trigger GPIO pin */
int trigger_gpio = 7;

/* Global variables */
time_t time_convert;
uint64_t app_start_time, app_end_time, kernel_launch_start_time, kernel_launch_end_time;
float *A = NULL, *B = NULL, *C_ref = NULL, *C_BLAS = NULL;
float **C = NULL;
float *check_C_col = NULL, *check_C_row = NULL;
float *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC_BLAS = NULL;
float **dC = NULL;
float *dcheck_C_col = NULL, *dcheck_C_row = NULL;
const char* folder_name;
const char* output_folder;
const char* experiment_name;
bool enable_sanity_check;
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

void count_seu_errors(float* C, float* C_ref, int M, int N, int* seu_count, int* error_row_idx, int* error_col_idx, float* error_value)
{
    *seu_count = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            // Save the row and column indices and the value of the error
            error_row_idx[*seu_count] = i % M;
            error_col_idx[*seu_count] = i / M;
            error_value[*seu_count] = C[i];

            // Increment the SEU count
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

// Signal handler for Ctrl+C
void signal_handler(int signum) {
    if (wait_trigger)
    {
        fprintf(stdout,"User pressed Ctrl+C. Stopping wait for beam line trigger if waiting...\n");
        wait_trigger = 0;
    }
    else
    {
        fprintf(stdout,"User pressed Ctrl+C. Exiting program...\n");
        exit(0);
    }
}

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

    // reset CUDA device to cleanup
    cudaDeviceReset();

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

    // Register signal handler for Ctrl+C (SIGINT)
    std::signal(SIGINT, signal_handler);
  
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

    if (argc < 11 + 1)
    {
        fprintf(stdout,"Expected 11 arguments: Matrix start size, end size, step size, kernel start number, end number, repetitions, enable_trigger_signal, input folder, output folder, experiment number, enable_sanity_check. Some arguments are missing! Exiting...\n");
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
    enable_sanity_check = atoi(argv[11]);

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

    // SEU count variables
    int seu_count = 0;
    int seu_count_total = 0;
    int error_row_idx[MAX_SIZE * MAX_SIZE];
    int error_col_idx[MAX_SIZE * MAX_SIZE];
    float error_value[MAX_SIZE * MAX_SIZE];

    printf("Allocating host memory...\n");
    A = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    B = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    // C = (float *)calloc(MAX_SIZE * MAX_SIZE * repeat_kernel, sizeof(float));
    C_ref = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    check_C_col = (float *)calloc(MAX_SIZE, sizeof(float));
    check_C_row = (float *)calloc(MAX_SIZE, sizeof(float));

    // Allocate host memory for output matrices for each repetition
    C = (float **)malloc(sizeof(float *) * MAX_CONCURRENT_KERNELS);
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        C[i] = (float *)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));
    }

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
    // CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * MAX_SIZE * MAX_SIZE * repeat_kernel));
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));     
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyHostToDevice));

    // Allocate device memory for output matrices for each repetition
    dC = (float**)malloc(sizeof(float*) * MAX_CONCURRENT_KERNELS);
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        cudaMalloc((void**) &(dC[i]), sizeof(float) * MAX_SIZE * MAX_SIZE);
    }

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

    if (enable_sanity_check)
    {
        // Sanity check with CUBLAS
        // cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC, M);

        // Allocate device memory for sanity check results
        CUDA_CALLER(cudaMalloc((void**) &(dC[0]), sizeof(float) * MAX_SIZE * MAX_SIZE));
        C[0] = (float*)calloc(MAX_SIZE * MAX_SIZE, sizeof(float));

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
        uint64_t sanity_kernel_start_time;
        timestamp_kernel(&sanity_kernel_start_time);
        ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC[0], alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);

        // Timestamp end of kernel with cuda event
        cudaEventRecord(sanity_stop);
        cudaEventSynchronize(sanity_stop);
        uint64_t sanity_kernel_end_time;
        timestamp_kernel(&sanity_kernel_end_time);
        cudaEventElapsedTime(&sanity_time_ms, sanity_start, sanity_stop);
        fprintf(stdout,"Sanity check kernel execution time: %f ms\n", sanity_time_ms);

        cudaDeviceSynchronize();
        printf("Copying sanity check results to host memory...\n");
        CUDA_CALLER(cudaMemcpy(C[0], dC[0], sizeof(float) * MAX_SIZE * MAX_SIZE, cudaMemcpyDeviceToHost)); // TODO: make async with stream

        // Inject error
        C[0][7*M + 15] *= 0.1; // Inject an error in the first element of C for testing

        // Count SEU errors by comparing C with C_ref
        printf("Calcuating SEU count for sanity check...\n");
        count_seu_errors(C[0], C_ref, M, N, &seu_count, error_row_idx, error_col_idx, error_value);

        write_events_to_file(
            events_file_path,
            -1,
            seu_count,
            (uint64_t)(sanity_time_ms * 1000), // Convert ms to us, CUDA event timing
            sanity_kernel_start_time, // Start timestamp of sanity check kernel, OS timestamp timing
            sanity_kernel_end_time, // End timestamp of sanity check kernel, OS timestamp timing
            0,
            0
        );

        write_seu_data_to_file(
            seu_data_file_path,
            -1,
            seu_count,
            error_col_idx,
            error_row_idx,
            error_value
        );

        fprintf(stdout,"Sanity check completed. SEUs prior to beam detected: %d\n", seu_count);
    }

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

    // Setup cuda events and streams for kernel scheduling
    cudaEvent_t kernel_start[MAX_CONCURRENT_KERNELS], kernel_stop[MAX_CONCURRENT_KERNELS];
    cudaStream_t kernel_stream[MAX_CONCURRENT_KERNELS];
    cudaStream_t mem_stream;
    cudaStreamCreateWithFlags(&mem_stream, cudaStreamNonBlocking);
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        cudaEventCreate(&kernel_start[i]);
        cudaEventCreate(&kernel_stop[i]);
        cudaStreamCreateWithFlags(&kernel_stream[i], cudaStreamNonBlocking);
    }
    kernel_time_ms = (float*)malloc(sizeof(float) * repeat_kernel);

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
                    fprintf(stdout,"%d,", active_streams[i]);
                }
            }
            fprintf(stdout, "\n");

            // Save error to events file
            // TODO

            // Cleanup and destroy all streams
            for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
                if (active_streams[i] != -1) {
                    fprintf(stdout,"Repetition %d cancelled due to some CUDA error.\n", active_streams[i]);

                    // Check for SEU errors for the active streams
                    // TODO

                    // Save results to file for the active streams
                    // TODO

                    // Save events to file for the active streams
                    // TODO

                    // Save SEU data to file for the active streams
                    // TODO

                    // Save which streams were cancelled to events file
                    // TODO: Mark all active streams as cancelled

                    // LAST STEP - Update state variables for the cancelled stream
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

            // Handle one completed stream at a time
            if (cudaEventQuery(kernel_stop[i]) == cudaSuccess) {
                timestamp_kernel(&(kernel_exec_end_time[active_streams[i]])); // Record end time
                cudaEventElapsedTime(&kernel_time_ms[active_streams[i]], kernel_start[i], kernel_stop[i]);
                fprintf(stdout,"%d - CPU: %lu us, CUDA: %d us, ", active_streams[i], kernel_exec_end_time[active_streams[i]] - kernel_exec_start_time[active_streams[i]], int(kernel_time_ms[active_streams[i]]*1000.0));

                // Copy results from device to host for this repetition
                cudaMemcpyAsync(C[i], dC[i], sizeof(float) * M * N, cudaMemcpyDeviceToHost, mem_stream);

                // Add error injection for testing
                // if ( active_streams[i] == 7 ) // Inject error for the 8th repetition (index 7)
                // {
                //     C[i][4*M + 13] *= 0.1; // Introduce a small error in the first element
                // }

                // Check for SEU errors for this repetition
                count_seu_errors(C[i], C_ref, M, N, &seu_count, error_row_idx, error_col_idx, error_value);
                // fprintf(stdout,"SEU: %d (tot. %d)\n", seu_count, seu_count_total + seu_count);
                // if (seu_count > 0) {
                //     fprintf(stdout, "c: %d, r: %d, v: %e\n", error_col_idx[0], error_row_idx[0], error_value[0]);
                // }
                fprintf(stdout, "\n");

                // Save events to file for this repetition
                write_events_to_file(
                    events_file_path,
                    active_streams[i],
                    seu_count,
                    (uint64_t)(kernel_time_ms[active_streams[i]] * 1000), // Convert ms to us, CUDA Event timing
                    kernel_exec_start_time[active_streams[i]], // Start timestamp of this kernel, OS timestamp timing
                    kernel_exec_end_time[active_streams[i]], // End timestamp of this kernel, OS timestamp timing
                    0,
                    0
                );

                // Save seu data to file for this repetition
                write_seu_data_to_file(
                    seu_data_file_path,
                    active_streams[i],
                    seu_count,
                    error_col_idx,
                    error_row_idx,
                    error_value
                );

                // Accumulate SEU counts over all kernel repetitions
                seu_count_total += seu_count;

                // Save results to file for this repetition
                // TODO

                // LAST STEP - Update state variables
                stream_completed[active_streams[i]] = true;
                stream_running[active_streams[i]] = false;
                active_streams[i] = -1; // Mark this stream as inactive
                completed_streams++;

                break; // Only handle one completed stream each iteration, so that new kernels can be scheduled
            }
        }

        /* --------------------------------------------- */
        /* Schedule new kernels for any inactive streams */
        /* --------------------------------------------- */
        for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
            // Find an inactive stream slot
            if (active_streams[i] == -1) {
                int new_repetition = -1;

                // Check which stream needs scheduling
                for (int j = 0; j < repeat_kernel; j++) {
                    if (!stream_completed[j] && !stream_running[j]) {
                        active_streams[i] = j;
                        new_repetition = j;
                        break;
                    }
                }

                // Launch the kernel for the new active stream if found
                if (new_repetition != -1)
                {
                    // Mark this stream as running
                    stream_running[new_repetition] = true; // Mark this stream as running

                    // Launch the kernel for this repetition
                    timestamp_kernel(&(kernel_exec_start_time[new_repetition])); // Record start time
                    cudaEventRecord(kernel_start[i], kernel_stream[i]);
                    if(kernel_number == 11){
                        dim3 blockDim(64);  
                        dim3 gridDim(CEIL_DIV(M, 16), CEIL_DIV(N, 16));
                        #ifdef SYNC_BETWEEN_KERNELS
                        cudaDeviceSynchronize();
                        #endif
                        ft_sgemm_small<<<gridDim, blockDim, 0, kernel_stream[i]>>>(M, N, K, dA, dB, dC[i], alpha, beta);
                    }  
                    else if(kernel_number == 12){
                        dim3 blockDim(64);  
                        dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
                        #ifdef SYNC_BETWEEN_KERNELS
                        cudaDeviceSynchronize();
                        #endif
                        // TODO: Remove after DEBUGGING - START
                        int mem_access_error = 0;
                        // if (new_repetition == 7) mem_access_error = MAX_SIZE * MAX_SIZE * sizeof(float) * repeat_kernel; // Introduce memory access error for the 8th repetition (index 7)
                        // TODO: Remove after DEBUGGING - END
                        ft_sgemm_medium<<<gridDim, blockDim, 0, kernel_stream[i]>>>(M, N, K, dA, dB, dC[i]+mem_access_error, alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);  
                    }
                    else
                    {
                        fprintf(stdout,"Kernel number %d is not implemented. Exiting...\n", kernel_number);
                        exit(-1);
                    }
                    cudaEventRecord(kernel_stop[i], kernel_stream[i]);

                    // fprintf(stdout,"Repetition %d started.\n", new_repetition);
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

    // Clean up CUDA events and streams
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        cudaEventDestroy(kernel_start[i]);
        cudaEventDestroy(kernel_stop[i]);
        cudaStreamDestroy(kernel_stream[i]);
    }

    // Print total SEU errors across all repetitions
    fprintf(stdout,"Total SEU errors across all repetitions: %d\n", seu_count_total);

    // Timestamp end of application
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
