#include <stdio.h>     
#include <cublas_v2.h>        
#include "utils/utils.cuh"            
#define PPP 1
#include <cuda_runtime.h> 
#include <helper_functions.h> 
#include <helper_cuda.h>
#include "kernels.cuh"
// #include <ctime>
#include <fstream>
#include <filesystem>
#include <chrono>
#define multi 20

#include <thread>
#include <unistd.h>
#include <csignal>
#include <iomanip>
#include <sys/wait.h>

#define RESULTS_FILE "results.csv"
#define EVENTS_FILE "events.csv"
#define SEU_DATA_FILE "seu_data.bin"
#define STDOUT_FILE "stdout.txt"
#define STDERR_FILE "stderr.txt"
// #define SYNC_BETWEEN_KERNELS
#define MAX_CONCURRENT_KERNELS 6
#define KERNEL_MAX_TIMEOUT_DURATION (5 * 1000) /* [us] */
#define KERNEL_ALLOWED_MAX_TIMEOUTS 4 /* Maximum number of allowed kernel timeouts before treating as unrecoverable error */

// #define SANITY_CHECK_EVENT_ID -1
#define NULL_EVENT_ID -2
#define BEAM_START_EVENT_ID -10
#define BEAM_STOP_EVENT_ID -11
#define ARDUINO_EVENT_ID -12
#define ERROR_EVENT_ID -99
#define APP_EXIT_EVENT_ID -100

// #define ENABLE_SEU_DATA_LOGGING

/* Global variables */
time_t time_convert;
uint64_t app_start_timestamp = 0, app_end_timestamp = 0, kernel_launch_start_time = 0, kernel_launch_end_time = 0;
float *A = NULL, *B = NULL, *C_ref = NULL, *C_BLAS = NULL;
float **C = NULL;
float *check_C_col = NULL, *check_C_row = NULL;
float *dA = NULL,*dB = NULL, *dC_ref = NULL, *dC_BLAS = NULL;
float **dC = NULL;
float *dcheck_C_col = NULL, *dcheck_C_row = NULL;
const char* folder_name;
const char* output_folder;
const char* experiment_name;
bool enable_beam_signal_wait = false;
bool enable_sanity_check = false;
bool enable_seu_data_logging = false;
char file_A_name[256];
char file_B_name[256];
char file_C_name[256];
std::string file_timestamp;
std::string results_file_path;
std::string events_file_path;
std::string seu_data_file_path;
std::fstream *results_file = NULL;
std::fstream *events_file = NULL;
std::fstream *seu_data_file = NULL;
int matrix_size = 99;
long long int seu_count = 0;
long long int seu_count_total = 0;
int kernel_number = 12; // 12 is used as default kernel number
int repeat_kernel = -1; // Default of repetitions for the kernel execution
int completed_streams = 0;
pid_t pid;
unsigned long spawn_count = 0;

/* Gobal variables for Kernel Execution Loop */
int M, N, K;
float alpha = 1.0;
float beta = 0.0;
// SEU count variables
int* error_row_idx;
int* error_col_idx;
float* error_value;

/* Global variables for CUDA events */
float sanity_time_ms = 0.0f;
float* kernel_time_ms = NULL;
uint64_t gpu_reset_timestamp = 0; // Track timestamp of any GPU reset

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

void write_header_to_results_file(std::fstream *file) {

    /* ----------------------------- */
    /* Write header for results file */
    /* ----------------------------- */
    
    // Write header for results file
    std::stringstream results_header;
    results_header << "matrix_size" << ","
        << "experiment_name" << ","
        << "file_A" << ","
        << "file_B" << ","
        << "file_C" << ","
        << "app_start_timestamp" << ","
        << "app_end_timestamp" << ","
        // << "kernel_launch_start_time" << ","
        // << "kernel_launch_end_time" << ","
        << "beam_signal_wait_enabled" << ","
        << "seu_data_logging_enabled" << ","
        << "sanity_check_enabled" << ","
        << "seu_count_total" << ","
        << "kernel_number" << ","
        << "repeat_kernel" << ","
        << "completed_streams" << "\n";
    // fprintf(stdout,results_header.str().c_str());
    (*file) << results_header.str();
    (*file).flush(); // Ensure everything is written to the file immediately
}

void write_header_to_events_file(std::fstream *file) {
    
    /* ----------------------------- */
    /* Write header for events file */
    /* ----------------------------- */

    // Write header for events file
    std::stringstream events_header;
    events_header
        << "repetition" << ","
        << "seu_count" << ","
        << "kernel_duration" << ","
        << "kernel_start_timestamp" << ","
        << "kernel_end_timestamp" << ","
        << "repetition_cancelled" << ","
        << "cuda_error_code" << ","
        << "error_timestamp" << ","
        << "beam_start_timestamp" << ","
        << "beam_stop_timestamp" << ","
        << "arduino_timestamp" << ","
        << "app_start_timestamp" << ","
        << "app_end_timestamp" << "\n";
    // fprintf(stdout,events_header.str().c_str());
    (*file) << events_header.str();
    (*file).flush(); // Ensure everything is written to the file immediately
}

void write_results_to_file(
        std::fstream *file,
        int matrix_size,
        char* file_A,
        char* file_B,
        char* file_C,
        // uint64_t kernel_launch_start_time,
        // uint64_t kernel_launch_end_time,
        bool beam_signal_wait_enabled,
        bool seu_data_logging_enabled,
        bool sanity_check_enabled,
        long long int seu_count_total,
        int kernel_number,
        int repeat_kernel,
        int completed_streams
    ) {

    // Write the results as a new line in the CSV file
    std::stringstream result_line;
    result_line << matrix_size << ","
        << experiment_name << ","
        << file_A << ","
        << file_B << ","
        << file_C << ","
        << app_start_timestamp << "," // global variable for application start timestamp
        << app_end_timestamp << "," // global variable for application end timestamp
        // << kernel_launch_start_time << ","
        // << kernel_launch_end_time << ","
        << beam_signal_wait_enabled << ","
        << seu_data_logging_enabled << ","
        << sanity_check_enabled << ","
        << seu_count_total << ","
        << kernel_number << ","
        << repeat_kernel << ","
        << completed_streams << "\n";

    (*file) << result_line.str();
    // (*file).flush(); // Ensure everything is written to the file immediately
}

void write_events_to_file(
        std::fstream *file,
        int repetition, // kernel info
        long long int seu_count, // kernel info
        uint64_t kernel_duration, // kernel info
        uint64_t kernel_start_timestamp, // kernel info
        uint64_t kernel_end_timestamp, // kernel info
        bool repetition_cancelled, // repetition info
        int cuda_error_code, // error info
        uint64_t error_timestamp, // error info
        unsigned long beam_start_timestamp, // trigger info
        unsigned long beam_stop_timestamp, // trigger info
        unsigned long arduino_timestamp // arduino info
    ) {

    // Write the events to csv file
    std::stringstream result_line;
    result_line
        << repetition << ","
        << seu_count << ","
        << kernel_duration << ","
        << kernel_start_timestamp << ","
        << kernel_end_timestamp << ","
        << repetition_cancelled << ","
        << cuda_error_code << ","
        << error_timestamp << ","
        << beam_start_timestamp << ","
        << beam_stop_timestamp << ","
        << arduino_timestamp << ","
        << app_start_timestamp << "," // global variable for application start timestamp
        << app_end_timestamp << "\n"; // global variable for application end timestamp
    (*file) << result_line.str();
    // (*file).flush(); // Ensure everything is written to the file immediately
}

void write_seu_data_to_file(
    std::fstream *file,
    int repetition,
    long long int seu_count,
    int* error_col_index,
    int* error_row_index,
    float* error_value
)
{

    // Write the error indices and values
    (*file).write(reinterpret_cast<const char*>(&repetition), sizeof(int));
    (*file).write(",", sizeof(char)); // Separator
    (*file).write(reinterpret_cast<const char*>(&seu_count), sizeof(long long int));
    (*file).write(",", sizeof(char)); // Separator
    if (seu_count > 0)
    {
        (*file).write(reinterpret_cast<const char*>(error_col_index), sizeof(int) * seu_count);
        (*file).write(reinterpret_cast<const char*>(error_row_index), sizeof(int) * seu_count);
        (*file).write(reinterpret_cast<const char*>(error_value), sizeof(float) * seu_count);
    }
    // (*file).flush(); // Ensure everything is written to the file immediately
}

void kernel_event_to_file(
        std::fstream *file,
        int repetition, // kernel info
        long long int seu_count, // kernel info
        uint64_t kernel_duration, // kernel info
        uint64_t kernel_start_timestamp, // kernel info
        uint64_t kernel_end_timestamp, // kernel info
        bool repetition_cancelled // repetition info
)
{
    write_events_to_file(
        file,
        repetition,
        seu_count,
        kernel_duration,
        kernel_start_timestamp,
        kernel_end_timestamp,
        repetition_cancelled,
        0, // cuda_error_code
        0, // error_timestamp
        0, // beam_start_timestamp
        0, // beam_stop_timestamp
        0 // arduino_timestamp
    );
}

void error_event_to_file(
                std::fstream *file,
                int event_id,
                int cuda_error_code,
                uint64_t error_time
)
{
    write_events_to_file(
        file,
        event_id, // repetition
        0, // seu_count
        0, // kernel_duration
        0, // kernel_start_timestamp
        0, // kernel_end_timestamp
        false, // repetition_cancelled
        cuda_error_code, // cuda_error_code
        error_time, // error_timestamp
        0, // beam_start_timestamp
        0, // beam_stop_timestamp
        0 // arduino_timestamp
    );
}

/* ------------------- */
/* Timestamp Functions */
/* ------------------- */

void save_timestamp(void* data)
{
    uint64_t* timestamp = static_cast<uint64_t*>(data);
    *(timestamp) = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

/* ------------------------ */
/* Error Counting Functions */
/* ------------------------ */

void count_seu_errors(float* C, float* C_ref, int M, int N, long long int* seu_count, int* error_row_idx, int* error_col_idx, float* error_value)
{
    *seu_count = 0;
    for (int i = 0; i < M * N; i++) {
        // if (C[i] - C_ref[i] > 1e-2 || C_ref[i] - C[i] > 1e-2) { // Use a tolerance for floating-point comparison like the authors of the ABFT paper if using cuBLAS generated matrix as reference
        if (C[i] != C_ref[i]) {
            #ifdef ENABLE_SEU_DATA_LOGGING
            // Save the row and column indices and the value of the error
            error_row_idx[*seu_count] = i % M;
            error_col_idx[*seu_count] = i / M;
            error_value[*seu_count] = C[i];
            #endif

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

// Signal handler for Ctrl+C
void signal_handler(int signum) {
    fprintf(stdout,"\nUser pressed Ctrl+C. Exiting program...\n");
    exit(0);
}

void atexit_handler() {
    if (pid == 0)
    {
        fprintf(stdout,"\n# -------------------------------------------------- #\n");
        fprintf(stdout,"# --- KERNEL EXECUTION LOOP %d ended. --- #\n", spawn_count);
        fprintf(stdout,"# -------------------------------------------------- #\n\n");
    }
    else
    {
        fprintf(stdout,"\n# ---------------------------------------------- #\n");
        fprintf(stdout,"# --- Exit application and cleanup resources --- #\n");
        fprintf(stdout,"# ---------------------------------------------- #\n\n");

        // Free up memories
        fprintf(stdout, "Release host memory.\n");
        free(A);
        free(B);
        free(C);
        free(C_ref);
        free(check_C_col);
        free(check_C_row);

        // Flush and close files
        fprintf(stdout, "Close files.\n");
        if (results_file != NULL){
            results_file->flush();
            results_file->close();
            // fprintf(stdout,"Saved results to %s\n", results_file_path.c_str());
        }
        if (events_file != NULL) {
            events_file->flush();
            events_file->close();
            // fprintf(stdout,"Saved events to %s\n", events_file_path.c_str());
        }
        #ifdef ENABLE_SEU_DATA_LOGGING
        if (seu_data_file != NULL) {
            seu_data_file->flush();
            seu_data_file->close();
            // fprintf(stdout,"Saved SEU data to %s\n", seu_data_file_path.c_str());
        }
        #endif

        // reset CUDA device to cleanup
        fprintf(stdout, "Reset GPU.\n");
        cudaDeviceReset();

        uint64_t exit_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        time_convert = static_cast<time_t>(exit_time/1e6); // Convert microseconds to seconds
        fprintf(stdout,"\nProgram exit timestamp: %lu [us], %s", exit_time, ctime(&time_convert));
        fprintf(stdout, "\n--------------------------------------------------------------------------\n");
    }
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

/* ------------------------------------- */
/* --- Cleanup KERNEL EXECUTION LOOP --- */
/* ------------------------------------- */
void cleanup_kernel_execution_loop()
{
    // Print summary
    fprintf(stdout, "Completed streams: %d, Total SEU count: %lld\n", completed_streams, seu_count_total);

    // Timestamp end of application
    app_end_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(app_end_timestamp/1e6); // Convert microseconds to seconds
    fprintf(stdout,"\nKernel execution loop end timestamp: %lu [us], %s", app_end_timestamp, ctime(&time_convert));
    fprintf(stdout,"nKernel execution loop duration: %lu [us]\n", app_end_timestamp - app_start_timestamp);

    // Flush files
    if (results_file != NULL){
        // Write results to results file before exiting
        write_results_to_file(
            results_file,
            matrix_size,
            file_A_name,
            file_B_name,
            file_C_name,
            // kernel_launch_start_time,
            // kernel_launch_end_time,
            enable_beam_signal_wait,
            enable_seu_data_logging,
            enable_sanity_check,
            seu_count_total,
            kernel_number,
            repeat_kernel,
            completed_streams
        );

        results_file->flush();
        fprintf(stdout,"Saved results to %s\n", results_file_path.c_str());
    }
    if (events_file != NULL) {
        kernel_event_to_file(
            events_file,
            APP_EXIT_EVENT_ID,
            0, // seu_count
            0, // Convert ms to us, CUDA event timing
            0, // Start timestamp of sanity check kernel, OS timestamp timing
            0, // End timestamp of sanity check kernel, OS timestamp timing
            false // repetition_cancelled
        );
        events_file->flush();
        fprintf(stdout,"Saved events to %s\n", events_file_path.c_str());
    }
    #ifdef ENABLE_SEU_DATA_LOGGING
    if (seu_data_file != NULL) {
        seu_data_file->flush();
        fprintf(stdout,"Saved SEU data to %s\n", seu_data_file_path.c_str());
    }
    #endif

    // reset CUDA device to cleanup
    fprintf(stdout, "Reset GPU.\n");
    cudaDeviceReset();
}

/* ----------------------------- */
/* --- KERNEL EXECUTION LOOP --- */
/* ----------------------------- */
void kernel_execution_loop(){
    /* ---------------------------- */
    /* --- Initialize variables --- */
    /* ---------------------------- */
    // TODO: Reset variables for each kernel execution loop iteration

    /* ----------------------------------- */
    /* --- Allocate CUDA device memory --- */
    /* ----------------------------------- */
    fprintf(stdout,"\n# ----------------------------------- #\n");
    fprintf(stdout,"# --- Allocate CUDA device memory --- #\n");
    fprintf(stdout,"# ----------------------------------- #\n\n");

    int deviceId;          
    cudaGetDevice(&deviceId);            
    cudaDeviceProp props = getDetails(deviceId);
    CUDA_CALLER(cudaMalloc((void**) &dA, sizeof(float) * M * N));
    CUDA_CALLER(cudaMalloc((void**) &dB, sizeof(float) * M * K));  
    // CUDA_CALLER(cudaMalloc((void**) &dC, sizeof(float) * M * N * repeat_kernel));
    CUDA_CALLER(cudaMemcpy(dA, A, sizeof(float) * M * N, cudaMemcpyHostToDevice));     
    CUDA_CALLER(cudaMemcpy(dB, B, sizeof(float) * M * K, cudaMemcpyHostToDevice));

    // Allocate device memory for output matrices for each repetition
    dC = (float**)malloc(sizeof(float*) * MAX_CONCURRENT_KERNELS);
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        cudaMalloc((void**) &(dC[i]), sizeof(float) * M * N);
    }

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
        fprintf(stdout,"\n# ---------------------------------------- #\n");
        fprintf(stdout,"# --- Running sanity check on matrices --- #\n");
        fprintf(stdout,"# ---------------------------------------- #\n\n");

        // Sanity check with CUBLAS
        // cublasSgemm(handle, CUBLAS_OP_N,CUBLAS_OP_T, M, N, K,  &alpha, dA, M, dB, N, &beta, dC, M);

        // Allocate device memory for sanity check results
        CUDA_CALLER(cudaMalloc((void**) &(dC[0]), sizeof(float) * M * N));
        C[0] = (float*)calloc(M * N, sizeof(float));

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
        save_timestamp(&sanity_kernel_start_time);
        ft_sgemm_medium<<<gridDim, blockDim>>>(M, N, K, dA, dB, dC[0], alpha, beta); // , dcheck_A_col, dcheck_B_row, d_debug_int);

        // Timestamp end of kernel with cuda event
        cudaEventRecord(sanity_stop);
        cudaEventSynchronize(sanity_stop);
        uint64_t sanity_kernel_end_time;
        save_timestamp(&sanity_kernel_end_time);
        cudaEventElapsedTime(&sanity_time_ms, sanity_start, sanity_stop);
        fprintf(stdout,"Sanity check kernel execution time: %f ms\n", sanity_time_ms);

        cudaDeviceSynchronize();
        printf("Copying sanity check results to host memory...\n");
        CUDA_CALLER(cudaMemcpy(C[0], dC[0], sizeof(float) * M * N, cudaMemcpyDeviceToHost)); // TODO: make async with stream

        // Inject error
        // C[0][7*M + 15] *= 0.1; // Inject an error in the first element of C for testing

        // Count SEU errors by comparing C with C_ref
        printf("Calcuating SEU count for sanity check...\n");
        count_seu_errors(C[0], C_ref, M, N, &seu_count, error_row_idx, error_col_idx, error_value);

        kernel_event_to_file(
            events_file,
            -1,
            seu_count,
            (uint64_t)(sanity_time_ms * 1000), // Convert ms to us, CUDA event timing
            sanity_kernel_start_time, // Start timestamp of sanity check kernel, OS timestamp timing
            sanity_kernel_end_time, // End timestamp of sanity check kernel, OS timestamp timing
            false // repetition_cancelled
        );

        #ifdef ENABLE_SEU_DATA_LOGGING
        write_seu_data_to_file(
            seu_data_file,
            -1,
            seu_count,
            error_col_idx,
            error_row_idx,
            error_value
        );
        #endif

        fprintf(stdout,"Sanity check completed. SEUs prior to beam detected: %lld\n", seu_count);
    }

    /* -------------------- */
    /* Run selected kernels */
    /* -------------------- */
    fprintf(stdout,"\n# --------------------------------- #\n");
    fprintf(stdout,"# --- Starting kernel execution --- #\n");
    fprintf(stdout,"# --------------------------------- #\n\n");

    // Print kernel start timestamp
    kernel_launch_start_time = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(kernel_launch_start_time/1e6); // Convert microseconds to seconds
    fprintf(stdout,"Kernel launch start timestamp: %lu [us], %s", kernel_launch_start_time, ctime(&time_convert));

    fprintf(stdout,"Matrix size: %d, Kernel number: %d, Repetitions: %d\n", matrix_size, kernel_number, repeat_kernel);

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
    kernel_time_ms = (float*)malloc(sizeof(float) * MAX_CONCURRENT_KERNELS);

    // CONCURRENT STREAM SCHEDULING - START

    /* --------------------------------------- */
    /* Concurrent streaming scheduling - SETUP */
    /* --------------------------------------- */
    fprintf(stdout,"\n# ---------------------------------------------- #\n");
    fprintf(stdout,"# --- Setting up concurrent streamscheduling --- #\n");
    fprintf(stdout,"# ---------------------------------------------- #\n\n");

    completed_streams = 0;
    uint64_t error_timestamp = 0; // Track timestamp of any error
    uint64_t kernel_exec_start_time[MAX_CONCURRENT_KERNELS] = {0}; // Track start time of each kernel
    uint64_t kernel_exec_end_time[MAX_CONCURRENT_KERNELS] = {0}; // Track end time of each kernel
    int scheduled_streams = 0;
    int active_streams[MAX_CONCURRENT_KERNELS]; // Track active streams
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        active_streams[i] = -1;
    }

    uint64_t kernel_timeout_start = 0; // Track start time of kernel timeout
    uint64_t kernel_timeout_end = 0; // Track end time of kernel timeout
    int kernel_timeout_counter = 0; // Track how many kernel timeouts happend in a row

    /* -------------------------------------- */
    /* Concurrent streaming scheduling - LOOP */
    /* -------------------------------------- */
    fprintf(stdout,"\n# ------------------------------------------------- #\n");
    fprintf(stdout,"# --- Running concurrent stream scheduling loop --- #\n");
    fprintf(stdout,"# ------------------------------------------------- #\n\n");

    while (completed_streams < repeat_kernel || repeat_kernel == -1) {
    
        /* ------------------------------------------ */
        /* ERROR HANDLING - Check for any CUDA errors */
        /* ------------------------------------------ */
        bool unrecoverable_timeouts_detected = false;
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            save_timestamp(&error_timestamp);
            fprintf(stdout, "ERROR: CUDA error detected at %lu: (%s) %s\n", error_timestamp, cudaGetErrorName(err), cudaGetErrorString(err));

            // Report which streams were active at the time of the error
            fprintf(stdout, "WARNING: Current active streams: ");
            for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
                if (active_streams[i] != -1) {
                    fprintf(stdout,"%d,", active_streams[i]);
                }
            }
            fprintf(stdout, "\n");

            // Save error to events file
            error_event_to_file(
                events_file,
                ERROR_EVENT_ID,
                (int)err,
                error_timestamp
            );

            // Cleanup and destroy all streams
            for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
                if (active_streams[i] != -1) {
                    fprintf(stdout,"WARNING: Repetition %d cancelled due to some CUDA error.\n", active_streams[i]);

                    // Wait for the stream to finish
                    bool stream_timeout = false;
                    save_timestamp(&kernel_timeout_start);
                    while(cudaEventQuery(kernel_stop[i]) != cudaSuccess)
                    {
                        // TODO: Adjust timeout threshold
                        save_timestamp(&kernel_timeout_end);
                        if (kernel_timeout_end - kernel_timeout_start > KERNEL_MAX_TIMEOUT_DURATION) {
                            stream_timeout = true;
                            break;
                        }
                    }

                    // NOTE: Edge case to be handled if the only seen error is the CUDA kernels throwing errors, but no SEUs detecting the resulting matrices C. If SEUs detected in result matrices C, then this is not necessary.
                    // Case A1 - Handle successful kernel execution during clean up due to a CUDA error
                    if (stream_timeout == false)
                    {
                        kernel_timeout_counter = 0; // Reset kernel timeout counter
                        fprintf(stdout, "No kernel timeout\n");

                        // Record kernel execution time and end timestamp
                        save_timestamp(&(kernel_exec_end_time[i])); // Record end time
                        cudaEventElapsedTime(&kernel_time_ms[i], kernel_start[i], kernel_stop[i]);

                        // Copy results from device to host for this repetition
                        cudaMemcpyAsync(C[i], dC[i], sizeof(float) * M * N, cudaMemcpyDeviceToHost, mem_stream);

                        // Check for SEU errors for the active streams
                        count_seu_errors(C[i], C_ref, M, N, &seu_count, error_row_idx, error_col_idx, error_value);

                        // Save events to file for the active stream
                        kernel_event_to_file(
                            events_file,
                            active_streams[i],
                            seu_count,
                            (uint64_t)(kernel_time_ms[i] * 1000), // Convert ms to us, CUDA Event timing
                            kernel_exec_start_time[i], // Start timestamp of this kernel, OS timestamp timing
                            kernel_exec_end_time[i], // End timestamp of this kernel, OS timestamp timing
                            true // repetition_cancelled
                        );

                        // Save seu data to file for the active stream
                        #ifdef ENABLE_SEU_DATA_LOGGING
                        write_seu_data_to_file(
                            seu_data_file,
                            active_streams[i],
                            seu_count,
                            error_col_idx,
                            error_row_idx,
                            error_value
                        );
                        #endif

                        // Accumulate SEU counts over all kernel repetitions
                        seu_count_total += seu_count;
                    }
                    // Case A2 - Handle kernel timeout during clean up due to a CUDA error
                    else
                    {
                        kernel_timeout_counter++;
                        fprintf(stdout, "Kernel timeout\n");

                        // Save event to file about kernel timeout
                        kernel_event_to_file(
                            events_file,
                            active_streams[i],
                            0, // seu_count
                            0, // Convert ms to us, CUDA Event timing
                            kernel_exec_start_time[i], // Start timestamp of this kernel, OS timestamp timing
                            0, // End timestamp of this kernel, OS timestamp timing
                            true // repetition_cancelled
                        );
                    }

                    // LAST STEP - Update state variables for the cancelled stream
                    active_streams[i] = -1; // Mark this stream as inactive
                    completed_streams++;
                }
            }
        }

        /* ----------------------------------------------------- */
        /* STREAM RECYCLING - Check which streams have completed */
        /* ----------------------------------------------------- */
        for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
            if (active_streams[i] == -1) continue; // Skip inactive streams

            // CASE B1 - Handle one completed stream at a time during normal execution
            if (cudaEventQuery(kernel_stop[i]) == cudaSuccess) {
                kernel_timeout_counter = 0; // Reset kernel timeout counter
                save_timestamp(&(kernel_exec_end_time[i])); // Record end time
                cudaEventElapsedTime(&kernel_time_ms[i], kernel_start[i], kernel_stop[i]);
                fprintf(stdout,"INFO: %d - CPU: %lu us, CUDA: %d us, ", active_streams[i], kernel_exec_end_time[i] - kernel_exec_start_time[i], int(kernel_time_ms[i]*1000.0));

                // Copy results from device to host for this repetition
                cudaMemcpyAsync(C[i], dC[i], sizeof(float) * M * N, cudaMemcpyDeviceToHost, mem_stream);

                // Add error injection for testing
                // if ( active_streams[i] == 7 ) // Inject error for the 8th repetition (index 7)
                // {
                //     C[i][4*M + 13] *= 0.1; // Introduce a small error in the first element
                // }

                // Check for SEU errors for this repetition
                count_seu_errors(C[i], C_ref, M, N, &seu_count, error_row_idx, error_col_idx, error_value);
                fprintf(stdout,"SEU: %lld (tot. %lld)", seu_count, seu_count_total + seu_count);
                // if (seu_count > 0) {
                //     fprintf(stdout, "c: %d, r: %d, v: %e\n", error_col_idx[0], error_row_idx[0], error_value[0]);
                // }
                fprintf(stdout, "\r\n");

                // Save events to file for this repetition
                kernel_event_to_file(
                    events_file,
                    active_streams[i],
                    seu_count,
                    (uint64_t)(kernel_time_ms[i] * 1000), // Convert ms to us, CUDA Event timing
                    kernel_exec_start_time[i], // Start timestamp of this kernel, OS timestamp timing
                    kernel_exec_end_time[i], // End timestamp of this kernel, OS timestamp timing
                    false // repetition_cancelled
                );

                // Save seu data to file for this repetition
                #ifdef ENABLE_SEU_DATA_LOGGING
                write_seu_data_to_file(
                    seu_data_file,
                    active_streams[i],
                    seu_count,
                    error_col_idx,
                    error_row_idx,
                    error_value
                );
                #endif

                // Accumulate SEU counts over all kernel repetitions
                seu_count_total += seu_count;

                // LAST STEP - Update state variables
                active_streams[i] = -1; // Mark this stream as inactive
                completed_streams++;

                break; // Only handle one completed stream each iteration, so that new kernels can be scheduled
            }
            // CASE B2 - Check and handle timeout of the kernel execution for this stream during normal execution
            else
            {
                save_timestamp(&kernel_timeout_end);
                if (kernel_timeout_end - kernel_exec_start_time[i] > KERNEL_MAX_TIMEOUT_DURATION) {
                    fprintf(stdout,"ERROR: Kernel execution for repetition %d has timed out. Marking stream as inactive.\n", active_streams[i]);
                    kernel_timeout_counter++; // Increment the timeout counter

                    // Save event to file about kernel timeout
                    kernel_event_to_file(
                        events_file,
                        active_streams[i],
                        0, // seu_count
                        0, // Convert ms to us, CUDA Event timing
                        kernel_exec_start_time[i], // Start timestamp of this kernel, OS timestamp timing
                        0, // End timestamp of this kernel, OS timestamp timing
                        true // repetition_cancelled
                    );

                    // LAST STEP - Update state variables for the cancelled stream
                    active_streams[i] = -1; // Mark this stream as inactive
                    completed_streams++;
                }
            }
        }

        /* ----------------------------------------------------------------------- */
        /* UNRECOVERABLE ERROR HANDLING - Handle unrecoverable error and reset GPU */
        /* ----------------------------------------------------------------------- */
        // Check if kernel timeout counter has exceeded threshold
        if (kernel_timeout_counter >= KERNEL_ALLOWED_MAX_TIMEOUTS) {
            unrecoverable_timeouts_detected = true;
        }

        // Exit application for unrecoverable CUDA errors
        if (err == cudaErrorIllegalAddress || unrecoverable_timeouts_detected == true)
        {
            if (err == cudaErrorIllegalAddress)
            {
                fprintf(stdout,"ERROR: An illegal memory access was detected, which is a unrecoverable error.\n");
            }

            if (unrecoverable_timeouts_detected == true)
            {
                fprintf(stdout,"ERROR: Last %d streams have timed out. This is treated as a unrecoverable error.\n", kernel_timeout_counter);
            }

            // Reset the device to recover from the error
            save_timestamp(&gpu_reset_timestamp);
            cudaDeviceReset();
            fprintf(stdout,"WARNING: CUDA device was reset at %lu\n", gpu_reset_timestamp);

            // Exit application
            return;
        }

        /* ----------------------------------------------------------------- */
        /* STREAM SCHEDULING - Schedule new kernels for any inactive streams */
        /* ----------------------------------------------------------------- */
        for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {

            // Find an inactive stream slot
            if (active_streams[i] == -1) {

                // Assign the next repetition to this stream
                active_streams[i] = scheduled_streams;
                scheduled_streams++; // Increment the scheduled streams counter

                // Launch the kernel for this repetition
                save_timestamp(&(kernel_exec_start_time[i])); // Record start time
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
                    // if (active_streams[i] == 7) mem_access_error = M * N * sizeof(float) * repeat_kernel; // -> illegal memory access error
                    // if (active_streams[i] == 2) cudaMalloc((void**)&dC[i], -1); // -> device out of memory and subsequent illegal memory access error
                    // TODO: Remove after DEBUGGING - END
                    ft_sgemm_medium<<<gridDim, blockDim, 0, kernel_stream[i]>>>(M, N, K, dA, dB, dC[i+mem_access_error], alpha, beta);
                }
                else
                {
                    fprintf(stdout,"ERROR: Kernel number %d is not implemented. Exiting application.\n", kernel_number);
                    return;
                }
                cudaEventRecord(kernel_stop[i], kernel_stream[i]);

                // fprintf(stdout,"Repetition %d started.\n", active_streams[i]);
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

    /* ------------------------------------ */
    /* Cleanup concurrent stream scheduling */
    /* ------------------------------------ */
    fprintf(stdout,"\n# ------------------------------------------------ #\n");
    fprintf(stdout,"# --- Cleaning up concurrent stream scheduling --- #\n");
    fprintf(stdout,"# ------------------------------------------------ #\n\n");

    // Clean up CUDA events and streams
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        cudaEventDestroy(kernel_start[i]);
        cudaEventDestroy(kernel_stop[i]);
        cudaStreamDestroy(kernel_stream[i]);
    }

    // Print total SEU errors across all repetitions
    fprintf(stdout,"Total SEU errors across all repetitions: %lld\n", seu_count_total);
}

/* ------------- */
/* Main Function */
/* ------------- */

int main(int argc, char **argv){
    fprintf(stdout, "\n--------------------------------------------------------------------------\n");
    // Print start timestamp
    app_start_timestamp = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    time_convert = static_cast<time_t>(app_start_timestamp/1e6); // Convert microseconds to seconds
    fprintf(stdout,"Program start timestamp: %lu [us], %s", app_start_timestamp, ctime(&time_convert));

    /* ------------------------------ */
    /* Setup signal and exit handlers */
    /* ------------------------------ */
    fprintf(stdout,"\n# ------------------------------------------- #\n");
    fprintf(stdout,"# --- Setting up signal and exit handlers --- #\n");
    fprintf(stdout,"# ------------------------------------------- #\n\n");

    // Register atexit handler to ensure it gets called on normal exit
    const int result = std::atexit(atexit_handler); // Handler will be called

    // Register signal handler for Ctrl+C (SIGINT)
    std::signal(SIGINT, signal_handler);

    if (result != 0)
    {
        std::cerr << "atexit registration failed\n";
        return EXIT_FAILURE;
    }

    /* ---------------------- */
    /* Process compiler flags */
    /* ---------------------- */

    // Inform user if ABFT error correction is disabled
    #ifdef DISABLE_ERROR_CORRECTION
    fprintf(stdout,"\n----------------------------------------\n");
    fprintf(stdout,"!! ABFT error correction is DISABLED !!");
    fprintf(stdout,"\n----------------------------------------\n\n");
    #endif

    // Check compiler flags
    #ifdef ENABLE_SEU_DATA_LOGGING
    enable_seu_data_logging = true;
    #endif

    /* -------------------------- */
    /* Get command line arguments */
    /* -------------------------- */
    fprintf(stdout,"\n# -------------------------------------- #\n");
    fprintf(stdout,"# --- Parsing command line arguments --- #\n");
    fprintf(stdout,"# -------------------------------------- #\n\n");

    // Check if the required number of arguments is provided
    if (argc < 11 + 1)
    {
        fprintf(stdout,"Expected 11 arguments: Matrix start size, end size, step size, kernel start number, end number, repetitions, enable_beam_signal_wait, input folder, output folder, experiment number, enable_sanity_check. Some arguments are missing! Exiting...\n");
        exit(-1);
    }

    srand(10);
    // int start_size = atoi(argv[1]);
    matrix_size =  atoi(argv[2]);
    M = matrix_size; N = matrix_size;  K = matrix_size;
    // int gap_size =  atoi(argv[3]);
    // int st_kernel = atoi(argv[4]);
    kernel_number = atoi(argv[5]);
    repeat_kernel = atoi(argv[6]);
    enable_beam_signal_wait = atoi(argv[7]);
    folder_name = argv[8];
    sprintf(file_A_name, "%sA_%d.bin", folder_name, matrix_size);
    sprintf(file_B_name, "%sB_%d.bin", folder_name, matrix_size);
    sprintf(file_C_name, "%sC_%d.bin", folder_name, matrix_size);
    output_folder = argv[9];
    experiment_name = argv[10];
    enable_sanity_check = atoi(argv[11]);

    // SEU error tracking arrays
    error_row_idx = (int *)calloc(M * N, sizeof(int));
    error_col_idx = (int *)calloc(M * N, sizeof(int));
    error_value = (float *)calloc(M * N, sizeof(float));

    /* ----------------------------------------- */
    /* Allocate host memory and read input files */
    /* ----------------------------------------- */
    fprintf(stdout,"\n# ------------------------------------------------- #\n");
    fprintf(stdout,"# --- Allocate host memory and read input files --- #\n");
    fprintf(stdout,"# ------------------------------------------------- #\n\n");

    A = (float *)calloc(M * N, sizeof(float));
    B = (float *)calloc(M * K, sizeof(float));
    // C = (float *)calloc(M * N * repeat_kernel, sizeof(float));
    C_ref = (float *)calloc(M * N, sizeof(float));
    check_C_col = (float *)calloc(N, sizeof(float));
    check_C_row = (float *)calloc(M, sizeof(float));

    // Allocate host memory for output matrices for each repetition
    C = (float **)malloc(sizeof(float *) * MAX_CONCURRENT_KERNELS);
    for (int i = 0; i < MAX_CONCURRENT_KERNELS; i++) {
        C[i] = (float *)calloc(M * N, sizeof(float));
    }

    // Initialize arrays with random values (or read from file)
    // -- Generarte random matrices (Very slow !! Read from file instead !!)
    // Use the generate matrix utility -> see generate_matrices.cu

    // -- Read input matrices from file
    printf("Reading input matrices from input files...\n");
    read_matrix_from_file(file_A_name, A, matrix_size);
    read_matrix_from_file(file_B_name, B, matrix_size);
    read_matrix_from_file(file_C_name, C_ref, matrix_size);

    // Inject many error to reference matrix
    // for (int i = 0; i < M * N; i++)
    // {
    //     C_ref[i] *= 0.1;
    // }

    /* -------------------- */
    /* Prepare output files */
    /* -------------------- */
    fprintf(stdout,"\n# ------------------------------ #\n");
    fprintf(stdout,"# --- Preparing output files --- #\n");
    fprintf(stdout,"# ------------------------------ #\n\n");

    // Prepare file names and paths
    std::tm* f_ts = std::localtime(&time_convert);
    std::ostringstream ss;
    ss << std::put_time(f_ts, "%Y-%m-%d_%a_%H:%M:%S");
    file_timestamp = ss.str();
    results_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(RESULTS_FILE);
    events_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(EVENTS_FILE);
    seu_data_file_path = std::string(output_folder) + file_timestamp + std::string("_") + std::string("exp_") + std::string(experiment_name) + std::string("_") + std::string(SEU_DATA_FILE);

    // Open results file for writing
    results_file = new std::fstream(results_file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!results_file->is_open()) {
        fprintf(stderr, "Error opening results file for writing: %s\n", results_file_path.c_str());
        results_file->close();
        exit(EXIT_FAILURE);
    }

    // Open events file for writing
    events_file = new std::fstream(events_file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!events_file->is_open()) {
        fprintf(stderr, "Error opening events file for writing: %s\n", events_file_path.c_str());
        events_file->close();
        exit(EXIT_FAILURE);
    }

    // Open SEU data file for writing
    #ifdef ENABLE_SEU_DATA_LOGGING
    seu_data_file = new std::fstream(seu_data_file_path, std::ios::out | std::ios::binary | std::ios::app);
    if (!seu_data_file->is_open()) {
        fprintf(stderr, "Error opening SEU data file for writing: %s\n", seu_data_file_path.c_str());
        seu_data_file->close();
        exit(EXIT_FAILURE);
    }
    #endif

    // Write headers to output files
    printf("Writing header to output files...\n");
    write_header_to_results_file(results_file);
    write_header_to_events_file(events_file);

    while(true){
        // Increment child process spawn count
        spawn_count++;

        // Print statup message for kernel execution loop
        fprintf(stdout, "\n# ---------------------------------------- #\n");
        fprintf(stdout, "# --- KERNEL EXECUTION LOOP %lu started. --- #\n", spawn_count);
        fprintf(stdout, "# ---------------------------------------- #\n");

        // Spawn child process for kernel execution
        pid = fork();

        // FORK failure
        if (pid < 0) {
            fprintf(stderr, "Fork failed. Exiting...\n");
            exit(EXIT_FAILURE);

        // CHILD process - KERNEL EXECUTION LOOP
        } else if (pid == 0) {
            kernel_execution_loop();
            cleanup_kernel_execution_loop();
            break; // Exit child process after kernel execution loop

        // PARENT process - Wait for child process to complete
        } else {
            int status;
            fprintf(stdout, "PARENT: Wait for kernel execution loop %lu\n", spawn_count);
            waitpid(pid, &status, 0);
            fprintf(stdout, "PARENT: Kernel execution loop %lu status: %d\n", spawn_count, WEXITSTATUS(status));
        }

        // Stop if kernel repetition count is provided
        if(repeat_kernel != -1) break;
    }

    exit(0); // Exit normally to ensure atexit handler is called
}
