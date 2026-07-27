from multiprocessing import Pool

import colorlog
import logging
import multiprocessing
import pexpect
import signal
import time

# Script settings
WORKSPACE_FOLDER = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/beam_data/"
EXPERIMENT_INPUT_FOLDER = f"{WORKSPACE_FOLDER}exp_input_matrices/"
EXPERIMENT_RESULTS_FOLDER = f"{WORKSPACE_FOLDER}exp_results/"
KERNEL_PROCESS_START_INTERVAL = 0.1 # [seconds] # TODO: adjust to a reasonable value for our radiation test setup
LIMIT_KERNEL_PROCESSES = 1 # None: spawn infinite kernel processes, int: stop after this many kernel processes have been spawned # TODO: adjust to a reasonable value for our radiation test setup

# Logging configuration
LOG_FILE = "beam_script_dut.log"
BEAM_LOGGER_LEVEL = logging.DEBUG
CUDA_LOGGER_LEVEL = logging.DEBUG
LOG_RESET = "\033[0m"
LOG_GREEN = "\033[32m"
LOG_WHITE = "\033[37m"

# Logging loggers
beam_logger = logging.getLogger("BEAM")
beam_logger.handlers.clear() # Clear existing handlers
beam_logger.setLevel(BEAM_LOGGER_LEVEL)

# Logging handlers and formatters for BEAM Logger
beam_stream_handler = logging.StreamHandler()
beam_file_handler = logging.FileHandler(EXPERIMENT_RESULTS_FOLDER + LOG_FILE, mode='a')
beam_logger_formatter = colorlog.ColoredFormatter('%(log_color)s%(asctime)s- %(process)s - %(levelname)s - %(name)s - %(message)s')
beam_stream_handler.setFormatter(beam_logger_formatter) # Set formatter for stdout stream handler
beam_file_handler.setFormatter(beam_logger_formatter) # Set formatter for log file handler

# Set formatter for BEAM logger handlers
beam_logger.addHandler(beam_stream_handler) # Add stream handler to BEAM logger
beam_logger.addHandler(beam_file_handler) # Add file handler to BEAM logger

# Logging handlers and formatters for CUDA Logger
cuda_stream_handler = logging.StreamHandler()
cuda_file_handler = logging.FileHandler(EXPERIMENT_RESULTS_FOLDER + LOG_FILE, mode='a')
cuda_logger_formatter = logging.Formatter('%(asctime)s- %(process)s - %(levelname)s - %(name)s - %(message)s')
cuda_stream_handler.setFormatter(cuda_logger_formatter) # Set formatter for stdout stream handler
cuda_file_handler.setFormatter(cuda_logger_formatter) # Set formatter for log file handler

# Kernel settings
MATRIX_SIZE = 4096
KERNEL_NUMBER = 12
KERNEL_REPETITIONS = 1
KERNEL_ENABLE_TRIGGER = False
KERNEL_INPUT_FOLDER = EXPERIMENT_INPUT_FOLDER
KERNEL_OUTPUT_FOLDER = EXPERIMENT_RESULTS_FOLDER
KERNEL_COMMAND = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build/run_kernel_no_check"

# KERNEL_LIST = [
#     ("sgemm_small", 1),
#     ("sgemm_medium", 2),
#     ("sgemm_large", 3),
#     ("sgemm_tall", 4),
#     ("sgemm_wide", 5),
#     ("sgemm_huge", 6),
#     ("ft_sgemm_small", 11),
#     ("ft_sgemm_medium", 12),
#     ("ft_sgemm_large", 13),
#     ("ft_sgemm_tall", 14),
#     ("ft_sgemm_wide", 15),
#     ("ft_sgemm_huge", 16),
#     ("ft_sgemm_medium_96", 17),
# ]

MATRIX_TIMEOUTS = {
    1024: 1.0, # [seconds]
    2048: 1.0, # [seconds]
    3072: 2.0, # [seconds]
    4096: 4.0, # [seconds]
    5120: 4.0, # [seconds]
    6144: 4.0, # [seconds]
    7168: 8.0, # [seconds]
    8192: 10.0, # [seconds]
}

KERNEL_STARTUP_TIMEOUTS = {
    1: 5.0, # [seconds]
    2: 5.0, # [seconds]
    3: 5.0, # [seconds]
    4: 5.0, # [seconds]
    5: 5.0, # [seconds]
    6: 5.0, # [seconds]
    11: 10.0, # [seconds]
    12: 10.0, # [seconds]
    13: 10.0, # [seconds]
    14: 10.0, # [seconds]
    15: 10.0, # [seconds]
    16: 10.0, # [seconds]
    17: 10.0, # [seconds]
}

KERNEL_END_TIMEOUTS = {
    1: 5.0, # [seconds]
    2: 5.0, # [seconds]
    3: 5.0, # [seconds]
    4: 5.0, # [seconds]
    5: 5.0, # [seconds]
    6: 5.0, # [seconds]
    11: 5.0, # [seconds]
    12: 5.0, # [seconds]
    13: 5.0, # [seconds]
    14: 5.0, # [seconds]
    15: 5.0, # [seconds]
    16: 5.0, # [seconds]
    17: 5.0, # [seconds]
}

def get_kernel_logger(experiment_name: str, id: int = 0):
    logger = logging.getLogger(f"CUDA {experiment_name} {id}")
    logger.handlers.clear() # Clear existing handlers
    logger.setLevel(CUDA_LOGGER_LEVEL)

    logger.addHandler(cuda_stream_handler) # Add same handlers to CUDA logger
    logger.addHandler(cuda_file_handler) # Add same handlers to CUDA logger

    return logger

def run_kernel(matrix_size: int, kernel_number: int, kernel_repetitions: int, enable_trigger: bool, input_folder: str, output_folder: str, experiment_name: str, id: int = 0):
        cuda_logger = get_kernel_logger(experiment_name, id)

        kernel = pexpect.spawn("sudo", [KERNEL_COMMAND, str(matrix_size), str(matrix_size), str(matrix_size), str(kernel_number), str(kernel_number), str(kernel_repetitions), str(1 if enable_trigger else 0), input_folder, output_folder, experiment_name])

        # Waiting for kernel to start on GPU
        startup_timeout = KERNEL_STARTUP_TIMEOUTS.get(kernel_number, 5.0)
        while kernel.isalive():
            read = kernel.expect(['\r\n', pexpect.TIMEOUT, pexpect.EOF], timeout=startup_timeout)
            cuda_logger.info(kernel.before.decode())

            # Reset startup timeout
            startup_timeout = KERNEL_STARTUP_TIMEOUTS.get(kernel_number, 5.0)

            if read == 0:
                if "Waiting for trigger signal" in kernel.before.decode():
                    beam_logger.info(f"Kernel {experiment_name} {id} - WAIT indefinitely for trigger signal")
                    startup_timeout = None # Wait indefinitely for trigger after this point
                elif "Starting kernel execution..." in kernel.before.decode():
                    beam_logger.info(f"Kernel {experiment_name} {id} - STARTING KERNEL EXECUTION")
                    break
            elif read == 1:
                beam_logger.warning(f"Kernel {experiment_name} {id} - TIMEOUT during KERNEL STARTUP")
                read_eof = kernel.expect([pexpect.EOF, pexpect.TIMEOUT], timeout=5.0)
                cuda_logger.warning(kernel.before.decode())
                kernel.close()
                beam_logger.warning(f"Kernel {experiment_name} {id} - Exit status: {kernel.exitstatus}, Signal status: {kernel.signalstatus}")
                return
            elif read == 2:
                beam_logger.warning(f"Kernel {experiment_name} {id} - EOF during KERNEL STARTUP")
                kernel.close()
                beam_logger.warning(f"Kernel {experiment_name} {id} - Exit status: {kernel.exitstatus}, Signal status: {kernel.signalstatus}")
                return

        # Execution of kernel on GPU
        kernel_timeout = MATRIX_TIMEOUTS.get(matrix_size, 1.0) * kernel_repetitions
        while kernel.isalive():
            read = kernel.expect(["\r\n", pexpect.TIMEOUT, pexpect.EOF], timeout=kernel_timeout)
            cuda_logger.info(kernel.before.decode())

            if read == 0:
                if "Total SEU errors" in kernel.before.decode():
                    kernel_timeout = KERNEL_END_TIMEOUTS.get(kernel_number, 5.0)
            elif read == 1:
                beam_logger.warning(f"Kernel {experiment_name} {id} - TIMEOUT during KERNEL EXECUTION")
                kernel.close()
                beam_logger.warning(f"Kernel {experiment_name} {id} - Exit status: {kernel.exitstatus}, Signal status: {kernel.signalstatus}")
                break
            elif read == 2:
                beam_logger.info(f"Kernel {experiment_name} {id} - FINISHED KERNEL EXECUTION")
                kernel.close()
                beam_logger.info(f"Kernel {experiment_name} {id} - Exit status: {kernel.exitstatus}, Signal status: {kernel.signalstatus}")
                break
        return

def signal_handler(signum, frame):

    if multiprocessing.parent_process() is None:
        beam_logger.warning("Ctrl+C received. Terminating all kernel processes...")

        # Terminate all kernel processes
        for process in multiprocessing.active_children():
            process.terminate()
        beam_logger.warning("All kernel processes terminated.")

        # Inform user about termination
        print_script_timestamp("Ctrl+C stopped")
    
        exit(0)

def print_script_timestamp(state: str):
    beam_logger.info("")
    beam_logger.info(LOG_WHITE + "# ------------------------------------------------------------------------------------- #" + LOG_RESET)
    beam_logger.info(LOG_WHITE + f"# ------------------ {state} beam script [{time.ctime()}] ------------------ #" + LOG_RESET)
    beam_logger.info(LOG_WHITE + "# ------------------------------------------------------------------------------------- #" + LOG_RESET)
    beam_logger.info("")

def main():
    # Step 1 - Check if correct Orin power mode is set
    # See "Test setup" tab in notebook on google drive
    # Power mode: 25W (id=3)

    # Step 2 - Check if sudo password prompt is disabled for all relevant commands
    # See "Test setup" tab in notebook on google drive
    # Commands: run_kernel_no_check, ncu, nsys

    # Step 2 - Connect to control laptop
    # Setup streams to collect data during radiation

    # Step 3 - Run kernel
    # Start kernels
    kernel_processes = []
    kernel_id = 0
    while True:
        # Start a new kernel process
        kernel_processes.append((multiprocessing.Process(target=run_kernel, args=(MATRIX_SIZE, KERNEL_NUMBER, KERNEL_REPETITIONS, KERNEL_ENABLE_TRIGGER, KERNEL_INPUT_FOLDER, KERNEL_OUTPUT_FOLDER, "test", kernel_id)), kernel_id))
        beam_logger.info(f"Starting kernel {kernel_id} process.")
        kernel_processes[-1][0].start()

        # # Check if any kernel processes have finished and remove them from the list
        # for process in kernel_processes:
        #     process_obj, id = process
        #     if not process_obj.is_alive():
        #         beam_logger.info(f"Kernel {id} process finished.")
        #         kernel_processes.remove(process)

        # Increment kernel ID for the next kernel process
        kernel_id += 1

        # Wait a bit before starting the next kernel to avoid overwhelming the system
        time.sleep(KERNEL_PROCESS_START_INTERVAL)

        # Wait until a CPU core is available to start a new kernel process
        while len(kernel_processes) >= multiprocessing.cpu_count():
            # Check if any kernel processes have finished and remove them from the list
            for process in kernel_processes:
                process_obj, id = process
                if not process_obj.is_alive():
                    beam_logger.info(f"Kernel {id} process finished.")
                    kernel_processes.remove(process)

            # Give some time for the CPU to free up before checking again
            time.sleep(0.1)
        
        if LIMIT_KERNEL_PROCESSES:
            if kernel_id >= LIMIT_KERNEL_PROCESSES:
                break

    # Step 4 - Stream radiation data to control laptop
    # Log data and collect data from kernels

    return kernel_processes

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    print_script_timestamp("Starting")

    pending_processes = main()

    for pending_process in pending_processes:
        process_obj, id = pending_process
        beam_logger.info(f"MAIN finished. WAIT for kernel {id} to finish.")
        process_obj.join()

    print_script_timestamp("Finished")