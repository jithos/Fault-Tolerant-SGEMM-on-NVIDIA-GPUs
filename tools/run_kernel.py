import os
import subprocess
import sys

# Script settings
WORKSPACE_FOLDER = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/"

BEAM_DATA_FOLDER = f"{WORKSPACE_FOLDER}beam_data/"
EXPERIMENT_INPUT_FOLDER = f"{BEAM_DATA_FOLDER}exp_input_matrices/"
EXPERIMENT_RESULTS_FOLDER = f"{BEAM_DATA_FOLDER}exp_results/"

PROFILING_RESULTS_FOLDER = f"{WORKSPACE_FOLDER}profiling_results/"
NSYS_RESULTS_FOLDER = f"{PROFILING_RESULTS_FOLDER}nsys_exp_results/"
NCU_RESULTS_FOLDER = f"{PROFILING_RESULTS_FOLDER}ncu_exp_results/"

# Kernel settings
MATRIX_SIZE = 1024
KERNEL_NUMBER = 12
KERNEL_REPETITIONS = 1
KERNEL_ENABLE_TRIGGER = False
KERNEL_INPUT_FOLDER = EXPERIMENT_INPUT_FOLDER
KERNEL_OUTPUT_FOLDER = EXPERIMENT_RESULTS_FOLDER
EXPERIMENT_NAME = "X"
KERNEL_COMMAND = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build/run_kernel_no_check"

KERNEL_LIST = {
    1: "sgemm_small",
    2: "sgemm_medium",
    3: "sgemm_large",
    4: "sgemm_tall",
    5: "sgemm_wide",
    6: "sgemm_huge",
    11: "ft_sgemm_small",
    12: "ft_sgemm_medium",
    13: "ft_sgemm_large",
    14: "ft_sgemm_tall",
    15: "ft_sgemm_wide",
    16: "ft_sgemm_huge",
    17: "ft_sgemm_medium_96",
}

if __name__ == "__main__":

    # Check for command-line arguments
    run_ncu = False
    run_nsys = False
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "help":
            print("Usage: python run_kernel.py [option]")
            print("Options:")
            print("  help - Show this help message")
            print("  ncu - Profile the kernel with NVIDIA Nsight Compute")
            print("  nsys - Profile the kernel with NVIDIA Nsight Systems")
            sys.exit(1)
        elif sys.argv[1] == "ncu":
            run_ncu = True
        elif sys.argv[1] == "print-ncu":
            if len(sys.argv) < 3:
                print("Error: Please provide the NCU report file path.")
                sys.exit(1)
            print_ncu = True
            ncu_report_file = sys.argv[2]
        elif sys.argv[1] == "nsys":
            run_nsys = True
        else:
            print("Invalid option. Use 'help' for usage information.")
            sys.exit(1)

    # Construct the command to run the kernel
    command = [
        KERNEL_COMMAND,
        str(MATRIX_SIZE),
        str(MATRIX_SIZE),
        str(MATRIX_SIZE),
        str(KERNEL_NUMBER),
        str(KERNEL_NUMBER),
        str(KERNEL_REPETITIONS),
        str(int(KERNEL_ENABLE_TRIGGER)),
        KERNEL_INPUT_FOLDER,
        KERNEL_OUTPUT_FOLDER,
        EXPERIMENT_NAME
    ]

    # Add profiling tools if specified
    if run_ncu:
        command = ["sudo", "-E", "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu", "--config-file", "off", "--export", f"{NCU_RESULTS_FOLDER}ncu_exp_{EXPERIMENT_NAME}_{KERNEL_LIST[KERNEL_NUMBER]}", "--force-overwrite", "--kernel-name", f"{KERNEL_LIST[KERNEL_NUMBER]}"] + command
        print("NCU profiling enabled")
    elif run_nsys:
        command = ["sudo", "-E", "nsys", "profile", "--stats=true", "--output", f"{NSYS_RESULTS_FOLDER}nsys_exp_{EXPERIMENT_NAME}_{KERNEL_LIST[KERNEL_NUMBER]}", "--force-overwrite", "true"] + command
        print("NSYS profiling enabled")
    elif print_ncu:
        command = ["ncu", "-i", ncu_report_file]
        print("Printing NCU report")
    else:
        command = ["sudo", "-E"] + command

    # Print the command for debugging purposes
    print("Executing command:", " ".join(command))

    # Execute the command
    subprocess.run(command)