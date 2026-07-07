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
KERNEL_ENABLE_SANITY_CHECK = False
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
    print_ncu = False
    
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
        # elif sys.argv[1] == "print_ncu":
        #     if len(sys.argv) < 2:
        #         print("Error: Please provide the NCU report file path.")
        #         sys.exit(1)
        #     ncu_report_file = sys.argv[2]
        #     print_ncu = True
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
        EXPERIMENT_NAME,
        str(int(KERNEL_ENABLE_SANITY_CHECK))
    ]

    # Add profiling tools if specified
    if run_ncu:
        command = ["sudo", "-E", "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu", "--config-file", "off", "--export", f"{NCU_RESULTS_FOLDER}ncu_exp_{EXPERIMENT_NAME}_{KERNEL_LIST[KERNEL_NUMBER]}", "--force-overwrite", "--kernel-name", f"{KERNEL_LIST[KERNEL_NUMBER]}"] + command
        print("NCU profiling enabled")
    elif run_nsys:
        command = ["sudo", "-E", "nsys", "profile", "--stats=true", "--output", f"{NSYS_RESULTS_FOLDER}nsys_exp_{EXPERIMENT_NAME}_{KERNEL_LIST[KERNEL_NUMBER]}", "--force-overwrite", "true"] + command
        print("NSYS profiling enabled")
    # elif print_ncu:
    #     command = ["ncu", "-i", ncu_report_file]
    #     print("Printing NCU report")
    else:
        command = ["sudo", "-E"] + command

    # Make sure the build directory is up to date
    make_command = ["make", "-j"]
    print("----------------------------------------------------------------")
    print("Executing command: ", " ".join(make_command))
    make_process = subprocess.run(make_command, cwd=f"{WORKSPACE_FOLDER}/build")
    if make_process.returncode != 0:
        print("Error: Make command failed with return code", make_process.returncode)
        print("----------------------------------------------------------------")
        sys.exit(1)
    print("----------------------------------------------------------------")


    # Execute the command
    print("----------------------------------------------------------------")
    print("Executing command:", " ".join(command))
    kernel_process = subprocess.run(command)
    if kernel_process.returncode != 0:
        print("Error: Kernel execution failed with return code", kernel_process.returncode)
        print("----------------------------------------------------------------")
        sys.exit(1)
    if run_ncu:
        subprocess.run(["ncu", "-i", f"{NCU_RESULTS_FOLDER}ncu_exp_{EXPERIMENT_NAME}_{KERNEL_LIST[KERNEL_NUMBER]}.ncu-rep"])
    print("----------------------------------------------------------------")