import os
import subprocess
import sys

import kernel_settings as ks

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

    # Prepare executable command with profiling tools if specified
    if run_ncu:
        command = ["sudo", "-E", "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu", "--config-file", "off", "--export", f"{ks.NCU_RESULTS_FOLDER}ncu_exp_{ks.EXPERIMENT_NAME}_{ks.KERNEL_LIST[ks.KERNEL_NUMBER]}", "--force-overwrite", "--kernel-name", f"{ks.KERNEL_LIST[ks.KERNEL_NUMBER]}"] + ks.KERNEL_COMMAND
        print("NCU profiling enabled")
    elif run_nsys:
        command = ["sudo", "-E", "nsys", "profile", "--trace=cuda", "--stats=true", "--output", f"{ks.NSYS_RESULTS_FOLDER}nsys_exp_{ks.EXPERIMENT_NAME}_{ks.KERNEL_LIST[ks.KERNEL_NUMBER]}", "--force-overwrite", "true"] + ks.KERNEL_COMMAND
        print("NSYS profiling enabled")
    # elif print_ncu:
    #     command = ["ncu", "-i", ncu_report_file]
    #     print("Printing NCU report")
    else:
        command = ["sudo", "-E"] + ks.KERNEL_COMMAND

    # Make sure the build directory is up to date
    make_command = ["make", "-j"]
    print("----------------------------------------------------------------")
    print("Executing command: ", " ".join(make_command))
    make_process = subprocess.run(make_command, cwd=f"{ks.WORKSPACE_FOLDER}/build")
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
        subprocess.run(["ncu", "-i", f"{ks.NCU_RESULTS_FOLDER}ncu_exp_{ks.EXPERIMENT_NAME}_{ks.KERNEL_LIST[ks.KERNEL_NUMBER]}.ncu-rep"])
    print("----------------------------------------------------------------")