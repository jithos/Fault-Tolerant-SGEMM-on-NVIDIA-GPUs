
# Script settings
WORKSPACE_FOLDER = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/"

BEAM_DATA_FOLDER = f"{WORKSPACE_FOLDER}beam_data/"
EXPERIMENT_INPUT_FOLDER = f"{BEAM_DATA_FOLDER}exp_input_matrices/"
EXPERIMENT_RESULTS_FOLDER = f"{BEAM_DATA_FOLDER}exp_results/"

PROFILING_RESULTS_FOLDER = f"{WORKSPACE_FOLDER}profiling_results/"
NSYS_RESULTS_FOLDER = f"{PROFILING_RESULTS_FOLDER}nsys_exp_results/"
NCU_RESULTS_FOLDER = f"{PROFILING_RESULTS_FOLDER}ncu_exp_results/"

# Kernel settings
MATRIX_SIZE = 256
KERNEL_NUMBER = 12
KERNEL_REPETITIONS = -1
KERNEL_ENABLE_BEAM_SIGNAL_WAIT = False
KERNEL_ENABLE_SANITY_CHECK = False
KERNEL_ENABLE_SEU_DATA_LOGGING = False
KERNEL_INPUT_FOLDER = EXPERIMENT_INPUT_FOLDER
KERNEL_OUTPUT_FOLDER = EXPERIMENT_RESULTS_FOLDER
EXPERIMENT_NAME = "Y"
KERNEL_FOLDER = f"{WORKSPACE_FOLDER}build/"
KERNEL_EXECUTABLE = "run_kernel_seu_data" if KERNEL_ENABLE_SEU_DATA_LOGGING else "run_kernel_seu_count"

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

# Construct the command to run the kernel
KERNEL_COMMAND = [
    KERNEL_FOLDER + KERNEL_EXECUTABLE,
    str(MATRIX_SIZE),
    str(MATRIX_SIZE),
    str(MATRIX_SIZE),
    str(KERNEL_NUMBER),
    str(KERNEL_NUMBER),
    str(KERNEL_REPETITIONS),
    str(int(KERNEL_ENABLE_BEAM_SIGNAL_WAIT)),
    KERNEL_INPUT_FOLDER,
    KERNEL_OUTPUT_FOLDER,
    EXPERIMENT_NAME,
    str(int(KERNEL_ENABLE_SANITY_CHECK)),
]
