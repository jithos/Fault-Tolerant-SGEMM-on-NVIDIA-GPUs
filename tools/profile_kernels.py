import subprocess

# ------------------------------- #
# --- Settings from profiling --- #
# ------------------------------- #

KERNEL_LIST = [
    ("sgemm_small", 1),
    ("sgemm_medium", 2),
    ("sgemm_large", 3),
    ("sgemm_tall", 4),
    ("sgemm_wide", 5),
    ("sgemm_huge", 6),
    ("ft_sgemm_small", 11),
    ("ft_sgemm_medium", 12),
    ("ft_sgemm_large", 13),
    ("ft_sgemm_tall", 14),
    ("ft_sgemm_wide", 15),
    ("ft_sgemm_huge", 16),
    ("ft_sgemm_medium_96", 17),
]

KERNEL_SIZES = [
    1024,
    2048,
    3072,
    4096,
    5120,
    6144,
    7168,
    8192,
]

# -------------------------------- #
# --- Files, folders and paths --- #
# -------------------------------- #

# Experiment selection
EXPERIMENT_NAME = "exp_vanilla"
# EXPERIMENT_NAME = "exp_detection"

NCU_PATH = "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu"
FT_SGEMM_PATH = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build"
EXPERIMENT_PATH = f"{FT_SGEMM_PATH}/{EXPERIMENT_NAME}"
NCU_RESULTS_PATH = f"{EXPERIMENT_PATH}/ncu_results"

if __name__ == "__main__":
    for kernel_name, kernel_id in KERNEL_LIST:
        print(f"Profiling {kernel_name} kernel...")

        for size in KERNEL_SIZES:
            print(f"Profiling {kernel_name} with size {size}...")

            ncu_cmd = f"{NCU_PATH} --config-file off \
            --export {NCU_RESULTS_PATH}/ncu_{kernel_name}_{size} \
            --force-overwrite \
            --kernel-name {kernel_name} \
            {FT_SGEMM_PATH}/ft_sgemm {size} {size} {size+1} {kernel_id} {kernel_id}"
            print(ncu_cmd)

            subprocess.run(ncu_cmd, shell=True)

        print(f"Finished profiling {kernel_name} kernel.")