import inputimeout
import subprocess

# NOTE: Run this python script with "sudo -E" (i.e. "sudo -E python profile_kernels.py"), as ncu and other CLI commands required root access.

# ------------------------------- #
# --- Settings from profiling --- #
# ------------------------------- #

GPU_GLITCH_TIMEOUT = 10 # seconds

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

METRICS = [
    "gpu__time_active.sum",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.avg",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.avg.peak_sustained_active.per_second",
    "sm__sass_thread_inst_executed_op_fp16_pred_on.avg.pct_of_peak_sustained_active",
]

# -------------------------------- #
# --- Files, folders and paths --- #
# -------------------------------- #

# Experiment selection
# EXPERIMENT_NAME = "vanilla"
EXPERIMENT_NAME = "detection"

NCU_PATH = "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu"
FT_SGEMM_PATH = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs"
EXPERIMENT_PATH = f"{FT_SGEMM_PATH}/profiling_results/ncu_exp_{EXPERIMENT_NAME}"
NCU_RESULTS_PATH = f"{EXPERIMENT_PATH}/ncu_results"

METRIC_CSV_FILE = f"{EXPERIMENT_PATH}/ncu_metric_results.csv"

# ------------------------ #
# --- Helper functions --- #
# ------------------------ #

def assert_no_gpu_glitch():
    try:
        inputimeout.inputimeout(prompt="Press Enter to confirm no GPU glitch occured...", timeout=GPU_GLITCH_TIMEOUT)
    except inputimeout.TimeoutOccurred:
        print("\nAssuming GPU glitch occurred. Logging out from linux session to reset GPU state.")
        subprocess.run("sudo service gdm3 restart", shell=True)

def profile_metrics(metrics, csv_file=None):
    csv_arg = f"{csv_file}" if csv_file else METRIC_CSV_FILE

    with open(csv_file, 'r') as f:
        lines = f.readlines()
    if len(lines) == 0 or not lines[0].startswith('"ID","Process ID"'):
        with open(csv_file, 'w') as f:
            f.write(f'"ID","Process ID","Process Name","Host Name","Kernel Name","Context","Stream","Block Size","Grid Size","Device","CC","Section Name","Metric Name","Metric Unit","Metric Value"\n')

    print(f"Profiling SINGLE METRICS...")
    for kernel_name, kernel_id in KERNEL_LIST:
        print(f"- {kernel_name}", end="")
        for size in KERNEL_SIZES:
            print(f", {size}", end="")

            ncu_cmd = f"{NCU_PATH} \
            --config-file off \
            --csv \
            --metrics {','.join(metrics)} \
            --kernel-name {kernel_name} \
            {FT_SGEMM_PATH}/ft_sgemm {size} {size} {size+1} {kernel_id} {kernel_id} \
            | grep '127.0.0.1' \
            >> {csv_arg}"
            # print(f"\n{ncu_cmd}")

            subprocess.run(ncu_cmd, shell=True)
        print(f" -> DONE")

    assert_no_gpu_glitch()

def profile_basic_sections():
    print(f"Profiling BASIC SECTIONS...")
    for kernel_name, kernel_id in KERNEL_LIST:
        print(f"- {kernel_name}", end="")
        for size in KERNEL_SIZES:
            print(f", {size}", end="")

            ncu_cmd = f"{NCU_PATH} --config-file off \
            --export {NCU_RESULTS_PATH}/ncu_{kernel_name}_{size} \
            --force-overwrite \
            --kernel-name {kernel_name} \
            {FT_SGEMM_PATH}/ft_sgemm {size} {size} {size+1} {kernel_id} {kernel_id}"
            # print(f"\n{ncu_cmd}")

            subprocess.run(ncu_cmd, shell=True)
        print(f" -> DONE")

    assert_no_gpu_glitch()

if __name__ == "__main__":
    profile_basic_sections()
    # profile_metrics(METRICS, "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build/ncu_results.csv")
