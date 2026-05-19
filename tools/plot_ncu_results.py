import matplotlib.pyplot as plt
import mplfig
import pandas as pd

# ------------------------------- #
# --- Settings from profiling --- #
# ------------------------------- #

KERNEL_TUPLE = [
    # ("sgemm_small", 1),
    # ("sgemm_medium", 2),
    # ("sgemm_large", 3),
    # ("sgemm_tall", 4),
    # ("sgemm_wide", 5),
    # ("sgemm_huge", 6),
    ("ft_sgemm_small", 11),
    ("ft_sgemm_medium", 12),
    ("ft_sgemm_large", 13),
    ("ft_sgemm_tall", 14),
    ("ft_sgemm_wide", 15),
    ("ft_sgemm_huge", 16),
    ("ft_sgemm_medium_96", 17),
]

KERNEL_LIST = [kernel for kernel, id in KERNEL_TUPLE]

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

# --------------------- #
# --- Plot settings --- #
# --------------------- #

SHOW_PLOTS = True
OVERLAY_PLOTS = False
SAVE_IMG = False
PICKLE_PLOT = False

METRICS_TO_PLOT = [
    # "PLOT no overlay",
    "compute_sm_throughput",
    "memory_throughput",
    # "PLOT new figure",
    # "PLOT overlay",
    "duration",
    "achieved_occupancy",
]

# --- Label to Metric mapping
L_2_M = {
    # Column names in CSV
    "kernel_name": "Kernel Name",
    "kernel_size": "Kernel Size",
    "section": "Section",
    "metric_name": "Metric Name",
    "metric_unit": "Metric Unit",
    "value": "Value",
    # Metrics in CSV
    # --- Section: GPU Speed Of Light Throughput ---
    "sm_frequency": "SM Frequency",
    "elapsed_cycles": "Elapsed Cycles",
    "memory_throughput": "Memory Throughput",
    "duration": "Duration",
    "l1_tex_cache_throughput": "L1/TEX Cache Throughput",
    "l2_cache_throughput": "L2 Cache Throughput",
    "sm_active_cycles": "SM Active Cycles",
    "compute_sm_throughput": "Compute (SM) Throughput",
    # --- Section: Launch Statistics ---
    "block_size": "Block Size",
    "function_cache_configuration": "Function Cache Configuration",
    "grid_size": "Grid Size",
    "registers_per_thread": "Registers Per Thread",
    "shared_memory_configuration_size": "Shared Memory Configuration Size",
    "driver_shared_memory_per_block": "Driver Shared Memory Per Block",
    "dynamic_shared_memory_per_block": "Dynamic Shared Memory Per Block",
    "static_shared_memory_per_block": "Static Shared Memory Per Block",
    "number_of_sms": "# SMs",
    "threads": "Threads",
    "uses_green_context": "Uses Green Context",
    "waves_per_sm": "Waves Per SM",
    # --- Section: Occupancy ---
    "block_limit_sm": "Block Limit SM",
    "block_limit_registers": "Block Limit Registers",
    "block_limit_shared_mem": "Block Limit Shared Mem",
    "block_limit_warps": "Block Limit Warps",
    "theoretical_active_warps_per_sm": "Theoretical Active Warps per SM",
    "theoretical_occupancy": "Theoretical Occupancy",
    "achieved_occupancy": "Achieved Occupancy",
    "achieved_active_warps_per_sm": "Achieved Active Warps Per SM",
    # --- Section: GPU and Memory Workload Distribution ---
    "average_l1_active_cycles": "Average L1 Active Cycles",
    "total_l1_elapsed_cycles": "Total L1 Elapsed Cycles",
    "average_l2_active_cycles": "Average L2 Active Cycles",
    "total_l2_elapsed_cycles": "Total L2 Elapsed Cycles",
    "average_sm_active_cycles": "Average SM Active Cycles",
    "total_sm_elapsed_cycles": "Total SM Elapsed Cycles",
    "average_smsp_active_cycles": "Average SMSP Active Cycles",
    "total_smsp_elapsed_cycles": "Total SMSP Elapsed Cycles",
}

# --- Section to Metric mapping
S_2_M = {
    "speed_of_light":           {"name": "Section: GPU Speed Of Light Throughput",          "metrics": ["sm_frequency", "elapsed_cycles", "memory_throughput", "duration", "l1_tex_cache_throughput", "l2_cache_throughput", "sm_active_cycles", "compute_sm_throughput"]},
    "launch_statistics":        {"name": "Section: Launch Statistics",                      "metrics": ["block_size", "function_cache_configuration", "grid_size", "registers_per_thread", "shared_memory_configuration_size", "driver_shared_memory_per_block", "dynamic_shared_memory_per_block", "static_shared_memory_per_block", "number_of_sms", "threads", "uses_green_context", "waves_per_sm"]},
    "occupancy":                {"name": "Section: Occupancy",                              "metrics": ["block_limit_sm", "block_limit_registers", "block_limit_shared_mem", "block_limit_warps", "theoretical_active_warps_per_sm", "theoretical_occupancy", "achieved_occupancy", "achieved_active_warps_per_sm"]},
    "gpu_and_memory_workload":  {"name": "Section: GPU and Memory Workload Distribution",   "metrics": ["average_l1_active_cycles", "total_l1_elapsed_cycles", "average_l2_active_cycles", "total_l2_elapsed_cycles", "average_sm_active_cycles", "total_sm_elapsed_cycles", "average_smsp_active_cycles", "total_smsp_elapsed_cycles"]},
}

# -------------------------------- #
# --- Files, folders and paths --- #
# -------------------------------- #

# Experiment selection
EXPERIMENT_NAME = "vanilla"
# EXPERIMENT_NAME = "detection"

FT_SGEMM_PATH = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build"
EXPERIMENT_PATH = f"{FT_SGEMM_PATH}/ncu_exp_{EXPERIMENT_NAME}"
NCU_RESULTS_CSV_PATH = f"{EXPERIMENT_PATH}/ncu_results_csv"
PLOT_PATH = f"{EXPERIMENT_PATH}/plots"

NCU_CSV_FILE_NAME = "ncu_results.csv"
NCU_CSV_FILE = f"{NCU_RESULTS_CSV_PATH}/{NCU_CSV_FILE_NAME}"

# ------------------------ #
# --- PLOT_PATHHelper functions --- #
# ------------------------ #

def plot_x_y(x_label, y_label, measurement_label, x_data, y_data, measurements, **kwargs):
    # x_label: x-axis values, format: list([x1, x2, x3,...], [xx1, xx2, xx3,...])
    # y_label: y-axis values, format: list([y1, y2, y3,...], [yy1, yy2, yy3,...])
    # labels: separate measurements, format: [l1, l2, l3,...], len(labels) == len(x_label[i]) == len(y_label[i])
    # example: x_label = [[5,3], [6,4]], y_label = [["small_kernel","large_kernel"], ["small_kernel","large_kernel"]], labels = ["big_matrix","small_matrix"]

    # Check if overlaying plots is required
    figure = plt.gcf() if kwargs.get("overlay") else plt.figure()

    # Create plot
    for x, y, measurement in zip(x_data, y_data, measurements):
        plt.plot(
            x,
            y,
            marker='o',
            linestyle='--',
            label=measurement,
            figure=figure
        )

    # Set the title and labels
    plt.title(f"{y_label} vs {x_label}")
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    # Show legend
    plt.legend(title=f"{measurement_label}", bbox_to_anchor=(1, 0.5), loc='center left')

    # Save plot
    if kwargs.get("save_img"):
        plt.savefig(f"{PLOT_PATH}/{y_label}_vs_{x_label}.png")

    # Pickle plot
    if kwargs.get("pickle_plot"):
        mplfig.save_figure(plt.gcf(), f"{PLOT_PATH}/{y_label}_vs_{x_label}.mplpkl")

# Plot metric vs kernel, also overlay plots for different kernel sizes
def plot_metric(df, metric="duration", kernel_names=None, kernel_sizes=None, **kwargs):
    print(f"Plot: {L_2_M[metric]}")

    # Use values for parameters from script settings
    if not kernel_names: kernel_names = KERNEL_LIST
    if not kernel_sizes: kernel_sizes = KERNEL_SIZES
    if not "overlay" in kwargs.keys(): kwargs["overlay"] = OVERLAY_PLOTS
    if not "save_img" in kwargs.keys(): kwargs["save_img"] = SAVE_IMG
    if not "pickle_plot" in kwargs.keys(): kwargs["pickle_plot"] = PICKLE_PLOT

    # Get values for y axis
    y_column = L_2_M["metric_name"]
    y_metric = L_2_M[metric]
    y_values = df[df[y_column] == y_metric]

    # Get values for x axis
    x_column = L_2_M["kernel_name"]
    x_metric = x_column
    x_values = df[x_column]

    # Filter out kernels that are not required
    if kernel_names == None:
        pass
    else:
        df_kernel_names = x_values.unique()
        for kernel in df_kernel_names:
            if not kernel in kernel_names:
                items_to_drop = y_values[y_values[x_column] == kernel].index.values
                y_values = y_values.drop(index=items_to_drop)

    # Align x_values with y_values
    selected_items = list(y_values.index.values)
    x_values = x_values[selected_items]

    # Get list of kernel sizes
    x_labels = df.loc[selected_items, L_2_M["kernel_size"]].astype(int)
    x_labels = x_labels.unique()

    # Group duration values according to kernel sizes
    x_values_list = []
    y_values_list = []
    for label in x_labels:
        if label in kernel_sizes or kernel_sizes == None:
            y_values_for_label = y_values[y_values[L_2_M["kernel_size"]] == label]
            selected_items = list(y_values_for_label.index.values)
            x_values_list += [x_values.loc[selected_items]]
            y_values_list += [y_values.loc[selected_items, L_2_M["value"]].astype(float)]

    plot_x_y("Kernels", L_2_M[metric], "Matrix size", x_values_list, y_values_list, x_labels, **kwargs)

def load_plot_pickle(file):
    fig = mplfig.load_figure(file)

if __name__ == "__main__":
    # load csv file
    df = pd.read_csv(NCU_CSV_FILE)

    overlay = OVERLAY_PLOTS
    for metric in METRICS_TO_PLOT:
        if metric == "PLOT overlay":
            overlay = True
            continue
        elif metric == "PLOT no overlay":
            overlay = False
            continue
        elif metric == "PLOT new figure":
            plt.figure()
            continue
        else:
            plot_metric(df, metric, overlay=overlay)

    # Plot from pickled plot
    # load_plot_pickle(f"{EXPERIMENT_PATH}/plots/Duration_vs_Kernels.mplpkl")
    # load_plot_pickle(f"{EXPERIMENT_PATH}/plots/Achieved Occupancy_vs_Kernels.mplpkl")

    # Show the plot
    if SHOW_PLOTS:
        # plt.tight_layout()
        plt.show()
