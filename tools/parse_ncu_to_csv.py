import numpy as np
import pandas as pd
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
# EXPERIMENT_NAME = "vanilla"
EXPERIMENT_NAME = "detection"

NCU_PATH = "/opt/nvidia/nsight-compute/2024.3.1/target/linux-v4l_l4t-t210-a64/ncu" # Or just use "ncu" if to root privileges needed
FT_SGEMM_PATH = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/build"
EXPERIMENT_PATH = f"{FT_SGEMM_PATH}/ncu_exp_{EXPERIMENT_NAME}"

NCU_RESULTS_PATH = f"{EXPERIMENT_PATH}/ncu_results"
NCU_RESULTS_TXT_PATH = f"{EXPERIMENT_PATH}/ncu_results_txt"
NCU_RESULTS_CSV_PATH = f"{EXPERIMENT_PATH}/ncu_results_csv"

NCU_CSV_FILE = f"{NCU_RESULTS_CSV_PATH}/ncu_results.csv"

# --------------------------- #
# --- Conversion settings --- #
# --------------------------- #

RUN_NCU_2_TXT = True
RUN_TXT_2_CSV = True
NORMALIZE_CSV_UNITS = True # Normalize units in CSV (e.g. ms and us to seconds)

def save_ncu_to_txt(ncu_rep_file, txt_file):
    ncu_cmd = f"{NCU_PATH} --import {ncu_rep_file} > {txt_file}"
    # print(ncu_cmd)
    subprocess.run(ncu_cmd, shell=True)
    print(f" ncu -> txt", end="")
    # print(f": {txt_file}")
    print(", ", end="")

def save_txt_to_csv(txt_file, csv_file, kernel_name, kernel_size):
    with open(txt_file, "r") as f:
        lines = f.readlines()

    # Variables to hold metrics to be saved to dataframe
    section = None
    metric_name = None
    metric_unit = None
    value = None

    # Variables for extracting metrics
    line_offset = 0
    section_offset = 3
    metric_count = 0
    metric_id = 0
    skip_unit_list = []

    # Prepare new dataframe to save metrics
    df = pd.DataFrame()

    for line in lines:
        # Start of new section
        if section == None and "Section: GPU Speed Of Light Throughput" in line:
            section = "speed_of_light"
            line_offset = 0
            section_offset = 3
            metric_count = 8
            metric_id = 0
            skip_unit_list = []
            continue

        # Start of new section
        elif section == None and "Section: Launch Statistics" in line:
            section = "launch_statistics"
            line_offset = 0
            section_offset = 3
            metric_count = 12
            metric_id = 0
            skip_unit_list = [0,1,2,10,11]
            continue

        # Start of new section
        elif section == None and "Section: Occupancy" in line:
            section = "occupancy"
            line_offset = 0
            section_offset = 3
            metric_count = 8
            metric_id = 0
            skip_unit_list = []
            continue

        # Start of new section
        elif section == None and "Section: GPU and Memory Workload Distribution" in line:
            section = "gpu_and_memory_workload"
            line_offset = 0
            section_offset = 3
            metric_count = 8
            metric_id = 0
            skip_unit_list = []
            continue

        # Parse section metrics
        elif section != None:
            # Wait until we get to metrics and values
            if line_offset < section_offset:
                line_offset += 1
                continue

            # Get each metric and value
            # print(f"{line.strip()}")
            if metric_id < metric_count:
                value = line.split()[-1].strip()
                value = value.replace(",", "")
                if metric_id in skip_unit_list:
                    metric_unit = ""
                    metric_name = " ".join(line.split()[:-1]).strip()
                else:
                    metric_unit = line.split()[-2].strip()
                    metric_name = " ".join(line.split()[:-2]).strip()
                # print(f"{metric_name: <40}{metric_unit: <20}{value: <15}")
            metric_id += 1

            # Append to DataFrame
            df = df.append(
                {
                    "Kernel Name": kernel_name,
                    "Kernel Size": kernel_size,
                    "Section": section,
                    "Metric Name": metric_name,
                    "Metric Unit": metric_unit,
                    "Value": value,
                },
                ignore_index=True
            )

            # Reset for next section
            if metric_id == metric_count:
                section = None

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False, header=False, mode="a")

    print(f" txt -> csv", end="")
    # print(f": {csv_file}")

def normalize_csv_units(csv_file):
    # Load dataframe from csv file
    df = pd.read_csv(csv_file)

    norm_count = 0

    # Iterate through metrics and normalize values if needed
    for metric in df["Metric Name"].unique():
        units = df[df["Metric Name"] == metric]["Metric Unit"].unique()
        if len(units) > 1:
            executed = False # To check if any rule was executed

            # Normalize to seconds
            if ["ms", "us"] in units:
                # "ms" to "s"
                selected_items = np.where(np.logical_and(df["Metric Name"] == metric, df["Metric Unit"] == "ms"))
                selected_items = selected_items[0]
                df.loc[selected_items,"Value"] = df.iloc[selected_items]["Value"].astype(float) / 10**3
                df.loc[selected_items,"Metric Unit"] = "s"
                
                # "us" to "s"
                selected_items = np.where(np.logical_and(df["Metric Name"] == metric, df["Metric Unit"] == "us"))
                selected_items = selected_items[0]
                df.loc[selected_items,"Value"] = df.iloc[selected_items]["Value"].astype(float) / 10**6
                df.loc[selected_items,"Metric Unit"] = "s"

                # Check if normalization to seconds worked
                res_units = df[df["Metric Name"] == metric]["Metric Unit"].unique()
                assert res_units == ["s"], f"Normalization to seconds failed: Tired {units} -> ['s'], got {res_units}"
                print(f"\nNormalized '{metric}': ms and us to seconds")
                executed = True
            else:
                print(f"\nNo rule defined to normalize: {units}")

            # Count up if a normalization rule was applied
            if executed: norm_count += 1

    # Save normalized DataFrame to a CSV file
    if norm_count > 0:
        df.to_csv(csv_file, index=False, mode="w")
    else:
        print("\nNothing to normalize")

if __name__ == "__main__":
    if RUN_TXT_2_CSV:
        # Create an empty csv file
        df = pd.DataFrame()
        df["Kernel Name"] = []
        df["Kernel Size"] = []
        df["Section"] = []
        df["Metric Name"] = []
        df["Metric Unit"] = []
        df["Value"] = []
        df.to_csv(NCU_CSV_FILE, index=False, mode="w")

    if RUN_NCU_2_TXT or RUN_TXT_2_CSV:
        # Iterate through all NCU result files, parse them, and save to CSV
        for kernel_name, kernel_id in KERNEL_LIST:
            for size in KERNEL_SIZES:
                print(f"Parsing {kernel_name}, {size}:", end="")

                if RUN_NCU_2_TXT: save_ncu_to_txt(f"{NCU_RESULTS_PATH}/ncu_{kernel_name}_{size}.ncu-rep", f"{NCU_RESULTS_TXT_PATH}/ncu_{kernel_name}_{size}.txt")
                if RUN_TXT_2_CSV: save_txt_to_csv(f"{NCU_RESULTS_TXT_PATH}/ncu_{kernel_name}_{size}.txt", NCU_CSV_FILE, kernel_name, size)
                print()
    
    # Normalize csv if needed
    if NORMALIZE_CSV_UNITS:
        normalize_csv_units(NCU_CSV_FILE)
        print("\nCSV normalization done")

    print("\nFinished parsing all NCU results.")