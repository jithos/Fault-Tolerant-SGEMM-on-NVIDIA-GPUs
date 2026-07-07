import numpy as np
import os
import sys

# cuda data types [bytes]: unsigned int: 4, int: 4, char: 1, float: 4, long long int: 8
# CUDA data type sizes in [bytes]
LONG_LONG_INT_SIZE = 8
UNSIGNED_INT_SIZE = 4
INT_SIZE = 4
CHAR_SIZE = 1
FLOAT_SIZE = 4

# Folders
BEAM_DATA_FOLDER = "/home/jithin/repos/Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/beam_data/"
BEAM_INPUTS_MATRICES_FOLDER = BEAM_DATA_FOLDER + "exp_inputs_matrices/"
BEAM_RESULTS_FOLDER = BEAM_DATA_FOLDER + "exp_results/"

# Files
BEAM_SEU_DATA_FILE_NAME = "2026-07-07_Tue_08:36:48_exp_Y_seu_data.bin"

# Settings
PRINT_SEU_VALUES = True

if __name__ == "__main__":

    # Check command line arguments
    if len(sys.argv) > 0:
        argv_1 = sys.argv[1]
        if argv_1 == "-h":
            print("Usage: python print_seu_data.py [SEU_DATA_FILE]")
            exit(0)
        elif argv_1 == "-n": # Sort and find the newest file
            ls = os.listdir(BEAM_RESULTS_FOLDER)
            seu_data_files = [f for f in ls if f.endswith("_seu_data.bin")]
            seu_data_files.sort()
            beam_seu_data_file = BEAM_RESULTS_FOLDER + seu_data_files[-1] # Use the newest file
        else:
            beam_seu_data_file = argv_1 # Use file provided by user
    else:
        beam_seu_data_file = BEAM_RESULTS_FOLDER + BEAM_SEU_DATA_FILE_NAME # Use default file

    # Print SEU data from the binary file
    print(f"Reading SEU data from: {beam_seu_data_file}")
    with open(beam_seu_data_file, "rb") as f:
        # Initialize total SEU count and repetitions
        total_seu_count = 0
        total_repetitions = 0

        # Read the binary data until EOF
        while True:
            # Check for EOF
            last_pos = f.tell()  # Store the current position
            c = f.read(1)  # Read a single byte
            if not c:  # If we reached the end of the file, break the loop
                break
            else:
                f.seek(last_pos)  # Move back to the last position

            # Read repetition and seu_count from the binary data
            repetition = int.from_bytes(f.read(INT_SIZE), byteorder='little', signed=True)
            f.read(CHAR_SIZE) # Read comma
            seu_count = int.from_bytes(f.read(LONG_LONG_INT_SIZE), byteorder='little', signed=True)
            f.read(CHAR_SIZE) # Read comma
            print(f"Repetition: {repetition}, SEU Count: {seu_count}")

            # Increment total SEU count and repetitions
            total_seu_count += seu_count
            total_repetitions += 1
            
            # Read error row and column indices
            if seu_count > 0:
                error_col_indices = np.frombuffer(f.read(seu_count * INT_SIZE), dtype=np.int32) # Read column indices
                error_row_indices = np.frombuffer(f.read(seu_count * INT_SIZE), dtype=np.int32) # Read row indices
                error_values = np.frombuffer(f.read(seu_count * FLOAT_SIZE), dtype=np.float32) # Read error values
                if PRINT_SEU_VALUES:
                    print(f"Error Column Indices: {error_col_indices}")
                    print(f"Error Row Indices: {error_row_indices}")
                    print(f"Error Values: {error_values}")

        # Print summary
        print(f"SEU data file: {f.name.split('/')[-1]}")
        print(f"Total Repetitions: {total_repetitions}, Total SEU Count: {total_seu_count}")
