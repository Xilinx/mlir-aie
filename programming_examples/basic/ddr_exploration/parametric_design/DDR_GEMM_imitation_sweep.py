import subprocess
from pandas import DataFrame
import argparse
import sys
import re
import os


# How to run:
# python3 DDR_GEMM_imitation_sweep.py --npu_columns=4 --b_col_maj=0


def main():

    # Create the parser
    parser = argparse.ArgumentParser(
        description="This script imitates A and B tiled matrix reads from DDR"
    )

    parser.add_argument(
        "--npu_columns",
        type=int,
        choices=[1, 2, 4, 8],
        help="NPU columns. Phoenix: max 4, Strix/Kracken: max 8",
    )
    parser.add_argument(
        "--b_col_maj",
        type=int,
        choices=[0, 1],
        help="Is B col-major? 1 if yes, 0 if no",
    )

    # Parse the arguments
    args = parser.parse_args()

    b_col_maj = args.b_col_maj
    npu_columns = args.npu_columns

    # Define the directory and file path
    directory = "./sweep_results"

    B_layout = "_B_col_maj" if b_col_maj else "_B_row_maj"

    # just set a fixed value of GEMM sizes for this experiment
    M = 8192
    K = 8192
    N = 8192

    # common file name
    file_name = str(M) + "x" + str(K) + "x" + str(N) + B_layout

    # log file
    log_file_name = "DDR_GEMM_sweep_log_" + file_name
    log_file_path = os.path.join(directory, log_file_name)

    # csv file
    csv_file_name = "DDR_GEMM_sweep_" + file_name + ".csv"
    csv_file_path = os.path.join(directory, csv_file_name)

    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # open log file
    file = open(log_file_path, "w")

    # Create an empty DataFrame
    df = DataFrame(
        columns=[
            "m_k_n",
            "Total Size (MB)",
            "Avg DDR BW (GB/s)",
            "Max DDR BW (GB/s)",
            "Min DDR BW (GB/s)",
        ]
    )

    # set makefile var both_row to 0 if B in col-maj, else set to 1
    both_row = 0 if b_col_maj else 1

    # set m to 32 because it won't affect the DDR BW results.
    # Remember m is cannot be contiguous in DDR
    m = 32

    # sweep across k to capture contiguous data for A matrix
    k_values = [2**i for i in range(5, 10)]

    n_values = [2**i for i in range(5, 10)]

    for k in k_values:
        for n in n_values:

            make_variables = [
                f"m={m}",
                f"k={k}",
                f"n={n}",
                f"M={M}",
                f"K={K}",
                f"N={N}",
                f"both_row={both_row}",
                f"npu_cols{npu_columns}",
            ]

            # "make run" command here
            make_command = ["make", "run"]

            # Append the variables to the make command
            full_command = make_command + make_variables

            result = subprocess.run(full_command, capture_output=True, text=True)

            # npu output
            npu_output = result.stdout

            # print here the output into a log file, primarily for debugging/compilation_fail
            file.write(npu_output)

            # find DDR BW
            avg_ddr_bw_pattern = r"Avg DDR BW: ([\d.]+) GB/s"
            max_ddr_bw_pattern = r"Max DDR BW: ([\d.]+) GB/s"
            min_ddr_bw_pattern = r"Min DDR BW: ([\d.]+) GB/s"

            total_size_pattern = r"Total size: (\d+) MB"

            avg_ddr_bw = re.search(avg_ddr_bw_pattern, npu_output)
            max_ddr_bw = re.search(max_ddr_bw_pattern, npu_output)
            min_ddr_bw = re.search(min_ddr_bw_pattern, npu_output)
            total_size_MB = re.search(total_size_pattern, npu_output)

            # if design run successfully, save GOPs
            if avg_ddr_bw and max_ddr_bw and min_ddr_bw:

                df.loc[len(df)] = [
                    str(m) + "_" + str(k) + "_" + str(n),
                    total_size_MB.group(1),
                    avg_ddr_bw.group(1),
                    max_ddr_bw.group(1),
                    min_ddr_bw.group(1),
                ]

            # make clean for next iteration
            # Define the make command for cleaning
            make_command = ["make", "clean"]

            # Run the make clean command
            result = subprocess.run(make_command, capture_output=True, text=True)

            # print here the output into a log file, primarily for debugging/compilation_fail
            file.write(result.stdout)

    # Close the file
    file.close()

    # save the dataframe into a csv file
    df.to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    main()
