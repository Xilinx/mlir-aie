import subprocess
from pandas import DataFrame
import argparse
import sys
import re
import os


# How to run:
# python3 GEMM_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=0 --m=80 --k=88 --n=96 --cont_coeff=4


# arithmetic intensity is defined as operations per byte
# assumes A and B have the same data type
def arithmetic_intensity(M, K, N, in_ab_type, out_c_type):
    
    if in_ab_type == "i8":
        in_ab_bytes = 1
    elif in_ab_type == "i16":
        in_ab_bytes = 2
    elif in_ab_type == "bf16":
        in_ab_bytes = 2

    if out_c_type == "i8":
        out_c_bytes = 1
    elif out_c_type == "i16":
        out_c_bytes = 2
    elif out_c_type == "bf16":
        out_c_bytes = 2
    elif out_c_type == "i32":
        out_c_bytes = 4 
    elif out_c_type == "f32":
        out_c_bytes = 4

    # number of operations
    n_operations = (M * K * N) * 2

    n_bytes_A = (M * K) * in_ab_bytes
    n_bytes_B = (K * N) * in_ab_bytes
    n_bytes_C = (M * N) * out_c_bytes

    return n_operations/(n_bytes_A + n_bytes_B + n_bytes_C)


def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="GEMM sweep script")


    # Add an argument with possible choices for supported datatypes
    # Assumes A and B have the same datatype
    parser.add_argument('--in_ab_type', type=str, choices=['i8', 'i16', 'bf16'], help='input a,b data type')
    parser.add_argument('--out_c_type', type=str, choices=['i8', 'i16', 'i32', 'bf16', 'f32'], help='output c data type')
    parser.add_argument('--b_col_maj', type=int, choices=[0, 1], help='Is B col-major? 1 if yes, 0 if no')
    parser.add_argument('--m', type=int, help='Single core m dimension')
    parser.add_argument('--k', type=int, help='Single core k dimension')
    parser.add_argument('--n', type=int, help='Single core n dimension')
    parser.add_argument('--cont_coeff', type=int, default=4, help='Coefficient that specifies contiguous data. mtk, ktn = cont_coeff * k')


    # Parse the arguments
    args = parser.parse_args()

    in_ab_type = args.in_ab_type
    out_c_type = args.out_c_type
    b_col_maj = args.b_col_maj
    m = args.m
    k = args.k
    n = args.n
    cont_coeff = args.cont_coeff


    # set the default make variables, which don't change during sweep
    # Modify accordingly depending on what you want to run, e.g., specificy the number of iters
    default_make_vars = ['n_aie_cols=4', 'n_aie_rows=4', 'precompiled_flag=1', 'warmup=10', 'iters=100']


    # Just set mtk, and ktn to make sure it's more than 256B (burst length)
    mtk = cont_coeff * k
    ktn = cont_coeff * k

    # Specify starting point, step and max below
    # 4 rows for phoenix
    M_min = 4 * m
    M_step = 2 * M_min
    M_max = 8000

    # 4 columns for phoenix
    N_min = 4 * n
    N_step = 2 * N_min
    N_max = 8000

    K_min = mtk
    K_step = 4 * K_min
    K_max = 8000


    # Define the directory and file path
    directory = './sweep_results'

    B_layout = '_B_col_maj' if b_col_maj else '_B_row_maj'

    ktn_str = '_ktn_' + str(ktn) if b_col_maj else ''

    # common file name
    file_name = str(m) + 'x' + str(k) + 'x' + str(n) + '_' + in_ab_type + '_' + out_c_type + '_mtk_' + str(mtk) + ktn_str + B_layout

    # log file
    log_file_name = 'GEMM_sweep_log_kernel_' + file_name
    log_file_path = os.path.join(directory, log_file_name)

    # csv file
    csv_file_name = 'GEMM_sweep_kernel_' + file_name + '.csv'
    csv_file_path = os.path.join(directory, csv_file_name)


    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)

    # open log file
    file = open(log_file_path, 'w')

    # Create an empty DataFrame 
    df = DataFrame(columns=['M_K_N', 'GMACs', 'Arith Int (OPs/Byte)', 'Avg GOPs', 'Min GOPs', 'Max GOPs'])

    for M in range(M_min, M_max, M_step):
        for K in range(K_min, K_max, K_step):
            for N in range(N_min, N_max, N_step):

                sweep_make_vars = [f'm={m}', f'k={k}', f'n={n}', f'mtk={mtk}', f'ktn={ktn}', f'M={M}', f'K={K}', f'N={N}', 
                                f'b_col_maj={b_col_maj}', f'dtype_in={in_ab_type}', f'dtype_out={out_c_type}']

                # Append all makefile variables
                make_variables = default_make_vars + sweep_make_vars

                # Split the runargs variable into separate arguments
                runargs = make_variables.pop().split('=', 1)
                runargs_key = runargs[0]
                runargs_value = runargs[1].split()

                # "make -f Makefile_chess run" command here
                make_command = ['make', '-f', 'Makefile_chess', 'run']

                # Append the variables to the make command
                full_command = make_command + make_variables + [f'{runargs_key}={arg}' for arg in runargs_value]

                result = subprocess.run(full_command, capture_output=True, text=True)

                # npu output
                npu_output = result.stdout

                # print here the output into a log file, primarily for debugging/compilation_fail
                file.write(npu_output)

                # find GFLOPs
                avg_gflops_pattern = r"Avg NPU gflops:\s*([\d.]+)"
                max_gflops_pattern = r"Max NPU gflops:\s*([\d.]+)"
                min_gflops_pattern = r"Min NPU gflops:\s*([\d.]+)"

                avg_gflops = re.search(avg_gflops_pattern, npu_output)
                max_gflops = re.search(max_gflops_pattern, npu_output)
                min_gflops = re.search(min_gflops_pattern, npu_output)


                # if design run successfully, save GOPs
                if avg_gflops and max_gflops and min_gflops:

                    arith_intensity = arithmetic_intensity(M, K, N, in_ab_type, out_c_type)

                    df.loc[len(df)] = [str(M) + '_' + str(K) + '_' + str(N), M*K*N/1e9, arith_intensity, avg_gflops.group(1), min_gflops.group(1), max_gflops.group(1)]
                
                # make clean for next iteration
                # Define the make command for cleaning
                make_command = ['make', 'clean']

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