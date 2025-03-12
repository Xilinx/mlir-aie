import subprocess
from pandas import DataFrame
import argparse
import sys
import re
import os


# How to run:
# python3 contiguous_mtk_ktn_sweep.py --in_ab_type=i8 --out_c_type=i16 --b_col_maj=0 --m=80 --k=88 --n=96 --M=4160 --K=4224 --N=4224


def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="Script to find how many contiguous data needed")
    
    
    # Add an argument with possible choices for supported datatypes
    # Assumes A and B have the same datatype
    parser.add_argument('--in_ab_type', type=str, choices=['i8', 'i16', 'bf16'], help='input a,b data type')
    parser.add_argument('--out_c_type', type=str, choices=['i8', 'i16', 'i32', 'bf16', 'f32'], help='output c data type')
    parser.add_argument('--b_col_maj', type=int, choices=[0, 1], help='Is B col-major? 1 if yes, 0 if no')
    parser.add_argument('--m', type=int, help='Single core m dimension')
    parser.add_argument('--k', type=int, help='Single core k dimension')
    parser.add_argument('--n', type=int, help='Single core n dimension')
    parser.add_argument('--M', type=int, help='Final M dimension')
    parser.add_argument('--K', type=int, help='Final K dimension')
    parser.add_argument('--N', type=int, help='Final N dimension')
    
    
    
    # Parse the arguments
    args = parser.parse_args()
    
    in_ab_type = args.in_ab_type
    out_c_type = args.out_c_type
    b_col_maj = args.b_col_maj
    m = args.m
    k = args.k
    n = args.n
    M = args.M
    K = args.K
    N = args.N
    
    
    # set the default make variables, which don't change during sweep
    # Modify accordingly depending on what you want to run, e.g., specificy the number of iters
    default_make_vars = ['n_aie_cols=4', 'n_aie_rows=4', 'precompiled_flag=1', 'warmup=10', 'iters=100']
    
    
    
    # Define the directory and file path
    directory = './sweep_results'
    
    B_layout = '_B_col_maj' if b_col_maj else '_B_row_maj'
    
    
    # common file name
    file_name = str(M) + 'x' + str(K) + 'x' + str(N) + '_' + str(m) + 'x' + str(k) + 'x' + str(n) + '_' + in_ab_type + '_' + out_c_type + B_layout
    
    # log file
    log_file_name = 'contiguous_sweep_log_' + file_name
    log_file_path = os.path.join(directory, log_file_name)
    
    # csv file
    csv_file_name = 'contiguous_sweep_' + file_name + '.csv'
    csv_file_path = os.path.join(directory, csv_file_name)
    
    
    # Create the directory if it doesn't exist
    os.makedirs(directory, exist_ok=True)
    
    # open log file
    file = open(log_file_path, 'w')
    
    # Create an empty DataFrame 
    df = DataFrame(columns=['mtk', 'ktn', 'Avg GOPs', 'Min GOPs', 'Max GOPs'])
    
    
    for mtk in range(k, K, k):
    
        # for each valid mtk
        if K % mtk == 0:
    
            # set ktn to mtk if needed for B in col-maj
            ktn = mtk
            
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
    
                df.loc[len(df)] = [mtk, ktn if b_col_maj else 'N/A' , avg_gflops.group(1), min_gflops.group(1), max_gflops.group(1)]
    
            
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