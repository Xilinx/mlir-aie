import argparse
import pandas as pd
import matplotlib.pyplot as plt


# How to run:
# python3 plot_GEMM_roofline.py --in_ab_type=i8 --out_c_type=i16 --kernel_eff=0.69 --csvfile=./sweep_results/GEMM_sweep_kernel_80x88x96_i8_i16_mtk_352_B_row_maj.csv



# calculate the compute bound peak performance based on:
# 1) single AIE core kernel efficiency
# 2) data type 
# 3) number of cores
# 4) freq in GHz
def compute_bound_peak_GOPs(in_ab_type, kernel_eff, n_cores, freq):

    if in_ab_type == "i8":
        MACs_per_cycle = 256
    elif in_ab_type == "i16":
        MACs_per_cycle = 64
    elif in_ab_type == "bf16":
        MACs_per_cycle = 128

    theor_peak_GOPs = (MACs_per_cycle * 2) * n_cores * freq * kernel_eff

    return theor_peak_GOPs



def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="GEMM roofline plot")

    # Add an argument with possible choices for supported datatypes
    parser.add_argument('--in_ab_type', type=str, choices=['i8', 'i16', 'bf16'], help='input a,b data type')
    parser.add_argument('--out_c_type', type=str, choices=['i8', 'i16', 'i32', 'bf16', 'f32'], help='output c data type')
    parser.add_argument('--kernel_eff', type=float, help='Single AIE core kernel efficiency, e.g., 0.70')
    parser.add_argument('--csvfile', type=str, help='Give the path to csv file, e.g., ./sweep_results/GEMM_sweep.csv')

    # Parse the arguments
    args = parser.parse_args()

    in_ab_type = args.in_ab_type
    out_c_type = args.out_c_type
    kernel_eff = args.kernel_eff
    csvfile = args.csvfile

    # Read the CSV file and convert to DataFrame
    df = pd.read_csv(csvfile)


    arith_inten_list = []
    avg_gops_list = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        
        M_K_N = row['M_K_N']
        
        # Split the M_K_N string into separate values
        M, K, N = M_K_N.split('_')

        # Convert M, K, N to integers
        M = int(M)
        K = int(K)
        N = int(N)

        arith_intensity = row['Arith Int (OPs/Byte)']

        # append each arithmetic intensity point into a list
        arith_inten_list.append(arith_intensity)

        # append each GOPs point into a list
        avg_gops_list.append(row['Avg GOPs'])
        

    # 16 cores for Phoenix, 1 GHz clock
    comp_bound_peak_GOPs = compute_bound_peak_GOPs(in_ab_type, kernel_eff, 16, 1)


    # GEMM sweep plot
    plt.figure()
    plt.axhline(comp_bound_peak_GOPs, color='black', linestyle='--', linewidth=2)
    plt.plot(arith_inten_list, avg_gops_list, marker='o', linestyle='', color='b')
    plt.xlabel('Arithmetic Intensity (OPs/Byte)')
    plt.ylabel('Average GOPs')
    

    # Remove ".csv" and replace it with "_plot.png"
    plot_file_name = csvfile.replace(".csv", "_plot.png")
    plt.savefig(plot_file_name)



if __name__ == "__main__":
    main()