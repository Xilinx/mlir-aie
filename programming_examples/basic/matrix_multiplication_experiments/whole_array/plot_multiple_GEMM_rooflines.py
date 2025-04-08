import argparse
import pandas as pd
import matplotlib.pyplot as plt



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



def extract_arith_intensity_and_gops(df):
    arith_inten_list = []
    avg_gops_list = []
    M_K_N_list = []
    GMACs_list = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        
        M_K_N = row['M_K_N']
        M_K_N_list.append(M_K_N)

        GMAcs = row['GMACs']
        GMACs_list.append(GMAcs)
        
        # # Split the M_K_N string into separate values
        # M, K, N = M_K_N.split('_')

        # # Convert M, K, N to integers
        # M = int(M)
        # K = int(K)
        # N = int(N)

        arith_intensity = row['Arith Int (OPs/Byte)']

        # Append each arithmetic intensity point into a list
        arith_inten_list.append(arith_intensity)

        # Append each GOPs point into a list
        avg_gops_list.append(row['Avg GOPs'])

    return arith_inten_list, avg_gops_list, M_K_N_list, GMACs_list


def main():

    in_ab_type = "i8"
    out_c_type = "i16"

    
    kernel_eff_B_row = 0.70
    kernel_eff_B_col = 0.69

    csvfile_B_row_maj = './sweep_results/GEMM_sweep_kernel_80x88x96_i8_i16_mtk_352_B_row_maj.csv'
    csvfile_B_col_maj = './sweep_results/GEMM_sweep_kernel_80x88x96_i8_i16_mtk_352_ktn_352_B_col_maj.csv'

    # Read the CSV file and convert to DataFrame
    df_B_row_maj = pd.read_csv(csvfile_B_row_maj)
    df_B_col_maj = pd.read_csv(csvfile_B_col_maj)



    arith_inten_list_B_row_maj, avg_gops_list_B_row_maj, M_K_N_list_B_row_maj, GMACs_list_B_row_maj = extract_arith_intensity_and_gops(df_B_row_maj)

    arith_inten_list_B_col_maj, avg_gops_list_B_col_maj, M_K_N_list_B_col_maj, GMACs_list_B_col_maj = extract_arith_intensity_and_gops(df_B_col_maj)


    # calculte the performance difference

    # # Ensure the lists are of the same length
    # if len(avg_gops_list_B_col_maj) != len(avg_gops_list_B_row_maj):
    #     raise ValueError("The lists must have the same length")
    
    # # Divide the elements of the two lists element-wise
    # division_results = [col / row for col, row in zip(avg_gops_list_B_col_maj, avg_gops_list_B_row_maj)]

    # # Calculate the average of the division results
    # average_division_results = sum(division_results) / len(division_results)

    # # Calculate the average, minimum, and maximum of the division results
    # average_division_results = sum(division_results) / len(division_results)
    # min_division_results = min(division_results)
    # max_division_results = max(division_results)


    # # Find indices where values are lower than 1000
    # indices_between_1000_and_1500 = [index for index, value in enumerate(arith_inten_list_B_col_maj) if (value > 1000 and value < 1500)]

    # for index in indices_between_1000_and_1500:

    #     if (avg_gops_list_B_col_maj[index] > 5000):
    #         print (f"M_K_N = {M_K_N_list_B_col_maj[index]}, Arith Intensity = {arith_inten_list_B_col_maj[index]}, GOPS = {avg_gops_list_B_col_maj[index]}")

    
    # print(f"B col-maj has {average_division_results}x higher performance vs. B row-maj on average")
    # # print(f"Minimum of division results: {min_division_results}")

    # print(f"Max Perfromance B col-maj {max(avg_gops_list_B_col_maj)} GOPs")
    # print(f"Max Perfromance B row-maj {max(avg_gops_list_B_row_maj)} GOPs")
    


    # 16 cores for Phoenix, 1 GHz clock
    comp_bound_peak_GOPs_B_row = compute_bound_peak_GOPs(in_ab_type, kernel_eff_B_row, 16, 1)
    comp_bound_peak_GOPs_B_col = compute_bound_peak_GOPs(in_ab_type, kernel_eff_B_col, 16, 1)

    # peak GOPs
    peak_GOPs = compute_bound_peak_GOPs(in_ab_type, 1.0, 16, 1)


    avg_gops_list_B_row_maj_percentage_peak = [(value / peak_GOPs) * 100 for value in avg_gops_list_B_row_maj]
    avg_gops_list_B_col_maj_percentage_peak = [(value / peak_GOPs) * 100 for value in avg_gops_list_B_col_maj]

    # # GEMM sweep plot
    plt.figure()
    plt.axhline(comp_bound_peak_GOPs_B_row, color='b', linestyle='--', linewidth=2, label='Compute Bound GOPs (B row-maj)')
    plt.axhline(comp_bound_peak_GOPs_B_col, color='g', linestyle='--', linewidth=2, label='Compute Bound GOPs (B col-maj)')
    plt.plot(arith_inten_list_B_row_maj, avg_gops_list_B_row_maj, marker='x', linestyle='', color='b', label='B row-major')
    plt.plot(arith_inten_list_B_col_maj, avg_gops_list_B_col_maj, marker='*', linestyle='', color='g', label='B col-maj')
    plt.xlabel('Arithmetic Intensity (OPs/Byte)')
    plt.ylabel('Average GOPs')
    plt.title('Phoenix GEMM sweep int8/int16')
    # Display the legend
    plt.legend()

    
    plot_file_name = './sweep_results/GEMM_sweep_kernel_80x88x96_i8_i16_mtk_352_ktn_352_both_row_col.png'
    plt.savefig(plot_file_name)


    # # GEMM sweep plot (percentage of peak)
    # plt.figure()
    # plt.axhline(69, color='black', linestyle='--', linewidth=2)
    # plt.plot(arith_inten_list_B_row_maj, avg_gops_list_B_row_maj_percentage_peak, marker='x', linestyle='', color='b', label='B row-major')
    # plt.plot(arith_inten_list_B_col_maj, avg_gops_list_B_col_maj_percentage_peak, marker='*', linestyle='', color='g', label='B col-maj')
    # plt.xlabel('Arithmetic Intensity (OPs/Byte)')
    # plt.ylabel('Average GOPs (% of Peak)')
    # plt.title('Phoenix GEMM Sweep int8/int16')
    # # Display the legend
    # plt.legend()

    # # Increase the y-axis range
    # plt.ylim(0, 76)

    # # Print text above the axhline
    # plt.text((min(arith_inten_list_B_row_maj)+max(arith_inten_list_B_row_maj)/2), 71, 'Compute Bound Line (69% of Peak GOPs)', fontsize=12, ha='center')

    # # Remove ".csv" and replace it with "_plot.png"
    # plot_file_name = './sweep_results/GEMM_sweep_kernel_80x88x96_i8_i16_mtk_352_ktn_352_B_both_row_col_percentage'
    # plt.savefig(plot_file_name)



if __name__ == "__main__":
    main()