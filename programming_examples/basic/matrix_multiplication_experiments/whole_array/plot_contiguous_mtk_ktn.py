import argparse
import pandas as pd
import matplotlib.pyplot as plt


# How to run:
# python3 XXXXXXX --csvfile=./sweep_results/YYYYYYYYYYYYY




def main():

    # Create the parser
    parser = argparse.ArgumentParser(description="Plot GEMM vs mtk, ktn")

    # Add an argument with possible choices for supported datatypes
    parser.add_argument('--csvfile', type=str, help='Give the path to csv file, e.g., ./sweep_results/GEMM_sweep.csv')

    # Parse the arguments
    args = parser.parse_args()

    csvfile = args.csvfile

    # Read the CSV file and convert to DataFrame
    df = pd.read_csv(csvfile)


    mtk_list = []
    ktn_list = []
    avg_gops_list = []

    # Iterate over each row in the dataframe
    for index, row in df.iterrows():
        
        mtk = row['mtk']
        ktn = row['ktn']

        mtk_list.append(mtk)        

        if ktn != "N/A":
            ktn_list.append(ktn)

        # append each GOPs point into a list
        avg_gops_list.append(row['Avg GOPs'])
        


    # GEMM sweep plot
    plt.figure()
    plt.plot(mtk_list, avg_gops_list, marker='*', linestyle='-', color='b')
    plt.xlabel('mtk')
    plt.ylabel('Avg GOPs')
    # plt.title('Arithmetic Intensity vs Avg GOPs')

    # Remove ".csv" and replace it with "_plot.png"
    plot_file_name = csvfile.replace(".csv", "_plot.png")
    plt.savefig(plot_file_name)



if __name__ == "__main__":
    main()