from pandas import DataFrame
import argparse
import sys


# Create the parser
parser = argparse.ArgumentParser(
    description="Script to find optimal single AIE MatMul sizes for various data types"
)

# Add an argument with possible choices for supported datatypes
parser.add_argument(
    "--in_a_type", type=str, choices=["int8", "int16", "bf16"], help="input a data type"
)
parser.add_argument(
    "--in_b_type", type=str, choices=["int8", "int16", "bf16"], help="input b data type"
)
parser.add_argument(
    "--out_c_type",
    type=str,
    choices=["int8", "int16", "int32", "bf16", "fp32"],
    help="output c data type",
)


# Parse the arguments
args = parser.parse_args()


in_a_type = args.in_a_type
in_b_type = args.in_b_type
out_c_type = args.out_c_type

# provide here minimum (initial) values of m,k,n to enable the optimization
# m,k,n_{min} need to be multiples of r,s,t, respectively
if in_a_type == "int8" and in_b_type == "int8" and out_c_type == "int32":
    m_min = 64
    k_min = 64
    n_min = 64
if in_a_type == "int8" and in_b_type == "int8" and out_c_type == "int16":
    m_min = 112
    k_min = 24
    n_min = 96
elif in_a_type == "int8" and in_b_type == "int8" and out_c_type == "int8":
    m_min = 144
    k_min = 24
    n_min = 144
elif in_a_type == "bf16" and in_b_type == "bf16" and out_c_type == "bf16":
    m_min = 80
    k_min = 40
    n_min = 80
elif in_a_type == "bf16" and in_b_type == "bf16" and out_c_type == "fp32":
    m_min = 64
    k_min = 40
    n_min = 64
elif in_a_type == "int16" and in_b_type == "int16" and out_c_type == "int32":
    m_min = 48
    k_min = 96
    n_min = 48

# provide an expected kernel efficiency to help the
# model achieve optimal overlap
# typically, around 0.9, but might be more or less
# depending on the kernel
expected_kernel_efficiency = 0.40

if in_a_type == "int8":
    in_a_bytes = 1
elif in_a_type == "int16":
    in_a_bytes = 2
elif in_a_type == "bf16":
    in_a_bytes = 2

if in_b_type == "int8":
    in_b_bytes = 1
elif in_b_type == "int16":
    in_b_bytes = 2
elif in_b_type == "bf16":
    in_b_bytes = 2


if out_c_type == "int8":
    out_c_bytes = 1
elif out_c_type == "int16":
    out_c_bytes = 2
elif out_c_type == "bf16":
    out_c_bytes = 2
elif out_c_type == "int32":
    out_c_bytes = 4
elif out_c_type == "fp32":
    out_c_bytes = 4


if in_a_type == "int16" and in_b_type == "int16":
    r = 4
    s = 4
    t = 8
    peak_MACs_per_cycle = 128

elif in_a_type == "bf16" and in_b_type == "bf16":
    r = 8
    s = 8
    t = 8
    # set here something higher than peak attained
    peak_MACs_per_cycle = 180

elif in_a_type == "int8" and in_b_type == "int8":
    r = 8
    s = 8
    t = 8
    peak_MACs_per_cycle = 512

else:
    sys.exit("Currently only looking at same a and b data types.")


# DMA BW is given in Bytes/cycle
DMA_BW = 8


# Create an empty DataFrame
df = DataFrame(
    columns=[
        "m_k_n",
        "MACs",
        "mxn",
        "Data_mem_double_buff",
        "A_mem",
        "B_mem",
        "C_mem",
        "Ideal Cycles",
    ]
)


# check if desired intial sizes divisible by the AIE API size
assert m_min % r == 0, f"{m_min} is not divisible by {r}."
assert k_min % s == 0, f"{k_min} is not divisible by {s}."
assert n_min % t == 0, f"{n_min} is not divisible by {t}."


for m in range(m_min, 256, r):
    for k in range(k_min, 256, s):
        for n in range(n_min, 256, t):

            # find the expected kernel cycles
            expected_kernel_cycles = (
                m * k * n / peak_MACs_per_cycle
            ) / expected_kernel_efficiency

            # single buffer AIE-ML memory
            A_mem_size = m * k * in_a_bytes
            B_mem_size = k * n * in_b_bytes
            C_mem_size = m * n * out_c_bytes

            # total single buffer memory
            single_mem_size = A_mem_size + B_mem_size + C_mem_size

            # find the number of cycles for DMAs of each matrix
            # use only inputs and not the output C, because of accumulation
            # across C, there is time to write data out onto the other buffer
            # if big K is sufficiently large (which will be compared to small k)
            DMA_in_a_cycles = A_mem_size / DMA_BW
            DMA_in_b_cycles = B_mem_size / DMA_BW
            # DMA_out_c_cycles = C_mem_size/DMA_BW

            # ensure not I/O bound (DMA_BW basically)
            if (
                DMA_in_a_cycles <= expected_kernel_cycles
                and DMA_in_b_cycles <= expected_kernel_cycles
            ):
                # and DMA_out_c_cycles <= expected_kernel_cycles):

                # constraint up to 64KB - 1KB = 63KB for stack (1KB)
                # use less than 63KB for memory allocation to work
                if (2 * single_mem_size) < 63 * 1024:

                    # All int8, int16, bf16 singel AIE MatMul kenrel are unrolled 2 times in m and n dimension

                    if (m % (2 * r) == 0) and (n % (2 * t) == 0):

                        # add valid solution to dataframe
                        df.loc[len(df)] = [
                            str(m) + "_" + str(k) + "_" + str(n),
                            m * k * n,
                            m * n,
                            2 * single_mem_size / 1024,
                            A_mem_size / 1024,
                            B_mem_size / 1024,
                            C_mem_size / 1024,
                            (m * k * n / peak_MACs_per_cycle),
                        ]

                        # print(f"m = {m}, k = {k}, n = {n}")
                        # print((2*single_mem_size)/1024)
                        # # print(expected_kernel_cycles)
                        # print()


# short first by 'MACs' and then by 'mxn'
sorted_df = df.sort_values(by=["MACs", "mxn"], ascending=[False, False])
print(sorted_df.head(60))
