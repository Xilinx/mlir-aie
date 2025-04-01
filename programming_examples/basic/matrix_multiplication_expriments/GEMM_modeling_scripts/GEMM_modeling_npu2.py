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

# Give the single AIE core GEMM dimensions
m = 100
k = 60
n = 100

# give the expected AIE single core kernel efficiency
expected_kernel_efficiency = 0.65


# Give the desired M, K, N GEMM sizes
M = 4000
K = 4000
N = 4000

# Give the DDR BW in GB/s
DDR_BW = 50


# Give the AIE frequency in GHz
AIE_freq = 1.8

if in_a_type == "int8":
    in_a_bytes = 1
    # 1024 GOPs is for int8 for each core
    theoretical_GOPs_per_core = 1024

elif in_a_type == "int16":
    in_a_bytes = 2
    # 256 GOPs is for int16 for each core
    theoretical_GOPs_per_core = 256

elif in_a_type == "bf16":

    in_a_bytes = 2
    # 512 GOPs is for bf16 for each core
    theoretical_GOPs_per_core = 512

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


# do the checks with DMA BW for only A and B (later).
# For C is not important at all because if K is big, then there is time to overlap


# Number of AIE cols and rows.
# Start with full array for Strix
aie_rows = 4
aie_cols = 8


# number of Bytes we need to read and write to DDR for matrices A, B and C
DDR_A_size = ((float(M * K * N) / n) / aie_cols) * in_a_bytes

# if K small enough that fits:
# DDR_A_size = float(M * K) * in_a_bytes

DDR_B_size = ((float(M * K * N) / m) / aie_rows) * in_b_bytes

DDR_C_size = float(M * N) * out_c_bytes


# calculate the total DDR size in MB
total_DDR_size = (DDR_A_size + DDR_B_size + DDR_C_size) / 1024 / 1204


# calculate the DDR time in ms
DDR_time = total_DDR_size / DDR_BW


# expected memory bound TOPs
mem_bound_TOPs = (float(M * K * N) * 2) / DDR_time / (10**9)


# theoretical TOPs
theoretical_TOPs = theoretical_GOPs_per_core * AIE_freq * (aie_rows * aie_cols) / 1000


# compute bound TOPs
comp_bound_TOPs = theoretical_TOPs * expected_kernel_efficiency


print(f"DDR total A size = {DDR_A_size/1024/1024} MB")
print(f"DDR total B size = {DDR_B_size/1024/1024} MB")
print(f"DDR total C size = {DDR_C_size/1024/1024} MB")

print(f"Memory Bound performance = {mem_bound_TOPs} TOPs")
print(f"Compute Bound performance = {comp_bound_TOPs} TOPs")
