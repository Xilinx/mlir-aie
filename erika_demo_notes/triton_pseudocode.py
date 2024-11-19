import triton
import triton.language as tl
import torch

# Set up device and data
device = "cuda"
M = 16
N = 128
matrix = torch.rand((M, N), device=device, dtype=torch.float32)  # Matrix on GPU
bias = torch.tensor(3.14, device=device, dtype=torch.float32)  # Scalar bias


# Define Triton kernel to add bias
@triton.jit
def add_bias_kernel(
    matrix_ptr, bias_ptr, M, N, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Get the block's global row index
    pid = tl.program_id(0)

    # Calculate tile index in each dimension, used to select data to process
    num_tiles_m = tl.cdiv(M, BLOCK_SIZE_M)
    pid_tile_offset_m = pid % num_tiles_m
    pid_tile_offset_n = pid // num_tiles_m

    # We divide the matrix into blocks (we will load and process one block at a time)
    offs_m = pid_tile_offset_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_tile_offset_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    block_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Load the matrix into a tile
    block_data = tl.load(matrix_ptr, mask=block_mask)

    # Add the bias
    block_data += tl.load(bias_ptr)

    # Store the output
    tl.store(matrix_ptr, block_data, mask=block_mask)


# Launch the Triton kernel to add the bias
BLOCK_SIZE_M = 8
BLOCK_SIZE_N = 16
grid = (M // BLOCK_SIZE_M, N / BLOCK_SIZE_N)  # Number of blocks

# Run the kernel
add_bias_kernel[grid](
    matrix.data_ptr(), bias.data_ptr(), M, N, BLOCK_SIZE_M, BLOCK_SIZE_N
)

# Optionally, print the result to verify
print(matrix[:5, :5])  # Print a small portion of the matrix for inspection
