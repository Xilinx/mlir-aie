import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D, TensorTile
from util import construct_test


# RUN: %python %s | FileCheck %s
def ceildiv(a, b):
    return -(a // -b)


def run_checks(n_aie_cols, n_aie_rows, M, N, K, m, n, k):
    tb_max_n_rows = 4

    # Define tilers
    c_tiler = TensorTiler2D(M, N, m * n_aie_rows, n)
    c_iter = c_tiler.tile_iter(tile_repeat_step_horizontal=n_aie_cols)

    for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
        for pingpong in [0, 1]:
            row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
            tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
            if tb_n_rows <= 0:
                # for small input sizes, we may not even need a "pong" iteration
                break

            for col in range(n_aie_cols):
                C_row_offset = row_base * m * n_aie_rows * N
                C_col_offset = col * n
                C_offset = C_col_offset + C_row_offset
                C_sizes = [tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n]
                C_strides = [m * n_aie_rows * N, n * n_aie_cols, N, 1]
                expected_c_tile = TensorTile(
                    M, N, offset=C_offset, sizes=C_sizes, strides=C_strides
                )

                c_tile = next(c_iter)
                if c_tile != expected_c_tile:
                    # equivalence for tensor tile checks offset, size, stride
                    # but there may be different but equivalent transformations

                    reference_access, reference_count = expected_c_tile.access_tensors()
                    c_access, c_count = c_tile.access_tensors()

                    assert (reference_access == c_access).all(), (
                        f"C access orders do not match. "
                        f"Expected ({expected_c_tile}), got ({c_tile})"
                    )
                    assert (reference_count == c_count).all()


# CHECK-LABEL: matrix_whole_array_tiling_sweep
@construct_test
def matrix_whole_array_tiling_sweep():
    n_aie_cols_sweep = [1, 2, 4]  # TODO: when partial, add 3
    n_aie_rows_sweep = [1, 2, 4]  # TODO: when partial, add 3
    M_sweep = range(512, 4096, 512)
    K_sweep = range(512, 4096, 512)
    N_sweep = range(512, 4096, 512)
    m_sweep = [16, 32, 64]
    n_sweep = [16, 32, 64]
    k_sweep = [16, 32, 64]

    for n_aie_cols in n_aie_cols_sweep:
        for n_aie_rows in n_aie_rows_sweep:
            for M in M_sweep:
                for N in N_sweep:
                    for K in K_sweep:
                        for m in m_sweep:
                            for n in n_sweep:
                                for k in k_sweep:
                                    run_checks(
                                        n_aie_cols=n_aie_cols,
                                        n_aie_rows=n_aie_rows,
                                        M=M,
                                        N=N,
                                        K=K,
                                        m=m,
                                        k=k,
                                        n=n,
                                    )
    # XFAIL: *


"""
            print("C Transfers")
            print(f"C Tensor = M x N ({M} x {N}), m x n ({m} x {n}), n_aie_rows={n_aie_rows}, tb_max_n_rows={tb_max_n_rows}")
            for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
                for pingpong in [0, 1]:
                    M // m // n_aie_rows // tb_max_n_rows
                    row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
                    bd_id_base = 8 * pingpong
                    tb_n_rows = min(
                        [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
                    )
                    if tb_n_rows <= 0:
                        # for small input sizes, we may not even need a "pong" iteration
                        break
                    for col in range(n_aie_cols):
                        print(f"\tcol: {col}")

                        # C Output Transfer:
                        # The smallest transfer unit is a (m*n_aie_rows)-x-(n)-sized sub-tile of the matrix.
                        # Transfer one such tile for every (n_aie_cols)-th column, evenly spaced,
                        # then repeat that (tb_n_rows) times for the next contiguous blocks of rows.
                        # Each shim will start at a different column offset, transferring interleaved
                        # columns. For example, shim 0 may transfer the blocks marked 0 below, and shim 1
                        # may transfer the blocks marked 1.
                        #
                        #             N
                        #      ----------------
                        #     |0011    0011    |
                        #     |0011    0011    |
                        #     |0011    0011    |
                        # M   |0011    0011    |
                        #     |                |
                        #     |                |
                        #     |                |
                        #     |                |
                        #      ----------------
                        C_row_offset = row_base * m * n_aie_rows * N
                        C_col_offset = col * n
                        C_offset = C_col_offset + C_row_offset
                        print(f"C_offset={C_offset} (col*n) + (row_base * m * n_aie_rows * N)")
                        print(f"C_sizes={[tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n]} [tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n]")
                        print(f"C_strides={[m * n_aie_rows * N, n * n_aie_cols, N, 1]} [m * n_aie_rows * N, n * n_aie_cols, N, 1]")
                        npu_dma_memcpy_nd(
                            metadata=C_l2l3_fifos[col],
                            bd_id=bd_id_base,
                            mem=C,
                            offsets=[0, 0, 0, C_offset],
                            sizes=[tb_n_rows, N // n // n_aie_cols, m * n_aie_rows, n],
                            strides=[m * n_aie_rows * N, n * n_aie_cols, N, 1],
                        )

                        for tile_row in range(tb_n_rows):

                            # A input transfer:
                            #
                            # The smallest transfer unit is a (m*n_A_tiles_per_shim)-sized sub-tile of the input matrix.
                            # Transfer one such tile for every column, contiguously.
                            # Repeat this transfer with identical tiles a total of (N//n//n_aie_cols) times.
                            # Each shim transfers the tiles for separate rows. For example, shim 0 may transfer the
                            # tiles marked 0 below, and shim 1 may transfer the tiles marked 1.
                            #             K
                            #      ----------------
                            #     |0000000000000000|    (repeated N//n//n_aie_cols times)
                            #     |0000000000000000|
                            #     |1111111111111111|
                            # M   |1111111111111111|
                            #     |                |
                            #     |                |
                            #     |                |
                            #     |                |
                            #      ----------------
                            A_block_offset = (
                                (row_base + tile_row) * n_aie_rows * m * K
                            )  # base address for this transfer block for all BDs
                            A_row_offset = (
                                col * n_A_tiles_per_shim * m * K
                            )  # base address for the shim in this column
                            A_offset = A_block_offset + A_row_offset

                            npu_dma_memcpy_nd(
                                metadata=A_l3l2_fifos[col],
                                bd_id=bd_id_base + 2 * tile_row + 1,
                                mem=A,
                                offsets=[0, 0, 0, A_offset],
                                sizes=[
                                    N // n // n_aie_cols,
                                    K // k,
                                    m * n_A_tiles_per_shim,
                                    k,
                                ],
                                strides=[0, k, K, 1],
                            )

                            # B input transfer:
                            # Transfer the first a (n)-wide block of columns of B,
                            # Then transfer the (n_aie_columns)-th such block, and so on.
                            # Each shim will start at a different column offset.
                            # For example, shim 0 may transfer the tiles marked 0 below,
                            # and shim 1 may transfer the tiles marked 1.
                            #
                            #             N
                            #      ----------------
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            # K   |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #     |0011    0011    |
                            #      ----------------
                            B_col_offset = col * n if not b_col_maj else col * n * K
                            npu_dma_memcpy_nd(
                                metadata=B_l3l2_fifos[col],
                                bd_id=bd_id_base + 2 * tile_row + 2,
                                mem=B,
                                offsets=[0, 0, 0, B_col_offset],
                                sizes=(
                                    [N // n // n_aie_cols, K // k, k, n]
                                    if not b_col_maj
                                    else [N // n // n_aie_cols, K // k, n, k]
                                ),
                                strides=(
                                    [n * n_aie_cols, k * N, N, 1]
                                    if not b_col_maj
                                    else [n * n_aie_cols * K, k, K, 1]
                                ),
                            )
                    if tb > 0 or (tb == 0 and pingpong > 0):
                        print("WAIT")
                        dma_wait(*C_l2l3_fifos)
            print("WAIT")
            dma_wait(*C_l2l3_fifos)
"""
