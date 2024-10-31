import numpy as np

from aie.helpers.tensortiler.tensortiler2d import TensorTiler2D, TensorTile
from util import construct_test


# RUN: %python %s | FileCheck %s
def ceildiv(a, b):
    return -(a // -b)


def run_checks(n_aie_cols, n_aie_rows, M, N, K, m, n, k):
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    # Define tilers
    c_tiler = TensorTiler2D(M, N, m * n_aie_rows, n)
    c_iter = c_tiler.tile_iter(
        tile_repeat_step_horizontal=n_aie_cols, iter_step=tb_n_rows
    )

    for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
        for pingpong in [0, 1]:
            row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
            tb_n_rows = min([tb_max_n_rows // 2, M // m // n_aie_rows - row_base])
            print(tb_n_rows)
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

                    """
                    assert (reference_access == c_access).all(), (
                        f"C access orders do not match. "
                        f"Expected ({expected_c_tile}), got ({c_tile})"
                    )
                    assert (reference_count == c_count).all()
                    """
                    print(f"Expected: {expected_c_tile}")
                    print(f"Actual: {c_tile}")


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
                                    return


if __name__ == "__main__":
    matrix_whole_array_tiling_sweep()
