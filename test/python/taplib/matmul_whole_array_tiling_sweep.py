#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import random

from aie.helpers.taplib import TensorAccessPattern, TensorAccessSequence, TensorTiler2D
from aie.helpers.taplib.utils import ceildiv
from util import construct_test

# RUN: %python %s | FileCheck %s


def matmul_tiler_helper(M, K, N, m, k, n, n_aie_cols, b_col_maj, n_aie_rows):
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    n_A_tiles_per_shim = n_aie_rows // n_aie_cols
    tb_max_n_rows = 4
    tb_n_rows = tb_max_n_rows // 2

    A_ordered_tiles = []
    B_ordered_tiles = []
    C_ordered_tiles = []

    A_tiles = TensorTiler2D.group_tiler(
        (M, K),  # Size of A matrix
        (m * n_A_tiles_per_shim, k),  # Size of A (smallest) tile
        (1, K // k),  # Size of "group" of tiles
        pattern_repeat=N
        // n
        // n_aie_cols,  # Repeat data so can distribute across whole column
    )
    if b_col_maj:
        # These assertions are probably too broad.
        assert m % 32 == 0
        assert k % 32 == 0
        assert n % 32 == 0

        B_tiles = TensorTiler2D.step_tiler(
            (K, N),  # Size of B matrix
            (k, n),  # Size of B tile
            tile_group_repeats=(
                K // k // n_aie_cols,
                N // n,
            ),  # Number of tiles per transfer in each dimension (whole col, partial row)
            tile_group_steps=(
                n_aie_cols,
                1,
            ),  # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
        )
    else:
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),  # Size of B matrix
            (k, n),  # Size of B tile
            tile_group_repeats=(
                K // k,
                N // n // n_aie_cols,
            ),  # Number of tiles per transfer in each dimension (whole col, partial row)
            tile_group_steps=(
                1,
                n_aie_cols,
            ),  # Contiguous tile group in col, but send every n_aie_cols-th tile in the row
            tile_group_col_major=True,  # Send all tiles in column before moving on to next column
        )
    C_tiles = TensorTiler2D.step_tiler(
        (M, N),  # Size of C matrix
        (m * n_aie_rows, n),  # Size of C tile
        tile_group_repeats=(
            tb_n_rows,
            N // n // n_aie_cols,
        ),  # Number of tiles per transfer in each dimension (partial col, partial row)
        tile_group_steps=(
            1,
            n_aie_cols,
        ),  # Collect every n_aie_cols row at a time (mirroring how we sent in B data)
    )
    c_index = 0

    for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
        for pingpong in [0, 1]:
            if c_index >= len(C_tiles):
                # May not have pong iteration in some cases
                break
            row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
            current_tb_n_rows = min(
                [tb_max_n_rows // 2, M // m // n_aie_rows - row_base]
            )

            for col in range(n_aie_cols):
                C_ordered_tiles.append(C_tiles[c_index])
                c_index += 1

                for tile_row in range(current_tb_n_rows):
                    tile_offset = (row_base + tile_row) * n_aie_cols + col
                    A_ordered_tiles.append(A_tiles[tile_offset])
                    B_ordered_tiles.append(B_tiles[col])

    A_ordered_tiles = TensorAccessSequence.from_taps(A_ordered_tiles)
    B_ordered_tiles = TensorAccessSequence.from_taps(B_ordered_tiles)
    C_ordered_tiles = TensorAccessSequence.from_taps(C_ordered_tiles)
    return A_ordered_tiles, B_ordered_tiles, C_ordered_tiles


def matmul_reference(M, K, N, m, k, n, n_aie_cols, b_col_maj, n_aie_rows):
    assert (
        M % (m * n_aie_rows) == 0
    ), """A must be tileable into (m * n_aie_rows, k)-sized blocks"""
    assert K % k == 0
    assert (
        N % (n * n_aie_cols) == 0
    ), """B must be tileable into (k, n * n_aie_cols)-sized blocks"""

    if b_col_maj:
        # These assertions are probably too broad.
        assert m % 32 == 0
        assert k % 32 == 0
        assert n % 32 == 0
    n_A_tiles_per_shim = n_aie_rows // n_aie_cols

    A_tiles = []
    B_tiles = []
    C_tiles = []

    tb_max_n_rows = 4
    for tb in range(ceildiv(M // m // n_aie_rows, tb_max_n_rows)):
        for pingpong in [0, 1]:
            M // m // n_aie_rows // tb_max_n_rows
            row_base = tb * tb_max_n_rows + pingpong * tb_max_n_rows // 2
            bd_id_base = 8 * pingpong
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
                c_tile = TensorAccessPattern(
                    (M, N), C_offset, sizes=C_sizes, strides=C_strides
                )
                C_tiles.append(c_tile)

                for tile_row in range(tb_n_rows):

                    A_block_offset = (
                        (row_base + tile_row) * n_aie_rows * m * K
                    )  # base address for this transfer block for all BDs
                    A_row_offset = (
                        col * n_A_tiles_per_shim * m * K
                    )  # base address for the shim in this column
                    A_offset = A_block_offset + A_row_offset
                    A_sizes = [
                        N // n // n_aie_cols,
                        K // k,
                        m * n_A_tiles_per_shim,
                        k,
                    ]
                    A_strides = [0, k, K, 1]
                    a_tile = TensorAccessPattern(
                        (M, K), A_offset, sizes=A_sizes, strides=A_strides
                    )
                    A_tiles.append(a_tile)

                    B_col_offset = col * n if not b_col_maj else col * n * K
                    B_sizes = (
                        [N // n // n_aie_cols, K // k, k, n]
                        if not b_col_maj
                        else [N // n // n_aie_cols, K // k, n, k]
                    )
                    B_strides = (
                        [n * n_aie_cols, k * N, N, 1]
                        if not b_col_maj
                        else [n * n_aie_cols * K, k, K, 1]
                    )
                    b_tile = TensorAccessPattern(
                        (K, N), B_col_offset, sizes=B_sizes, strides=B_strides
                    )
                    B_tiles.append(b_tile)

    A_tiles = TensorAccessSequence.from_taps(A_tiles)
    B_tiles = TensorAccessSequence.from_taps(B_tiles)
    C_tiles = TensorAccessSequence.from_taps(C_tiles)
    return A_tiles, B_tiles, C_tiles


# CHECK-LABEL: matrix_vector_tiling_sweep
@construct_test
def matrix_vector_tiling_sweep():
    n_aie_cols_sweep = [2, 4]  # Note: reduced number of tests by removing 1
    n_aie_rows_sweep = [4]
    M_sweep = range(512, 4096, 512)
    m_sweep = [
        32
    ]  # [16, 32] # Note: reduced number of tests to reduce time to run tests
    K_sweep = range(512, 4096, 512)
    k_sweep = [16, 32]
    N_sweep = range(512, 4096, 512)
    n_sweep = [16, 32]
    b_col_maj = False

    for n_aie_cols in n_aie_cols_sweep:
        for n_aie_rows in n_aie_rows_sweep:
            for M in M_sweep:
                for K in K_sweep:
                    for N in N_sweep:
                        for m in m_sweep:
                            for k in k_sweep:
                                for n in n_sweep:
                                    actual_A_tiles, actual_B_tiles, actual_C_tiles = (
                                        matmul_reference(
                                            M,
                                            K,
                                            N,
                                            m,
                                            k,
                                            n,
                                            n_aie_cols,
                                            b_col_maj,
                                            n_aie_rows,
                                        )
                                    )
                                    new_A_tiles, new_B_tiles, new_C_tiles = (
                                        matmul_tiler_helper(
                                            M,
                                            K,
                                            N,
                                            m,
                                            k,
                                            n,
                                            n_aie_cols,
                                            b_col_maj,
                                            n_aie_rows,
                                        )
                                    )

                                    assert actual_A_tiles == new_A_tiles
                                    assert actual_C_tiles == new_C_tiles

                                    # B sizes/strides differ and it causes the tests to run a long time to
                                    # check functional equivalency so we only do a few checks for functional equivalency
                                    assert len(actual_B_tiles) == len(new_B_tiles)
                                    if M <= 1024 and N <= 1024 and K <= 1024:
                                        # Just check one random tile
                                        rand_idx = random.randrange(len(actual_B_tiles))
                                        assert new_B_tiles[
                                            rand_idx
                                        ].compare_access_orders(
                                            actual_B_tiles[rand_idx]
                                        )

    # CHECK: Pass!
    print("Pass!")


# CHECK-LABEL: matrix_vector_tiling_b_col_major
@construct_test
def matrix_vector_tiling_b_col_major():
    M = 256
    K = 256
    N = 256
    m = 32
    k = 32
    n = 32
    n_aie_cols = 4
    n_aie_rows = 4
    b_col_maj = True

    actual_A_tiles, actual_B_tiles, actual_C_tiles = matmul_reference(
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        b_col_maj,
        n_aie_rows,
    )
    new_A_tiles, new_B_tiles, new_C_tiles = matmul_tiler_helper(
        M,
        K,
        N,
        m,
        k,
        n,
        n_aie_cols,
        b_col_maj,
        n_aie_rows,
    )

    assert actual_A_tiles == new_A_tiles
    assert actual_C_tiles == new_C_tiles

    # B sizes/strides differ so check each tile for functional equivalence
    assert len(actual_B_tiles) == len(new_B_tiles)
    for actual_tile, new_tile in zip(actual_B_tiles, new_B_tiles):
        assert actual_tile.compare_access_orders(new_tile)

    # CHECK: Pass!
    print("Pass!")
