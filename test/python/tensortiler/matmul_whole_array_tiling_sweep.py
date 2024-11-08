#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np
from aie.helpers.tensortiler import TensorTile, TensorTileSequence, TensorTiler2D
from aie.helpers.tensortiler.utils import ceildiv


def my_matmul_tiler(M, K, N, m, k, n, n_aie_cols, b_col_maj, n_aie_rows):
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

    # Repeat tile pattern across whole column; send n_aie_row numbers of copies of each column
    A_tiles = TensorTiler2D.group_tiler(
        (M, K),
        (m * n_A_tiles_per_shim, k),
        (1, K // k),
        pattern_repeat=N // n // n_aie_cols,
    )
    # Reorder to be: [ping, pong] for aie_col 0 + [ping, pong] for aie_col 1, ...
    a_index = 0
    A_tiles_ordered = []
    while True:
        if a_index > len(A_tiles):
            break
        # For each column
        for i in range(n_aie_cols):
            # Add a pair of ping/pong tiles
            if a_index + i < len(A_tiles):
                A_tiles_ordered.append(A_tiles[a_index + i])
            if a_index + i + n_aie_cols < len(A_tiles):
                A_tiles_ordered.append(A_tiles[a_index + i + n_aie_cols])
        a_index += 2 * n_aie_cols
    A_tiles_ordered = TensorTileSequence.from_tiles(A_tiles_ordered)

    if b_col_maj:
        raise NotImplementedError(
            "Have not done on-the-fly transpose yet in this setting??"
        )
    else:
        # I believe this is accurate, each tile is just sent multiple times - so we need to figure out how to find the index to do the sending.
        B_tiles = TensorTiler2D.step_tiler(
            (K, N),
            (k, n),
            tile_group_repeats=(K // k, N // n // n_aie_cols),
            tile_group_steps=(1, n_aie_cols),
            tile_group_col_major=True,
        )

    C_tiles = TensorTiler2D.step_tiler(
        (M, N),
        (m * n_aie_rows, n),
        tile_group_repeats=(2, N // n // n_aie_cols),
        tile_group_steps=(1, n_aie_cols),
    )
    # TODO: In C tiles and for ordering A tiles, and probably for B tiles, I need to take into account transfer blocks.
    return A_tiles_ordered, B_tiles, C_tiles


def my_matmul_reference(M, K, N, m, k, n, n_aie_cols, b_col_maj, n_aie_rows):
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
                c_tile = TensorTile((M, N), C_offset, sizes=C_sizes, strides=C_strides)
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
                    a_tile = TensorTile(
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
                    b_tile = TensorTile(
                        (K, N), B_col_offset, sizes=B_sizes, strides=B_strides
                    )
                    B_tiles.append(b_tile)

    A_tiles = TensorTileSequence.from_tiles(A_tiles)
    B_tiles = TensorTileSequence.from_tiles(B_tiles)
    C_tiles = TensorTileSequence.from_tiles(C_tiles)
    return A_tiles, B_tiles, C_tiles
