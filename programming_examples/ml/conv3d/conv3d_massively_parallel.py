#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.

"""
Massively Parallel Conv3D Design - Scalable to 32 cores

This design demonstrates a highly scalable Conv3D implementation using:
- Auto-detection of device capabilities (1-8 columns on NPU2)
- "Stampable block" pattern: repeatable 2-core or 4-core blocks
- Multiple shim tiles for parallel DMA bandwidth (up to 8 input + 8 output channels)
- Spatial parallelism: split height dimension across all cores
- TensorAccessPattern for clean data distribution
- Efficient weight broadcasting

Architecture:
    For N cores arranged in n_cols columns x n_rows_per_col rows:
    - Each column uses its own shim tile (col 0, 1, 2, ..., n_cols-1)
    - Within each column, cores are stacked vertically
    - Each core processes height/N rows of the input volume
    - Weights are broadcast to all cores
    - Simple core function with no conditionals
"""

import numpy as np
import sys
import argparse

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.placers import SequentialPlacer
from aie.iron.device import (
    NPU2Col1,
    NPU2Col2,
    NPU2Col3,
    NPU2Col4,
    NPU2Col5,
    NPU2Col6,
    NPU2Col7,
    NPU2,
    Tile,
)
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern


def get_device_for_cores(n_cores):
    """
    Auto-detect and return the appropriate NPU2 device class for the requested
    number of cores.

    This function implements the "stampable block" concept: we arrange cores
    in a grid that fits the device topology.

    Args:
        n_cores: Number of cores (must be 1, 2, 4, 8, 16, or 32)

    Returns:
        tuple: (device, n_cols, n_rows_per_col)
            device: NPU2ColX device instance
            n_cols: Number of columns being used
            n_rows_per_col: Number of rows per column
    """
    if n_cores == 1:
        return NPU2Col1(), 1, 1
    elif n_cores == 2:
        return NPU2Col2(), 2, 1
    elif n_cores == 4:
        return NPU2Col4(), 4, 1
    elif n_cores == 8:
        # Use all 8 columns with 1 row each
        return NPU2(), 8, 1
    elif n_cores == 16:
        # Use all 8 columns with 2 rows each
        return NPU2(), 8, 2
    elif n_cores == 32:
        # Use all 8 columns with 4 rows each (full device)
        return NPU2(), 8, 4
    else:
        raise ValueError(
            f"Unsupported number of cores: {n_cores}. "
            f"Must be one of: 1, 2, 4, 8, 16, 32"
        )


def compute_tile_width(height_per_core, width, in_channels, out_channels):
    """
    Auto-calculate the largest power-of-2 tile_width that fits in L1 memory.

    L1 budget: ~48KB usable (out of 64KB total).
    Memory per tile: 3 * h * tw * ic (input triple-buffer) +
                     2 * h * tw * oc (output double-buffer) +
                     ic * oc * 27 (weights)

    Returns tile_width such that width % tile_width == 0.
    """
    weights_mem = in_channels * out_channels * 3 * 3 * 3
    l1_budget = 48 * 1024 - weights_mem

    # Try powers of 2 from largest to smallest
    for tw in [width, 512, 256, 128, 64, 32, 16, 8]:
        if tw > width:
            continue
        if width % tw != 0:
            continue
        tile_mem = (3 * in_channels + 2 * out_channels) * height_per_core * tw
        if tile_mem <= l1_budget:
            return tw

    raise ValueError(
        f"Cannot fit any tile width in L1: h_per_core={height_per_core}, "
        f"ic={in_channels}, oc={out_channels}"
    )


def conv3dk3_massively_parallel(
    depth: int,
    width: int,
    height: int,
    in_channels: int,
    out_channels: int,
    n_cores: int = 8,
    tile_width: int = 0,
):
    """
    Massively parallel Conv3D with spatial parallelism across height dimension
    and optional width tiling for large frames.

    Design pattern - "Stampable blocks":
        - For n_cores = 8: 8 columns x 1 row = 8 blocks (1 core each)
        - For n_cores = 16: 8 columns x 2 rows = 8 blocks (2 cores each)
        - For n_cores = 32: 8 columns x 4 rows = 8 blocks (4 cores each)

    Width tiling:
        - When tile_width < width, each core processes width in tiles
        - Core loops: for w_tile in n_width_tiles: for d in depth: process_tile
        - DMA uses 4D strided access to extract tiles from host memory
        - When tile_width >= width (or auto-calculated as such), no tiling occurs

    Args:
        depth: Depth of 3D volume
        width: Width of 3D volume
        height: Height of 3D volume (must be divisible by n_cores)
        in_channels: Number of input channels
        out_channels: Number of output channels
        n_cores: Total number of cores to use (1, 2, 4, 8, 16, or 32)
        tile_width: Width tile size (0 = auto-calculate)
    """

    # Get device configuration
    dev, n_cols, n_rows_per_col = get_device_for_cores(n_cores)

    # Validate input dimensions
    assert (
        height % n_cores == 0
    ), f"Height ({height}) must be divisible by n_cores ({n_cores})"
    assert width % 8 == 0, f"Width ({width}) must be divisible by 8"
    assert (
        in_channels % 8 == 0
    ), f"Input channels ({in_channels}) must be divisible by 8"
    assert (
        out_channels % 8 == 0
    ), f"Output channels ({out_channels}) must be divisible by 8"

    # Calculate per-core sizes (spatial split by height)
    height_per_core = height // n_cores

    # Auto-calculate tile_width if not specified
    if tile_width == 0:
        tile_width = compute_tile_width(
            height_per_core, width, in_channels, out_channels
        )

    assert (
        width % tile_width == 0
    ), f"Width ({width}) must be divisible by tile_width ({tile_width})"
    n_width_tiles = width // tile_width

    print(
        f"# Configuration: {n_cores} cores = {n_cols} columns x {n_rows_per_col} rows, "
        f"tile_width={tile_width}, n_width_tiles={n_width_tiles}",
        file=sys.stderr,
    )

    # Per-core per-tile sizes (what each core processes per acquire)
    actIn_per_tile = height_per_core * tile_width * in_channels
    actOut_per_tile = height_per_core * tile_width * out_channels

    # All cores share same weights (3x3x3 kernel)
    weights_size = in_channels * out_channels * 3 * 3 * 3

    # Total tensor sizes
    tensorInSize = depth * height * width * in_channels
    tensorOutSize = depth * height * width * out_channels

    # Type definitions — tile-sized for FIFOs
    actIn_ty = np.ndarray[(actIn_per_tile,), np.dtype[np.uint8]]
    weights_ty = np.ndarray[(weights_size,), np.dtype[np.int8]]
    actOut_ty = np.ndarray[(actOut_per_tile,), np.dtype[np.uint8]]

    tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
    tensorWts_ty = weights_ty
    tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

    print(
        f"# L1 tile: {actIn_per_tile}B in, {actOut_per_tile}B out, "
        f"{weights_size}B wts",
        file=sys.stderr,
    )

    # Kernel definition
    conv3dk3_kernel = Kernel(
        "conv3dk3_ui8",
        "conv3dk3_ui8.o",
        [
            actIn_ty,
            actIn_ty,
            actIn_ty,  # 3 planes (for 3D convolution depth)
            weights_ty,
            actOut_ty,
            np.int32,
            np.int32,
            np.int32,
            np.int32,  # w, h, ci, co
            np.int32,
            np.int32,
            np.int32,  # kw, kh, kd
            np.int32,
            np.int32,
            np.int32,  # check, scale, channel_offset
        ],
    )

    # Create ObjectFIFOs for each core
    # Organization: cores[col][row_in_col] for easy column-based shim assignment
    of_in_fifos = [[None] * n_rows_per_col for _ in range(n_cols)]
    of_wts_fifos = [[None] * n_rows_per_col for _ in range(n_cols)]
    of_out_fifos = [[None] * n_rows_per_col for _ in range(n_cols)]

    for col in range(n_cols):
        for row in range(n_rows_per_col):
            # Input activations (tile-sized)
            of_in_fifos[col][row] = ObjectFifo(
                actIn_ty,
                name=f"inOF_act_c{col}_r{row}",
                depth=3,  # Triple buffer
            )

            # Weights (broadcast to all cores)
            of_wts_fifos[col][row] = ObjectFifo(
                weights_ty, depth=1, name=f"inOF_wts_c{col}_r{row}"
            )

            # Output activations (tile-sized)
            of_out_fifos[col][row] = ObjectFifo(
                actOut_ty, name=f"outOF_c{col}_r{row}"
            )

    # Core function — loops over depth planes then width tiles
    # Depth-outer ordering keeps DMA outermost dimension (depth) small,
    # avoiding the 64-entry hardware limit on the wrap dimension.
    def core_fn(of_wts, of_in, of_out, kernel):
        elemWts = of_wts.acquire(1)

        for _d in range_(depth):
            for _w_tile in range_(n_width_tiles):
                plane = of_in.acquire(1)
                elemOut = of_out.acquire(1)

                kernel(
                    plane,
                    plane,
                    plane,
                    elemWts,
                    elemOut,
                    tile_width,
                    height_per_core,
                    in_channels,
                    out_channels,
                    3,
                    3,
                    1,  # 3x3x1 kernel (2D per plane)
                    1,
                    10,
                    0,  # check=middle, scale=10, no channel_offset
                )

                of_in.release(1)
                of_out.release(1)

        of_wts.release(1)

    # Create workers with explicit placement
    workers = []
    for col in range(n_cols):
        for row in range(n_rows_per_col):
            tile_row = 2 + row

            worker = Worker(
                core_fn,
                [
                    of_wts_fifos[col][row].cons(),
                    of_in_fifos[col][row].cons(),
                    of_out_fifos[col][row].prod(),
                    conv3dk3_kernel,
                ],
                while_true=False,
                placement=Tile(col, tile_row),
            )
            workers.append(worker)

    # Runtime sequence
    rt = Runtime()

    # Data layout in host memory: D{CI/8}HW{8} (depth, ci_groups, height, width, 8)
    # For ci=8 (ci8=1), this is effectively: D × H × W × 8
    # Kernel expects (y*W+x)*8+ic indexing = H×W×{8}
    #
    # Row size in bytes: width * in_channels (= width * 8 for ci=8)
    # Plane size (one depth plane): height * width * in_channels
    row_bytes_in = width * in_channels
    plane_bytes_in = height * row_bytes_in
    row_bytes_out = width * out_channels
    plane_bytes_out = height * row_bytes_out

    # Create TensorAccessPatterns for spatial slicing with width tiling
    # Core consumes in order: d0_tile0, d0_tile1, ..., d1_tile0, ...
    # (depth-outer to keep outermost DMA dimension ≤ 63)
    # TAP extracts tiles from the full tensor with 4D strided access:
    #   sizes:   [depth, n_width_tiles, height_per_core, tile_width * in_channels]
    #   strides: [plane_bytes_in, tile_width * in_channels, row_bytes_in, 1]
    in_taps = []
    for col in range(n_cols):
        for row in range(n_rows_per_col):
            core_id = col * n_rows_per_col + row

            # Offset: skip to this core's first row
            offset = core_id * height_per_core * row_bytes_in

            in_taps.append(
                TensorAccessPattern(
                    (1, tensorInSize),
                    offset,
                    [depth, n_width_tiles, height_per_core, tile_width * in_channels],
                    [
                        plane_bytes_in,
                        tile_width * in_channels,
                        row_bytes_in,
                        1,
                    ],
                )
            )

    # Output TAPs: same pattern as input but with out_channels
    out_taps = []
    for col in range(n_cols):
        for row in range(n_rows_per_col):
            core_id = col * n_rows_per_col + row
            offset = core_id * height_per_core * row_bytes_out

            out_taps.append(
                TensorAccessPattern(
                    (1, tensorOutSize),
                    offset,
                    [
                        depth,
                        n_width_tiles,
                        height_per_core,
                        tile_width * out_channels,
                    ],
                    [
                        plane_bytes_out,
                        tile_width * out_channels,
                        row_bytes_out,
                        1,
                    ],
                )
            )

    with rt.sequence(tensorIn_ty, tensorWts_ty, tensorOut_ty) as (I, W, O):
        # Start all workers
        for worker in workers:
            rt.start(worker)

        # Fill inputs: use parallel shim DMAs (one per column)
        for col in range(n_cols):
            for row in range(n_rows_per_col):
                core_id = col * n_rows_per_col + row
                rt.fill(
                    of_in_fifos[col][row].prod(),
                    I,
                    in_taps[core_id],
                    placement=Tile(col, 0),
                )

        # Broadcast weights to all cores
        for col in range(n_cols):
            for row in range(n_rows_per_col):
                rt.fill(of_wts_fifos[col][row].prod(), W, placement=Tile(col, 0))

        # Drain outputs: use parallel shim DMAs (one per column)
        for col in range(n_cols):
            for row in range(n_rows_per_col):
                core_id = col * n_rows_per_col + row
                wait = col == n_cols - 1 and row == n_rows_per_col - 1
                rt.drain(
                    of_out_fifos[col][row].cons(),
                    O,
                    out_taps[core_id],
                    wait=wait,
                    placement=Tile(col, 0),
                )

    return Program(dev, rt).resolve_program(SequentialPlacer())


def main():
    parser = argparse.ArgumentParser(
        description="Massively Parallel Conv3D - Scalable to 32 cores"
    )
    parser.add_argument(
        "--n_cores",
        type=int,
        choices=[1, 2, 4, 8, 16, 32],
        default=8,
        help="Number of cores to use (default: 8)",
    )
    parser.add_argument(
        "--depth", "-d", type=int, default=8, help="Depth of 3D volume (default: 8)"
    )
    parser.add_argument(
        "--width",
        "-w",
        type=int,
        default=64,
        help="Width of 3D volume, must be divisible by 8 (default: 64)",
    )
    parser.add_argument(
        "--height",
        "-ht",
        type=int,
        default=64,
        help="Height of 3D volume, must be divisible by n_cores (default: 64)",
    )
    parser.add_argument(
        "--in_channels",
        "-ic",
        type=int,
        default=8,
        help="Number of input channels, must be divisible by 8 (default: 8)",
    )
    parser.add_argument(
        "--out_channels",
        "-oc",
        type=int,
        default=8,
        help="Number of output channels, must be divisible by 8 (default: 8)",
    )
    parser.add_argument(
        "--tile_width",
        "-tw",
        type=int,
        default=0,
        help="Width tile size, 0 = auto-calculate (default: 0)",
    )

    args = parser.parse_args()

    # Generate the design
    module = conv3dk3_massively_parallel(
        args.depth,
        args.width,
        args.height,
        args.in_channels,
        args.out_channels,
        args.n_cores,
        args.tile_width,
    )

    print(module)


if __name__ == "__main__":
    main()
