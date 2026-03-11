#!/usr/bin/env python3
# 32-core Conv3D with WIDTH TILING
#
# Architecture: 8 columns × 4 rows, memtile split/join for data distribution.
# Input: shim → memtile (combined) → split → 4 cores
# Output: 4 cores → join → memtile (combined) → shim
# Weights: shim → broadcast → 4 cores (no memtile needed)
#
# DMA: per-depth-plane BDs to avoid the 1MB stride limit on large frames.

import numpy as np
import sys
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.iron.controlflow import range_

n_cores = 32
depth = int(sys.argv[1]) if len(sys.argv) > 1 else 8
width = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
height = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
in_channels = int(sys.argv[4]) if len(sys.argv) > 4 else 8
out_channels = int(sys.argv[5]) if len(sys.argv) > 5 else 8

n_cols = 8
n_rows_per_col = 4

assert height % n_cores == 0
height_per_core = height // n_cores

# --- Memory budget calculations ---

MEMTILE_SIZE = 512 * 1024  # 512 KB per column
L1_SIZE = 64 * 1024  # 64 KB per core
MAX_DMA_STRIDE = 0xFFFFF * 4  # Shim DMA stride limit: 20-bit words × 4 bytes/word ≈ 4MB

weights_size = in_channels * out_channels * 3 * 3 * 3


def select_tile_width(h_per_core, w, ic, oc):
    """Select largest power-of-2 tile_width that fits in both L1 and memtile."""
    for tw in [w, 512, 256, 128, 64, 32, 16, 8]:
        if tw > w or w % tw != 0:
            continue
        tile_in = h_per_core * tw * ic
        tile_out = h_per_core * tw * oc

        # L1 check: input(depth=2) + output(depth=2) + weights
        l1_usage = 2 * tile_in + 2 * tile_out + weights_size
        if l1_usage > L1_SIZE:
            continue

        # Memtile check: combined buffers (depth=2 each, shared via link)
        combined_in = tile_in * n_rows_per_col
        combined_out = tile_out * n_rows_per_col
        mt_usage = 2 * combined_in + 2 * combined_out
        if mt_usage > MEMTILE_SIZE:
            continue

        return tw

    raise ValueError(
        f"No tile_width fits: h_per_core={h_per_core}, w={w}, ic={ic}, oc={oc}"
    )


tile_width = select_tile_width(height_per_core, width, in_channels, out_channels)
assert width % tile_width == 0
n_width_tiles = width // tile_width

actIn_per_tile = height_per_core * tile_width * in_channels
actOut_per_tile = height_per_core * tile_width * out_channels

# Combined buffer sizes for memtile split/join
combined_in_size = actIn_per_tile * n_rows_per_col
combined_out_size = actOut_per_tile * n_rows_per_col

# Compute optimal FIFO depths within budgets
# L1: depth_in * actIn + depth_out * actOut + weights <= 64KB
# Memtile: depth_mt_in * combined_in + depth_mt_out * combined_out <= 512KB


def compute_fifo_depths(tile_in, tile_out, combined_in, combined_out, wts,
                        n_depth_bds):
    """Find max FIFO depths that fit in L1, memtile, and BD budget.

    Each linked FIFO side consumes ~depth BDs on the memtile DMA channel.
    With split (4-way), the memtile needs BDs for each consumer.
    Total memtile BDs per channel ≈ fifo_depth * n_consumers + n_depth_bds.
    Max 24 BDs per channel.
    """
    l1_avail = L1_SIZE - wts
    # BD budget: memtile channel has 24 BDs.
    # Split link uses: fifo_depth * n_rows_per_col BDs on memtile consumer side
    # Plus n_depth_bds for the producer side.
    max_memtile_bds = 24
    for d in [2]:
        l1_ok = d * tile_in + d * tile_out <= l1_avail
        mt_ok = d * combined_in + d * combined_out <= MEMTILE_SIZE
        bd_ok = d * n_rows_per_col + n_depth_bds <= max_memtile_bds
        if l1_ok and mt_ok and bd_ok:
            return d
    return 2  # minimum for double buffering


# DMA stride check and BD split calculation
plane_stride_in = height * width * in_channels
plane_stride_out = height * width * out_channels

# Find max planes that can share a BD without exceeding stride limit.
# Stride between groups = planes_per_bd * plane_stride, must be <= 1MB.
# When planes_per_bd=1, stride between groups doesn't apply (size[0]=1 → stride ignored).
# When planes_per_bd=depth, stride = plane_stride (single-plane stride).
if plane_stride_in <= MAX_DMA_STRIDE:
    planes_per_bd = depth  # all planes in one BD
    n_depth_bds = 1
else:
    planes_per_bd = 1  # one BD per depth plane
    n_depth_bds = depth

fifo_depth = compute_fifo_depths(
    actIn_per_tile, actOut_per_tile, combined_in_size, combined_out_size,
    weights_size, n_depth_bds,
)

# Validation
l1_usage = fifo_depth * actIn_per_tile + fifo_depth * actOut_per_tile + weights_size
mt_usage = fifo_depth * combined_in_size + fifo_depth * combined_out_size

assert l1_usage <= L1_SIZE, (
    f"L1 overflow: {l1_usage} > {L1_SIZE} "
    f"(fifo_depth={fifo_depth}, tile_in={actIn_per_tile}, tile_out={actOut_per_tile})"
)
assert mt_usage <= MEMTILE_SIZE, (
    f"Memtile overflow: {mt_usage} > {MEMTILE_SIZE} "
    f"(fifo_depth={fifo_depth}, combined_in={combined_in_size}, combined_out={combined_out_size})"
)

# Total tensor sizes
tensorInSize = depth * height * width * in_channels
tensorOutSize = depth * height * width * out_channels

print(
    f"// Config: {n_cores}cores, {depth}×{height}×{width}, "
    f"tile_w={tile_width}, n_tiles={n_width_tiles}",
    file=sys.stderr,
)
print(
    f"// L1: {l1_usage}B/{L1_SIZE}B ({100*l1_usage//L1_SIZE}%), "
    f"Memtile: {mt_usage}B/{MEMTILE_SIZE}B ({100*mt_usage//MEMTILE_SIZE}%), "
    f"FIFO depth={fifo_depth}",
    file=sys.stderr,
)

need_split_dma = planes_per_bd < depth

if need_split_dma:
    print(
        f"// DMA: {n_depth_bds} BDs × {planes_per_bd} planes each "
        f"(plane_stride={plane_stride_in} > {MAX_DMA_STRIDE})",
        file=sys.stderr,
    )

with mlir_mod_ctx() as ctx:

    @device(AIEDevice.npu2)
    def device_body():
        actIn_ty = np.ndarray[(actIn_per_tile,), np.dtype[np.uint8]]
        weights_ty = np.ndarray[(weights_size,), np.dtype[np.int8]]
        actOut_ty = np.ndarray[(actOut_per_tile,), np.dtype[np.uint8]]

        in_combined_ty = np.ndarray[(combined_in_size,), np.dtype[np.uint8]]
        out_combined_ty = np.ndarray[(combined_out_size,), np.dtype[np.uint8]]

        tensorIn_ty = np.ndarray[(tensorInSize,), np.dtype[np.uint8]]
        tensorWts_ty = np.ndarray[(weights_size,), np.dtype[np.int8]]
        tensorOut_ty = np.ndarray[(tensorOutSize,), np.dtype[np.uint8]]

        conv3dk3_i8 = external_func(
            "conv3dk3_ui8",
            inputs=[
                actIn_ty, actIn_ty, actIn_ty, weights_ty, actOut_ty,
                np.int32, np.int32, np.int32, np.int32,
                np.int32, np.int32, np.int32,
                np.int32, np.int32, np.int32,
            ],
        )

        shim_tiles = [tile(col, 0) for col in range(n_cols)]
        mem_tiles = [tile(col, 1) for col in range(n_cols)]
        core_tiles = [
            [tile(col, 2 + row) for col in range(n_cols)]
            for row in range(n_rows_per_col)
        ]

        # --- Input: shim → memtile (combined) → split → cores ---
        in_offsets = [actIn_per_tile * row for row in range(n_rows_per_col)]
        of_in_L3L2 = [None] * n_cols
        of_in_L2L1 = [[None] * n_cols for _ in range(n_rows_per_col)]

        for col in range(n_cols):
            of_in_L3L2[col] = object_fifo(
                f"in_L3L2_{col}",
                shim_tiles[col], mem_tiles[col], fifo_depth, in_combined_ty,
            )
            for row in range(n_rows_per_col):
                of_in_L2L1[row][col] = object_fifo(
                    f"in_L2L1_{row}_{col}",
                    mem_tiles[col], core_tiles[row][col], fifo_depth, actIn_ty,
                )
            object_fifo_link(
                of_in_L3L2[col],
                [of_in_L2L1[row][col] for row in range(n_rows_per_col)],
                [], in_offsets,
            )

        # --- Weights: shim → broadcast → cores (no memtile) ---
        of_wts = [None] * n_cols
        for col in range(n_cols):
            of_wts[col] = object_fifo(
                f"wts_{col}", shim_tiles[col],
                [core_tiles[row][col] for row in range(n_rows_per_col)],
                1, weights_ty,
            )

        # --- Output: cores → join → memtile (combined) → shim ---
        out_offsets = [actOut_per_tile * row for row in range(n_rows_per_col)]
        of_out_L1L2 = [[None] * n_cols for _ in range(n_rows_per_col)]
        of_out_L2L3 = [None] * n_cols

        for col in range(n_cols):
            for row in range(n_rows_per_col):
                of_out_L1L2[row][col] = object_fifo(
                    f"out_L1L2_{row}_{col}",
                    core_tiles[row][col], mem_tiles[col], fifo_depth, actOut_ty,
                )
            of_out_L2L3[col] = object_fifo(
                f"out_L2L3_{col}",
                mem_tiles[col], shim_tiles[col], fifo_depth, out_combined_ty,
            )
            object_fifo_link(
                [of_out_L1L2[row][col] for row in range(n_rows_per_col)],
                of_out_L2L3[col], out_offsets, [],
            )

        # --- Core logic ---
        for col in range(n_cols):
            for row in range(n_rows_per_col):

                @core(core_tiles[row][col], "conv3dk3_ui8.o")
                def core_body():
                    for _ in range_(0xFFFFFFFF):
                        elem_wts = of_wts[col].acquire(ObjectFifoPort.Consume, 1)
                        for d in range_(depth):
                            for w_tile in range_(n_width_tiles):
                                elem_in = of_in_L2L1[row][col].acquire(
                                    ObjectFifoPort.Consume, 1
                                )
                                elem_out = of_out_L1L2[row][col].acquire(
                                    ObjectFifoPort.Produce, 1
                                )
                                conv3dk3_i8(
                                    elem_in, elem_in, elem_in,
                                    elem_wts, elem_out,
                                    tile_width, height_per_core,
                                    in_channels, out_channels,
                                    3, 3, 1, 1, 10, 0,
                                )
                                of_in_L2L1[row][col].release(
                                    ObjectFifoPort.Consume, 1
                                )
                                of_out_L1L2[row][col].release(
                                    ObjectFifoPort.Produce, 1
                                )
                        of_wts[col].release(ObjectFifoPort.Consume, 1)

        # --- Runtime DMA sequence ---
        @runtime_sequence(tensorIn_ty, tensorWts_ty, tensorOut_ty)
        def sequence(I, W, O):
            col_height = n_rows_per_col * height_per_core
            row_bytes_in = width * in_channels
            row_bytes_out = width * out_channels

            # Input DMAs: one per column
            # Core loops: for d in depth: for tile in n_tiles:
            # DMA must stream data in this order.
            for col in range(n_cols):
                base_offset = col * col_height * row_bytes_in

                for bd_idx in range(n_depth_bds):
                    d_start = bd_idx * planes_per_bd
                    d_count = min(planes_per_bd, depth - d_start)
                    plane_offset = base_offset + d_start * plane_stride_in

                    npu_dma_memcpy_nd(
                        metadata=of_in_L3L2[col],
                        bd_id=bd_idx,
                        mem=I,
                        offsets=[0, 0, 0, plane_offset],
                        sizes=[
                            d_count, n_width_tiles,
                            col_height, tile_width * in_channels,
                        ],
                        strides=[
                            plane_stride_in if d_count > 1 else 0,
                            tile_width * in_channels,
                            row_bytes_in, 1,
                        ],
                    )

            # Weights
            for col in range(n_cols):
                npu_dma_memcpy_nd(
                    metadata=of_wts[col], bd_id=0, mem=W,
                    sizes=[1, 1, 1, weights_size],
                )

            # Output DMAs: one per column, same pattern as input
            for col in range(n_cols):
                base_offset = col * col_height * row_bytes_out

                for bd_idx in range(n_depth_bds):
                    d_start = bd_idx * planes_per_bd
                    d_count = min(planes_per_bd, depth - d_start)
                    plane_offset = base_offset + d_start * plane_stride_out

                    npu_dma_memcpy_nd(
                        metadata=of_out_L2L3[col],
                        bd_id=bd_idx,
                        mem=O,
                        offsets=[0, 0, 0, plane_offset],
                        sizes=[
                            d_count, n_width_tiles,
                            col_height, tile_width * out_channels,
                        ],
                        strides=[
                            plane_stride_out if d_count > 1 else 0,
                            tile_width * out_channels,
                            row_bytes_out, 1,
                        ],
                    )

            npu_sync(column=0, row=0, direction=0, channel=0)

    print(ctx.module)
