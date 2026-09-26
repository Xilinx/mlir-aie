#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Shared helpers for the mobilenet bottleneck modules."""

import os

import numpy as np
from aie.iron import Buffer


def i8(shape):
    """numpy ndarray type alias: int8 with the given shape."""
    return np.ndarray[shape, np.dtype[np.int8]]


def u8(shape):
    """numpy ndarray type alias: uint8 with the given shape."""
    return np.ndarray[shape, np.dtype[np.uint8]]


def load_wts(data_dir, filename, expected_size):
    """Load int8 weights from `data_dir/filename`.

    Raises FileNotFoundError if the file is missing or ValueError if its size
    doesn't match `expected_size`, rather than silently falling back to
    zero-filled weights, which would produce incorrect results.
    """
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"weight file not found: {path}")
    arr = np.fromfile(path, sep=",", dtype=np.int8)
    if arr.size != expected_size:
        raise ValueError(
            f"{path}: expected {expected_size} int8 elements, got {arr.size}"
        )
    return arr


def wts_buffer(data_dir, filename, sz):
    """Static Buffer holding `sz` bytes of int8 weights from `filename`."""
    return Buffer(i8((sz,)), initial_value=load_wts(data_dir, filename, sz))


def packed_wts_buffer(data_dir, filename, sizes, align=64):
    """Static Buffer holding the weight segments of `filename`, each starting at
    a multiple of `align` bytes so the kernels' vector loads can read them.

    Returns the buffer and the byte offset of each segment.
    """
    offsets, end = [], 0
    for sz in sizes:
        start = -(-end // align) * align
        offsets.append(start)
        end = start + sz
    data = load_wts(data_dir, filename, sum(sizes))
    packed = np.zeros(end, np.int8)
    for off, seg in zip(offsets, np.split(data, np.cumsum(sizes)[:-1])):
        packed[off : off + seg.size] = seg
    return Buffer(i8((end,)), initial_value=packed), offsets


def sf_key(blk_name):
    """JSON key for a block — 'bn3' -> 'BN3', 'init' -> 'INIT', 'post_l1' -> 'POST'."""
    if blk_name.startswith("bn"):
        return blk_name.upper()
    if blk_name.startswith("post"):
        return "POST"
    return blk_name.upper()


def layer_sf(blk, sf, idx):
    """Scale factor for blk.layers[idx], looked up via its sf_key."""
    return sf[sf_key(blk.name)][blk.layers[idx].sf_key]


def skip_sf(blk, sf):
    """Scale factor for the skip-add (only valid when blk.skip is True)."""
    return sf[sf_key(blk.name)][blk.skip_sf_key]


def sa_placer_flags(seed=3):
    """aiecc flags that place the design with the SA placer. The default
    sequential placer can't seat the cascade pairs next to each other."""
    return ["--placer=sa_placer", f"--sa-seed={seed}"]
