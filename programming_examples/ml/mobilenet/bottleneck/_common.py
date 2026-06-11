#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2026, Advanced Micro Devices, Inc.
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
    doesn't match `expected_size` — silent fallback to zero-filled weights
    used to compile a numerically-broken design that ran fine but produced
    wrong output.
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


def tile_kw(tile_or_dict, dict_key=None, *, kw="tile"):
    """Build a keyword-arg dict for optional tile placement.

    Returns ``{kw: tile}`` when a tile is provided, or ``{}`` when ``None``.
    Intended for ``**tile_kw(...)`` splat into Worker / ObjectFifo calls so
    that placement is only passed when explicitly requested.

    Args:
        tile_or_dict: A Tile, a dict of tiles, or None.
        dict_key: If *tile_or_dict* is a dict, extract this key.
        kw: Keyword name to use (default ``"tile"``).
    """
    if tile_or_dict is None:
        return {}
    tile = tile_or_dict[dict_key] if dict_key is not None else tile_or_dict
    return {kw: tile}
