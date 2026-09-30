<!-- Copyright (C) 2024 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Tiling Exploration

## Overview

This directory contains resources for the Tensor Access Pattern library (`taplib`) on AIEs with IRON. `taplib` describes DMA data movements as `Layout`s: strided views over a tensor that are tiled with `Layout.full(dims).tile(tile_dims)`, refined with `.group()`, `.order()`, `.permute_tile()` and `.repeat()`, and handed to `fill()`/`drain()` or to an `ObjectFifo`'s `to_stream`/`from_stream` directly (`.tap()` and `.stream_dims()` give the underlying `TensorAccessPattern` and `[(size, stride), ...]` forms).
* [introduction](introduction): an IPython notebook that introduces `taplib` and its layout algebra
* [per_tile](per_tile): an example design illustrating the order elements are accessed when tiling (`Layout.tile`)
* [tile_group](tile_group): an example design illustrating how to access elements when grouping tiles with a single DMA operation (`TileGrid.group`)