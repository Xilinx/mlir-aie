<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Shuffle Transpose

This design takes a single input, `in`,
which is a linerized array corresponding to a `16`&times;`16` matrix.
The design uses AIE core shuffle operations (VSHUFFLE) to transpose the 
`16`&times;`16` matrix.


## Data Movement

The data movement and call into the kernel (see below)
is described in `shuffle_transpose.py`.
The DMAs are configured (using object fifos) to copy the matrix from DRAM
into L1 as a `m`&times;`n` tile with
N-major layout, i.e., the elements in a row are laid out contigously in memory.
Since the data is laid out in DRAM as N-major, the DMAs merely do a linear copy.

A single AIE core is configured to process chunks of `m`&times;`n` of `in`
(`m` and `n` are configured to be 16).
The input and output are tiled into `M/m`&times;`N/n` tiles,
and the kernel function is called that number of times -
the example is configured to process one tile, but can be configured to transpose multiple `16`&times;`16`, one after the other.


## Kernel

The vectorized kernel is implemented in `kernel.cc`.

### Transpose Strategy

We employ a tree-based shuffle algorithm that hierarchically composes interleaving shuffles to combine elements across the rows in the same column (or the `m` direction).

The algorithm is composed of two key shuffle stages:

#### 1. 4x16 to 16x4 Shuffles

- Treat the matrix as four `4x16` row tiles. Each row tile holds 4 consecutive rows, `i`, `i+1`, `i+2`, `i+3`.
- Apply a `shuffle` intrinsic with mode `T8_4x16` (T8 specifies a 8-bit shuffle)
that transposes each tile to a `16x4` tile.
  - This reorganizes this subtile from row-major to column-major
  - This operation takes as input a single 512b register (X register) 

#### 2. Interleaving Shuffles

- Interleaving shuffles take two 512b (X) registers as input and produce a 512b output.
- Note that these shuffles interleave at 32-bit granularities since the first `4x16`->`16x4` shuffle has generated 4 consecutive column elements. The 32-bit (or T32_*) shuffle collapses the 4 consecutive column elements as one 32-bit element. This allows using a 32-bit `2x16->16x2` mode that combines the elements across the two input registers to produce 8 consecutive column elements.
- First we interleave `16x4` tiles into `8x8` tiles. Next, the `8x8` tiles are interleaved to create four `4x16` column-major tiles.
- At the end we have a full `16x16` transposed column-major matrix.

#### Extending to other tensor shapes 

- The tree-based shuffle algorithm is generalizable across tensor shapes.
- The specific shuffle modes need to change for different transpose shapes, but the approach of single register followed by hierarchical interleaving shuffles is similarly applicable.
- For instance, an INT8 32x32 transpose will need 16 single-register INT8 `2x32->32x2` shuffles, followed by interleaving shuffles.
- Full list of shuffle modes supported for AIE2p can be found at: https://github.com/Xilinx/llvm-aie/blob/aie-public/clang/lib/Headers/aie2p_enums.h 

