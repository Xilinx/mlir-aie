<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Vector Reduce Argmax

The argmax counterpart of [`vector_reduce_max`](../vector_reduce_max): the reduction returns *where* the maximum is, not just its value.

The input streams through a memtile buffer split across 4 cores; each core folds its share into a running record and the records fold pairwise down the column, so core 0 emits one 8-byte `(value, index)` result whatever the input length. Both `bf16` and `i32` are supported.

Ties resolve to the lowest index, matching `numpy.argmax`.

## Source Files Overview

1. `vector_reduce_argmax.py`: IRON design. Each core passes its slice's start as the kernel's `index_offset`, so every record leaves the core carrying a global index and the fold needs no per-core fixup. The first tile of a slice writes the running record outright rather than folding into it: a core-resident buffer keeps its value from the previous run of the same design, so seeding one with an identity record would make the result depend on run order.

1. `argmax.cc`: kernel, pulled from [`aie_kernels/aie2/`](../../../aie_kernels/aie2/). `argmax_vector*` streams the tile once, carrying a per-lane running maximum and the offset at which each lane last improved, and resolves the position after the loop; `argmax_combine*` merges two records. The record is two `int32` slots -- the winning value (`int32` as itself, `bfloat16` widened to `float` and bit-cast) and its index -- so one ObjectFifo carries both.

## Usage

```shell
python3 vector_reduce_argmax.py --dev npu2
python3 vector_reduce_argmax.py --dev npu2 -dt i32 -i1s 524288
```

`in1_size` is in bytes and must be a whole number of 2048-element iterations; see `--help`.
