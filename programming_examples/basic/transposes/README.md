<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# Transposes — four ways

A single [`@iron.jit`](./transposes.py) design exposing four distinct on-device transpose mechanisms.  All four produce the same end result: a full `M × K → K × M` transpose.  Only the on-device mechanism differs — pick one via `--strategy`.

| `--strategy`  | mechanism                                                     | dtypes      | size constraints                               |
| ------------- | ------------------------------------------------------------- | ----------- | ---------------------------------------------- |
| `dma`         | pure shim-DMA stride; no compute core                         | int32/uint32 only<sup>1</sup>      | any                                            |
| `dma_packet`  | same as `dma` but lowered with `--packet-sw-objFifos`         | int32/uint32 only<sup>1</sup>      | any                                            |
| `shuffle`     | per-tile VSHUFFLE (hand-coded `transpose_16x16` kernel)       | uint8 only<sup>2</sup>             | M = K = 16 only                                |
| `combined`    | hybrid: shim DMA outer reshuffle + VSHUFFLE inner sub-tile    | i8 / i16 / i32 (`--dtype-bytes`)   | `m \| M`, `n \| K`, `s \| m`, `s \| n`<sup>3</sup> |

<sup>1</sup> Shim DMA stride-1 must be ≥ 4 bytes, so 1- and 2-byte elements would lower to `aie.dma_bd` with stride < 4 bytes and be rejected.
<sup>2</sup> The `shuffle_16x16.cc` kernel is hand-written for `uint8 16x16`; supporting other dtypes / sizes would require new kernels.
<sup>3</sup> Plus an empirical lower bound — `s = 8` needs `m, n ≥ 32` for the underlying `transpose_8x8` VECTOR_SIZE arithmetic to do the right block interleave.

Each `@iron.jit` function in [`transposes.py`](./transposes.py) raises `ValueError` if asked for a combo outside its support envelope.

## Usage

Standalone (JIT + verify on NPU, no C++):

```shell
python3 transposes.py -d npu2 -s dma          # int32 64x64
python3 transposes.py -d npu2 -s dma_packet   # int32 64x32
python3 transposes.py -d npu2 -s shuffle      # uint8 16x16
python3 transposes.py -d npu2 -s combined     # int32 128x128, m=n=32, s=8
```

Pick `--dtype-bytes 1|2|4`, `-M`, `-K`, plus `-m`/`-n`/`--ss` for `combined`.  Strategies that can't satisfy the request fail fast with a clear `ValueError`.

Via Makefile + C++ testbench (default `STRATEGY=dma`):

```shell
make
make run
```

Switch strategy / sizes:

```shell
make STRATEGY=shuffle run                       # uint8 16x16
make STRATEGY=combined M=128 K=128 run          # int32 128x128 with built-in m=n=32, s=8
```

## What replaces this directory

This directory consolidates four earlier examples:

- `basic/dma_transpose/`           → `--strategy=dma`
- `basic/dma_transpose_packet/`    → `--strategy=dma_packet`
- `basic/shuffle_transpose/`       → `--strategy=shuffle`
- `basic/combined_transpose/`      → `--strategy=combined`

All four are removed in favour of this single dispatcher.
