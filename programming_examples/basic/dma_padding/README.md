<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# DMA Constant Padding

These reference designs require a target whose MemTile MM2S DMA provides the
`CONSTANT_PAD_VALUE` register.

Data is brought from external memory to a MemTile and back without a compute
tile, and the MemTile DMA **pads** the transfer on the way out: `pad_dimensions`
gives the `(before, after)` count per dimension and `pad_value` is the constant
that fills the added region. A `REAL`-element input is widened to
`PAD_BEFORE + REAL + PAD_AFTER`, with the new elements set to `pad_value`.

The constant pad value is written to the MemTile MM2S channel's
`CONSTANT_PAD_VALUE` register; a target without that register cannot set a
non-zero pad value. Because each transfer is a pure DMA passthrough, the
read-back directly exposes the pad fill, which the harness verifies.

## Layout

One file per interface; each entrypoint that takes `pad_value` is an `--api`
mode, and the pad kind is a `--pad` case. Shared run/verify/CLI infra lives in
the harness so each file isolates only the API-specific design code.

| File | Interface | `--api` entrypoints |
|------|-----------|---------------------|
| [objectfifo.py](./objectfifo.py) | ObjectFifo | `forward`, `split`, `link` |
| [tile_dma.py](./tile_dma.py)     | TileDma (explicit) | `dma_channel` |
| [harness.py](./harness.py)       | shared: pad cases, run/verify sweep, CLI | -- |

`--pad` cases (element type x value):

- `zero`  -- `int32`, `0`: the hardware default (no register write).
- `int32` -- `int32`, `1000`: a full 32-bit value held in the register.
- `int8`  -- `int8`, `8`: a sub-word value replicated across the word (`0x08080808`).

## Running

Each file sweeps all of its `--api` modes across all `--pad` cases and verifies:
```shell
python3 objectfifo.py      # forward + split + link, all pad cases
python3 tile_dma.py        # dma_channel, all pad cases
```

Narrow the sweep with flags:
```shell
python3 objectfifo.py --api split --pad int8
```
