<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Slice-Addressed DMA Memcpy</ins>

This reference design moves a **strided slice** of DDR into a tile and straight
back out again. It targets a Ryzen™ AI NPU (npu2, single column), and there is no
compute: the point is entirely the data movement.

The same dataflow is written three ways:

| File | Varies | How |
|---|---|---|
| [`tile_dma.py`](./tile_dma.py) | staging tile | `Buffer` + `Lock` + `TileDma` + `Flow`, written out by hand |
| [`objectfifo.py`](./objectfifo.py) | staging tile | one `ObjectFifo.forward()` |
| [`static_dma.py`](./static_dma.py) | where DDR's address comes from | `aie.external_buffer` at a fixed address + a static `aie.shim_dma` |

The first two are a comparison of API level; the third reuses `tile_dma.py`'s
tile side unchanged and varies only the DDR end. Shared geometry and the
run/verify path live in [`harness.py`](./harness.py), so both runnable designs
are checked against the same reference by the same code.

The part of DDR that moves is written in numpy slice notation:

```python
DEVMEM_SHAPE = (16, 16, 4096)
DEVMEM_SLICE = np.s_[0::2, 1::2, ...]
```

That is every other index on the first axis, the odd indices on the second, and
all 4096 bytes of the third — 256 KiB gathered out of a 1 MiB buffer, in 64
strides of 4 KiB.

## Saying the slice instead of deriving it

`RuntimeData.__getitem__` turns a slice into the access pattern it implies, so
the transfer says what part of the buffer moves rather than the offset, sizes and
strides that encode it. Both versions write this identically:

```python
in_flow.fill(a, tap=a[DEVMEM_SLICE])   # tile_dma.py
out_flow.drain(c, tap=c[...], wait=True)
```

Nothing is allocated to answer the question. `TensorAccessPattern.from_slice`
applies the slice to a stand-in built with `as_strided` over a zero-byte base, so
only view metadata is touched, and reads the geometry back off the result. A
one-byte stand-in makes numpy's byte offsets and strides read directly as element
counts, so no dtype is needed.

## Data path

```
DDR (sliced)   --shim DMA MM2S 0--> staging tile --> shim DMA S2MM 0 --> DDR (whole)
```

The slice is 64× larger than the staging buffer, so it lands one 4 KiB chunk at a
time. Sending each chunk straight back out is what makes the landing observable:
the host compares the returned bytes against `devmem.numpy()[DEVMEM_SLICE]`, so a
slice that walked DDR incorrectly produces a wrong answer rather than a silent
pass. Both versions verify against that same reference.

## `tile_dma.py` — the explicit version

One buffer, two channels, two locks. `buf_free` starts at 1 because the buffer
begins empty and the inbound channel may write it; `buf_full` starts at 0 because
there is nothing to send yet.

| Channel | acquires | releases |
|---|---|---|
| S2MM 0 (in) | `buf_free` | `buf_full` |
| MM2S 0 (out) | `buf_full` | `buf_free` |

The two channels hand the buffer back and forth for all 64 chunks. No core is
involved — a compute tile's DMA and locks live in its memory module and work
whether or not the core is running.

### `loop=False` is what makes `repeat_count` mean anything

This is the part worth taking away. A `DmaChannel`'s BD chain either loops or
ends, and that choice decides whether a repeat count does anything at all:

```python
DmaChannel(
    direction=DMAChannelDir.S2MM,
    channel=0,
    loop=False,             # end the chain after its last BD
    repeat_count=CHUNKS - 1,
    bds=[...],
)
```

`loop=True` (the default) chains the last BD back to the head. That chain is
endless, so it is a task that never completes — it runs for as long as its locks
allow, and there is nothing for a repeat count to count. Pairing it with
`repeat_count` hangs.

`loop=False` points the last BD at the region's `aie.end`, which ends the chain.
Only then is it a task the channel can finish and re-run, and `repeat_count`
governs how many times. The two must agree; `repeat_count` is 0-based, so
`CHUNKS - 1` gives `CHUNKS` executions.

```mlir
%0 = aie.dma_start(S2MM, 0, ^bb1, ^bb2, repeat_count = 63)
^bb1:
  aie.use_lock(%buf_free, AcquireGreaterEqual, %c1_i32)
  aie.dma_bd(%comp05_tile_buffer : memref<4096xi8> len = 4096)
  aie.use_lock(%buf_full, Release, %c1_i32_0)
  aie.next_bd ^bb4        // the aie.end block: chain ends here
```

## `objectfifo.py` — the same dataflow, one level up

`forward()` is the direct analogue of what `tile_dma.py` writes out by hand: it
stages the stream through a tile's memory with a producer/consumer lock pair, no
core involved.

```python
of_in = ObjectFifo(chunk_ty, depth=2, name="devmem_in")
of_out = of_in.cons().forward(tile=staging, name="devmem_out")
```

That is the whole staging description. The buffers, the locks, the two DMA
channels, the two flows and the chunk count are all generated — the shim BDs come
out identical to the explicit version's.

One real difference: `depth=2` double-buffers the staging tile, so a chunk can
arrive while the previous one leaves, where `tile_dma.py`'s single buffer
serializes the two. Depth is the knob `ObjectFifo` exposes for that; the explicit
version would need a second `Buffer` and a longer BD chain.

## `static_dma.py` — naming DDR in the design

`tile_dma.py` takes DDR as a runtime-sequence argument, so its address is patched
in at dispatch and the host supplies a buffer per run. `static_dma.py` takes the
opposite trade: both DDR buffers become `aie.external_buffer`s at fixed
addresses, driven by a static `aie.shim_dma` program, so the runtime sequence has
nothing left to do and comes out empty. It imports `tile_side` and `tile_state`
from `tile_dma.py` unchanged, so the whole difference between the two files is
where DDR's address comes from.

```mlir
%device_memory_devmem = aie.external_buffer {address = 2147483648 : i64, ...} : memref<16x16x4096xi8>
%shim_dma_0_0 = aie.shim_dma(%logical_shim_noc) {
  %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2, repeat_count = 7)
^bb1:
  aie.dma_bd(%device_memory_devmem : memref<16x16x4096xi8> offset = 4096 len = 32768
             sizes = [8, 8, 512] strides = [8192, 512, 1])
             {iteration = #aie.bd_iteration<size = 8, stride = 131072, current = 0>}
```

This is emit-only. With the addresses written into the design there is no host
buffer to hand it, and nothing checks that an allocation lives there.

Note what the slice costs on this path. `tile_dma.py` hands a whole tap to `fill`
and lets `aie-decompose-large-dma-bd` legalize it; a static `aie.shim_dma` gets
no such pass, so the geometry has to be hardware-legal as written:

- A wrap is capped at 1023, counted in **elements** here rather than 32-bit
  words, so the 4096-element contiguous run cannot be one dimension. Spelling it
  as `8 x 512` costs one of the three available dimensions.
- That leaves no dimension for the slice's outer axis, so it goes in the BD's
  iteration state — which advances once per BD execution, and executions come
  from `repeat_count`. Same rule as above: the two must agree, hence
  `repeat_count = 7` for 8 iterations.

## Usage

```bash
python3 tile_dma.py --dev npu2 --col 0     # explicit Buffer/Lock/TileDma/Flow
python3 objectfifo.py --dev npu2 --col 0   # same dataflow via ObjectFifo
python3 static_dma.py                      # print MLIR for the fixed-address variant
```

Use `--col` to select another legal NPU2 column. The `run_and_verify` path
compiles, runs on the NPU, and checks the output against a numpy reference in a
single call. `--emit-mlir` prints the generated MLIR without touching hardware.
