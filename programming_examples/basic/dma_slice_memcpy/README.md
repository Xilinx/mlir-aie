<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Slice-Addressed DMA Memcpy</ins>

One dataflow, written four ways. A strided slice of DDR is staged through a
tile, a buffer at a time, and sent straight back out to DDR. There is no
compute: the point is entirely the data movement, and how much of it you write.

The part of DDR that moves is said in numpy slice notation:

```python
np.ndarray[(16, 16, 512), np.dtype[np.int8]]    # 128 KiB
[0::2, 1::2, ...]                               # 32 KiB of it
```

Every other index on the first axis, the odd ones on the second, all 512 bytes
of the third: 64 runs of 512 B gathered out of a 128 KiB buffer. The staging
buffer holds one run.

## The four

| File | Shares | Differs in |
|---|---|---|
| [`tile_dma.py`](./tile_dma.py) | — | the baseline: `Buffer` + `Lock` + `TileDma` + `Flow`, written out |
| [`objectfifo.py`](./objectfifo.py) | dataflow, placement | the staging tile comes from one `ObjectFifo.forward()` |
| [`static_dma.py`](./static_dma.py) | the tile side | DDR is named in the design at a fixed address, not passed in |
| [`copy_buffer.py`](./copy_buffer.py) | everything | the wiring is derived by a helper rather than written |

`tile_dma.py` and `objectfifo.py` run on device and verify. `static_dma.py` and
`copy_buffer.py` address DDR themselves, so there is no host buffer to hand them
and nothing to verify; they are checked by emitting MLIR.

Shared run/verify machinery lives in [`harness.py`](./harness.py). Each design
states its own geometry, where it is used.

## Saying the slice

`RuntimeData.__getitem__` and `ExternalBuffer.__getitem__` turn a slice into the
access pattern it implies, so a transfer says which part of a buffer moves
rather than the offset, sizes and strides that encode it:

```python
into_tile.fill(a, tap=a[0::2, 1::2, ...])       # a runtime-sequence argument
part = devmem[0::2, 1::2, ...]                  # a fixed-address buffer
```

A slice of an `ExternalBuffer` is still an `ExternalBuffer` — it can be copied
from wherever the whole one can, and the two share one declaration. `Bd` takes
the pattern directly, via `tap=`.

Nothing is allocated to answer a slice: the pattern is computed from the key
arithmetically, with no array involved.

## The lock handshake

One staging buffer, two channels, two locks. `buf_free` starts at 1 (the buffer
begins empty, so the inbound channel may write it); `buf_full` starts at 0
(there is nothing to send yet).

| Channel | acquires | releases |
|---|---|---|
| S2MM (in) | `buf_free` | `buf_full` |
| MM2S (out) | `buf_full` | `buf_free` |

The two hand the buffer back and forth for all 64 runs. No core is involved — a
compute tile's DMA and locks live in its memory module and work whether or not
the core is running.

### Why the tile's chains loop and the shim's do not

A `DmaChannel`'s BD chain either loops back to its head or ends after its last
BD. Which you want depends on what paces it.

The tile's channels are paced by the lock pair, so their chains loop: each runs
for as long as its locks allow, and what bounds the transfer is the slice asked
for, not a count in the design. This is also what `ObjectFifo` lowers to.

The shim's channels in `static_dma.py` take no locks. A looping chain there
would re-send the slice forever, so they set `loop=False` and move it once. A
chain that ends is also the only kind `repeat_count` means anything for: it runs
exactly `repeat_count + 1` times.

## `objectfifo.py` — the same control, generated

```python
into_tile = ObjectFifo(chunk, depth=2, name="into_tile")
out_of_tile = into_tile.cons(channel=0).forward(tile=tile, channel=0, name="out_of_tile")
```

That is the whole staging description: buffers, locks, both DMA channels and
both routes are generated. Placement stays the caller's — the same tile, and all
four hardware DMA channels pinned — so the two designs put the same things in
the same places, and their shim descriptors come out identical.

`depth=2` double-buffers, so a run can arrive while the previous one leaves,
where `tile_dma.py`'s single buffer serializes the two.

What `ObjectFifo` does not expose: lock identity (the generated locks take
whatever ids the lowering assigns), and buffer bank or address, which
`iron.Buffer` can pin.

## `static_dma.py` — DDR named in the design

```mlir
%devmem = aie.external_buffer {address = 2147483648 : i64, sym_name = "devmem"} : memref<16x16x512xi8>
%shim_dma_0_0 = aie.shim_dma(%logical_shim_noc) {
  %0 = aie.dma_start(MM2S, 0, ^bb1, ^bb2)
^bb1:
  aie.dma_bd(%devmem : memref<16x16x512xi8> offset = 512 len = 32768
             sizes = [8, 8, 512] strides = [16384, 1024, 1])
```

One descriptor, three dimensions, no iteration state: the slice goes over
exactly as the slice describes it. With the addresses in the design the runtime
sequence has nothing to do, and is emitted empty. Nothing checks that an
allocation lives at those addresses, which is the cost of naming them here.

That the slice fits three dimensions is a property of this geometry, not a
general one. A descriptor has three dimensions plus an iteration state, and a
wrap is capped — on a static `aie.shim_dma` at 1023 elements, per dimension. A
longer strided run has to be factored into two dimensions, which displaces the
outermost into the iteration state, which in turn has to be matched by a repeat
count. The runtime-sequence path has `aie-decompose-large-dma-bd` to do that; a
static program does not, because the pass declines anything inside an
`aie.shim_dma` / `aie.mem` / `aie.memtile_dma` region. Contiguous transfers are
exempt from the cap.

## `copy_buffer.py` — the wiring derived

```python
copy_buffer(
    rt,
    src_buffer=devmem[0::2, 1::2, ...],
    src_channel=0,
    dst_buffer=staging,
    dst_channel=0,
    dst_wait_for_lock=buf_free,
    dst_release_lock=buf_full,
    through_shim=shim,
)
```

Which end is the tile buffer decides the direction; the route, a channel at each
end and their descriptors all follow. It introduces no types of its own, and
reaches the same design `static_dma.py` writes by hand — byte-identical MLIR,
which [`emit.lit`](./emit.lit) checks with a `diff` so the claim stays true.

Deliberately **not** part of the IRON library: a sketch of what such an API could
look like, kept next to the primitives it is built from.

## Usage

```bash
python3 tile_dma.py --dev npu2 --col 0     # explicit Buffer/Lock/TileDma/Flow
python3 objectfifo.py --dev npu2 --col 0   # the same, generated
python3 static_dma.py                      # print MLIR for the fixed-address variant
python3 copy_buffer.py                     # the same, derived
```

`--col` selects another legal NPU2 column for the two that run; `--emit-mlir`
prints their MLIR without touching hardware.
