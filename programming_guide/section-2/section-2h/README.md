<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Section 2h - Advanced ObjectFifo + Cross-Tile Buffer</ins>

Two opt-in patterns that show up in real designs but are not part of
the introductory `ObjectFifo` API.  Both stay out of the way of simple
designs and only kick in when explicitly requested.

## <u>Asymmetric `ObjectFifo`: `consumer_obj_type=`</u>

`ObjectFifo`'s default contract is symmetric — the producer and
consumer agree on a single element type.  `consumer_obj_type=` decouples
producer-side and consumer-side transfer granularity: the producer
sends `obj_type`-sized chunks; the consumer receives
`consumer_obj_type`-sized chunks.  The producer element count must be
an integer multiple of the consumer's (the underlying MLIR verifier
enforces this).

```python
prod_ty = np.ndarray[(40,), np.dtype[np.int32]]
cons_ty = np.ndarray[(10,), np.dtype[np.int32]]

wts = ObjectFifo(
    prod_ty,
    depth=1,
    name="wts",
    consumer_obj_type=cons_ty,    # 4:1 ratio — one prod fill, four cons acquires
)
```

Useful when a single DMA fan-out serves consumers that want to walk
the data in smaller chunks (e.g. weight broadcast feeding a row of
compute tiles that each acquire a sub-slice), without paying for two
separate fifos and a join.

Each `ObjectFifo` declared with `consumer_obj_type=` lowers to one
`aie.objectfifo` op carrying both types:

```mlir
aie.objectfifo @wts(...) : !aie.objectfifo<memref<40xi32>>
                        -> !aie.objectfifo<memref<10xi32>>
```

Canonical demos:

- [`test/python/objFifo_asymmetric.py`](../../../test/python/objFifo_asymmetric.py)
  — minimal 40-element producer / 10-element consumer; runs end-to-end.
- [`programming_examples/ml/mobilenet/bottleneck/post_l1.py`](../../../programming_examples/ml/mobilenet/bottleneck/post_l1.py)
  and [`post_l2.py`](../../../programming_examples/ml/mobilenet/bottleneck/post_l2.py)
  — production use feeding the MobileNet FC tiles a different chunk size
  than the upstream weight stream produces.

## <u>Direct AIE-stream `ObjectFifo`: `aie_stream=(end, port)`</u>

`aie_stream=(end, port)` marks the fifo as a *direct AIE-stream*
connection — the underlying `aie.objectfifo` op gets the `aie_stream` /
`aie_stream_port` attributes stamped on it, telling the lowering to
treat the producer side as wire-only.  No L1 buffer is allocated; the
consumer reads straight off the stream.

Pair it with kernels that emit per-element via `put_ms()` instead of
acquire/release:

```python
of_dout_L1L3 = ObjectFifo(
    dout_ty,
    name="of_dout_L1L3",
    depth=2,
    aie_stream=(0, 0),            # (end, port) — wire-only producer side
)
```

The kernel's C++ body owns the writes (`aie::stream::put_ms(value)`)
and the core body never acquires or releases the producer handle:

```python
def core_body(of_in, _of_out_unused, lut0, ..., kernel):
    for _ in range_(N):
        di = of_in.acquire(1)
        kernel(di, lut0, ...)     # kernel emits output via put_ms()
        of_in.release(1)
```

Lowered MLIR (extract from `magika/group2.py --emit-mlir`):

```mlir
aie.objectfifo @of_dout_L1L3(%logical_core, {%logical_shim_noc}, 2 : i32)
    {aie_stream = 0 : i32, aie_stream_port = 0 : i32}
    : !aie.objectfifo<memref<214xi32>>
```

Canonical demo:
[`programming_examples/ml/magika/group2.py`](../../../programming_examples/ml/magika/group2.py)
— a compute tile streams its output directly to the shim via `put_ms()`.

## <u>Cross-tile `Buffer` in `Worker.fn_args`</u>

AIE CoreTiles can read their north / south / east / west neighbors' L1
memory directly via shared-memory paths.  IRON honors that capability:
a `Buffer` pinned to a different tile than the `Worker` it is passed
to no longer raises; `Program.resolve` discovers the neighbor tile via
`Buffer.tiles()` and the kernel sees a memref pointing at the
neighbor's L1.

```python
compute_tile = Tile(col=1, row=3, tile_type=AIETileType.CoreTile)
west_tile    = Tile(col=0, row=3, tile_type=AIETileType.CoreTile)

lut_buf = Buffer(
    tile=west_tile,                      # pinned to neighbor, not compute_tile
    type=lut_ty,
    initial_value=lut_data,
    name="lut",
)

worker = Worker(
    body,
    fn_args=[..., lut_buf, ...],         # cross-tile Buffer reaches the kernel
    tile=compute_tile,                   # different tile — that's the point
)
```

The convenience path is unchanged: a `Buffer` with no `tile=` set
still auto-pins to the `Worker`'s tile, so simple designs don't see
this surface at all.  The `Buffer` is rejected only when it is already
owned by *another* `Worker` — sharing across compute tiles via
shared-memory access is fine; sharing across kernels is not.

Useful when one compute tile needs more L1 than fits on its own tile.
The canonical demo —
[`programming_examples/ml/magika/group2.py`](../../../programming_examples/ml/magika/group2.py) —
spreads four large lookup tables across a compute tile and its three
CoreTile neighbors (south, north, west) so the kernel can read all
four without DMA round-trips.

-----
[[Prev - Section 2g](../section-2g/)] [[Top](..)] [[Next - Section 3](../../section-3/)]
