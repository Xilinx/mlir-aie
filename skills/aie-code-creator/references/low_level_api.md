<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Lower-Level API — explicit tiles

Use this only when you need explicit tile coordinates, custom routing, or features not
exposed in the high-level API. In practice this is rarely needed: `Worker` plus
`ObjectFifo` covers the large majority of designs, including custom DMA access patterns
(see `python_api.md`), and all current examples in this repo are written against the
high-level API. Reach for the lower-level primitives below only for a genuine gap — a
`@mem(tile)` body to script DMA channels by hand, or porting an existing external
MLIR-AIE example 1:1 — not as a default starting point.

## Skeleton

```python
import numpy as np
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

def design():
    @device(AIEDevice.npu2)              # or AIEDevice.npu1
    def device_body():
        # Declare tiles at specific (col, row)
        ShimTile    = tile(0, 0)
        MemTile     = tile(0, 1)
        ComputeTile = tile(0, 2)

        tile_ty = np.ndarray[(1024,), np.dtype[np.int32]]

        of_in  = object_fifo("in",  ShimTile,    ComputeTile, 2, tile_ty)
        of_out = object_fifo("out", ComputeTile, ShimTile,    2, tile_ty)

        @core(ComputeTile)
        def core_body():
            for _ in range_(0xFFFFFFFF):                       # "forever" loop
                e_in  = of_in.acquire(ObjectFifoPort.Consume, 1)
                e_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                for i in range_(1024):
                    e_out[i] = e_in[i] + 1
                of_in.release(ObjectFifoPort.Consume, 1)
                of_out.release(ObjectFifoPort.Produce, 1)

        # Host-side runtime sequence
        @runtime_sequence(tile_ty, tile_ty)
        def sequence(a_in, c_out):
            npu_dma_memcpy_nd(metadata=of_in,  bd_id=0, mem=a_in,
                              sizes=[1, 1, 1, 1024])
            npu_dma_memcpy_nd(metadata=of_out, bd_id=1, mem=c_out,
                              sizes=[1, 1, 1, 1024])
            dma_wait(of_out)

with mlir_mod_ctx() as ctx:
    design()
    assert ctx.module.operation.verify() == True
    print(ctx.module)
```

## Key differences from the high-level API

| | High-level | Lower-level |
|---|------------|---------------------|
| Tile reference | implicit (`AnyComputeTile`) or by `Tile(col,row)` | explicit `tile(col, row)` |
| Core definition | `Worker(fn, args)` | `@core(tile) def core_body()` |
| ObjectFifo acquire | `of.acquire(n)` | `of.acquire(ObjectFifoPort.Consume/Produce, n)` |
| Runtime | `Runtime(seq_fn, fn_args)` | `@runtime_sequence(...)` |
| DMA | `ObjectFifoHandle.fill` / `.drain` | `npu_dma_memcpy_nd` + `dma_wait` |
| Compilation | `Program(dev, rt, workers=[...]).resolve_program()` | wrapped in `mlir_mod_ctx()` |
| Verification | implicit | call `ctx.module.operation.verify()` |

## When dropping to the lower-level API is justified

This is genuinely rare — all current examples in this repo are written against the
high-level API, and custom DMA access patterns are already possible there (see
`python_api.md`) without a hand-scripted `@mem(tile)` body. Reach for the primitives below
only when:

- You're porting an *external*, pre-existing MLIR-AIE example (not from this repo) and
  want to keep its structure 1:1 rather than rewrite it against `Worker`/`ObjectFifo`.
- You've hit a genuine gap in the high-level API's DMA/flow support — in which case, also
  consider raising it upstream rather than only working around it here.
- You need a custom inter-tile flow (`flow(src, src_bundle, src_ch, dst, ...)`) that isn't
  covered by `iron.Flow`/`iron.PacketFlow` below (which already work from the high-level
  API — check there first).

Explicit tile coordinates alone are not a reason to drop to this level: the high-level API
takes a `tile=Tile(col, row)` pin on `Worker`/`Buffer` when you need to constrain
placement.

## Routing helpers (when you need them)

```python
# Explicit ObjectFifo link through a mem tile
object_fifo_link(fifo_ins=[of_a, of_b], fifo_outs=[of_out],
                 offsets_in=[0, 1024], offsets_out=[])

# Raw flow between switchbox endpoints
flow(src_tile, WireBundle.DMA, 0, dst_tile, WireBundle.DMA, 0)

# Mem-tile script
@mem(MemTile)
def mem_body():
    s0 = dma_start(DMAChannelDir.S2MM, 0, dest=bb1, chain=bb2)
    ...
```

Most of the time you want to escape to the high-level API as soon as possible — it's a much shorter route to a working design.

## `iron.Flow` / `iron.PacketFlow` — mid-level flow topology

Between the raw `flow()`/`object_fifo_link()` dialect calls above and a full `ObjectFifo`, `aie.iron` also exposes `Flow` and `PacketFlow` as standalone topology primitives, for when you're driving a `TileDma` + `Buffer`/`Lock` by hand and just need the switchbox routing declared:

```python
from aie.iron import Flow, PacketFlow, PacketDest

# Circuit-switched, one dedicated route:
f = Flow(src_tile, dst_tile)
rt.add_flow(f)   # register with the Runtime so Program.resolve_program() resolves it

# Packet-switched — pkt_id is positional (first arg), with optional fan-out to extra destinations:
pf = PacketFlow(1, src_tile, dst_tile,
                extra_dsts=[PacketDest(other_tile)], keep_pkt_header=False)
rt.add_flow(pf)
```

Constructing a `Flow`/`PacketFlow` does not register it anywhere by itself — you must call `rt.add_flow(...)` on your `Runtime` instance (both share the same registration method). Without that call, `Program.resolve_program()` never sees the flow and it silently never resolves. Both resolve to `aie.flow`/packet-switching ops when the program is placed — they only declare the topology edge; you still own the `TileDma`/`Buffer`/`Lock` wiring on each end. Reach for `ObjectFifo` first; these are for the rare case where you need routing control without the full ObjectFifo abstraction.

A `Flow` or `PacketFlow` with a shim end has `fill()`/`drain()` like an `ObjectFifoHandle` (`PacketFlow.fill` stamps the packet header on the shim descriptor; see `programming_examples/basic/packet_switch`). A route between the shim and a mem tile has a DMA at each end, and the runtime sequence can drive both: `fill`/`drain` program the shim end, and `flow.endpoint(mem).task(...)` programs the mem or core tile end, so its lengths, offsets and access patterns can change per dispatch rather than being fixed at load time as with `TileDma`. The endpoint supplies the tile, direction and channel; `DmaEndpoint(tile, direction, channel)` pins a channel by index instead. See `programming_guide/section-2/section-2d/DMATasks.md` and `test/python/npu-xrt/test_tile_dma_task_dispatch.py`:

```python
from aie.iron import Acquire, Bd, Release

into = Flow(shim, mem)   # host -> mem tile; compiler picks both channels
out = Flow(mem, shim)    # mem tile -> host

def sequence(A, C, n):
    into.fill(A)                                        # shim end of `into` (MM2S)
    into.endpoint(mem).task(                            # mem end of `into` (S2MM)
        Bd(buf, acquires=[Acquire(empty)], releases=[Release(full)])
    ).start()
    out.endpoint(mem).task(                             # mem end of `out` (MM2S)
        Bd(buf, length=n, acquires=[Acquire(full)], releases=[Release(empty)])
    ).start()
    out.drain(C, transfer_len=n, wait=True)             # shim end of `out` (S2MM)
```

`task(*bds, runs=1, wait=False)` writes the descriptors where it is called; `start()` pushes the task and may be called again, so a task built before a loop is restarted with one push per iteration. `free()` returns its BDs, so call it only after the last start and once the task is known to have finished, through its own `await_()` or a waited transfer that depends on it. A BD freed earlier can be reprogrammed for the next task on the tile while its transfer is still queued. In a straight-line sequence, leave tasks unfreed: when a tile runs out of BDs, the compiler takes them back from tasks it can prove finished. A `Bd` in a task takes no `next`, and takes both an acquire and a release or neither. The buffers and locks a task or `TileDma` uses are found from its `Bd`s; register the flows with `rt.add_flow`. `wait=True` works on mem/core tile tasks too: the compiler routes their completion token back to the shim, so `task.await_()` returns (`test/python/npu-xrt/test_tile_dma_task_token.py`).

Both `Flow` and `PacketFlow` (and `CascadeFlow`, documented in `python_api.md`) implement the `Resolvable` protocol (`aie.iron.resolvable.Resolvable`) — a structural `Protocol` requiring `resolve(loc, ip)` and `tiles()`. This is the advertised way to work with low-level primitives from an otherwise high-level design: implement `Resolvable` on your primitive and pass it to `rt.add_flow(...)` so `Program.resolve_program()` picks it up during placement/resolution, rather than dropping the whole design down to the `@device`/`@core` skeleton above. There's generally no reason to write a full design in low-level primitives — reach for `Resolvable` when you need one custom piece of topology, and keep everything else on `Worker`/`ObjectFifo`.
