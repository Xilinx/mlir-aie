# Flexible Data Movement Patterns for NPU/AIE

This directory contains 7 prototype designs that explore different strategies
for maximizing shimDMA bandwidth utilization and achieving flexible data
distribution patterns from DDR through MemTile to AIE compute cores.

## Hardware Context

The fundamental bottleneck is shimDMA: each **Shim Tile has only 2 MM2S + 2 S2MM
DMA channels** connecting DDR to the AIE array. Meanwhile, **MemTiles have 6+6
DMA channels, 48 BDs, and 512KB memory** — significantly richer resources.

The key architectural insight is to use **MemTile as a programmable distribution
hub**: funnel data through limited shimDMA channels into MemTile, then use
MemTile's 6 output channels to fan out to multiple cores.

```
DDR ──[2 MM2S]──► Shim Tile ──► MemTile (512KB, 6+6 DMA) ──► Core[0,2]
                                         ├──► Core[0,3]
                                         ├──► Core[0,4]
                                         └──► Core[0,5]
```

## Prototypes

### 01: ObjectFIFO Broadcast + Split Baseline

Uses high-level `object_fifo_link()` with split offsets to distribute data
to 4 cores, and join offsets to gather results.

- **Pattern**: `object_fifo_link(of_in, [of_split_0..3], [], [0, 256, 512, 768])`
- **ShimDMA**: 1 MM2S + 1 S2MM
- **Cores**: 4
- **API level**: ObjectFIFO (high-level)

### 02: Packet-Switched Channel Multiplexing

Uses `packetflow()` with packet IDs to multiplex 2 logical data streams
over 1 physical shimDMA channel. MemTile routes packets by ID to different cores.

- **Pattern**: `packetflow(pkt_id=N, source=ShimTile, ..., keep_pkt_header=True)`
- **ShimDMA**: 1 MM2S + 1 S2MM (multiplexed via packet IDs)
- **Cores**: 2 (limited by packet arbiter mask constraints)
- **API level**: Low-level (`buffer()`, `lock()`, `packetflow()`, `@memtile_dma`)
- **Note**: ObjectFIFO does not yet support `packet_flow`. This is an area where
  a new abstraction (e.g., `PacketObjectFifo`) would reduce boilerplate
  significantly. See the TODO in `programming_examples/basic/packet_switch/`.

### 03: Time-Multiplexed Multi-Phase Pipeline

Reuses shimDMA channels across sequential iterations using `dma_wait()` as
barriers. Processes 4 batches through the same hardware without reconfiguring.

- **Pattern**: Loop in `runtime_sequence` with `npu_dma_memcpy_nd()` + `dma_wait()`
- **ShimDMA**: 1 MM2S + 1 S2MM (reused across 4 iterations)
- **Cores**: 4
- **API level**: ObjectFIFO + runtime sequence

### 04: BD-Chained Streaming Without Host Intervention

Uses `iter_count` on ObjectFIFOs to control BD chain iteration count, and
`set_repeat_count()` to cause automatic data repetition. A single host DMA
command triggers 8 autonomous iterations.

- **Pattern**: `object_fifo(..., iter_count=8)` + `sf.set_repeat_count(2)`
- **ShimDMA**: 1 BD command → 8 autonomous iterations
- **Cores**: 2
- **API level**: ObjectFIFO with iter_count/repeat_count

### 05: Multi-Column Bandwidth Scaling

Uses 2 columns to double aggregate shimDMA bandwidth. Each column has
independent Shim/MemTile/Core tiles running in parallel.

- **Pattern**: `tile(col, row)` across 2 columns with independent ObjectFIFOs
- **ShimDMA**: 4 MM2S + 4 S2MM (2 per column, parallel)
- **Cores**: 4 (2 per column)
- **API level**: ObjectFIFO (high-level)

### 06: Runtime-Programmable MemTile Hub

Uses `dma_configure_task()` on MemTile — **no `@memtile_dma()` decorator** — to
program all MemTile DMA buffer descriptors at runtime. Buffers and flows are
pre-wired at compile time, but which data goes where is decided by the
runtime_sequence.

- **Pattern**: `dma_configure_task(MemTile, S2MM/MM2S, ch)` with locks for sync
- **ShimDMA**: 1 MM2S + 1 S2MM
- **Cores**: 2 (Core(0,2) and Core(0,3))
- **MemTile DMAs**: 6 tasks configured at runtime (S2MM:0, MM2S:0, MM2S:2,
  S2MM:2, S2MM:4, MM2S:4)
- **API level**: Low-level (`buffer()`, `lock()`, `flow()`, `dma_configure_task()`)
- **Key insight**: MemTile is a runtime-programmable data hub. Pre-wire flows
  to all possible cores, then at runtime choose which channels to activate
  and which buffer data to move through them.
- **Known issue**: MemTile odd DMA channels (1, 3, 5) have a BD ID allocation
  bug in the compiler (BDs 24-47 required but 0-23 assigned). Workaround:
  use only even channels (0, 2, 4). See `memtile_hub_abstraction.py` for
  details and proposed compiler fix.
- **Abstraction sketch**: `memtile_hub_abstraction.py` proposes a `MemTileHub`
  class that reduces ~200 lines of boilerplate to ~15 lines.

### 07: MemTile Pool Allocator

Uses a **single pool buffer** in MemTile with `dma_bd(pool, offset=N, len=M)` to
address different sub-regions of the same `aie.buffer()`. Replaces Prototype 6's
4 separate buffer objects with 1 pool buffer and offset-based BD addressing.

- **Pattern**: `dma_bd(pool, offset=N, len=M)` with different offsets per BD
- **ShimDMA**: 1 MM2S + 1 S2MM
- **Cores**: 2 (Core(0,2) and Core(0,3))
- **MemTile buffers**: 1 pool (`memref<2048xui8>`) instead of 4 separate buffers
- **API level**: Low-level (`buffer()`, `lock()`, `flow()`, `dma_configure_task()`)
- **Key insight**: One contiguous allocation can serve multiple logical buffers
  via offset-based addressing in the BD configuration — the foundation for a
  runtime pool allocator where the host decides memory layout on the fly.
- **Abstraction**: `memtile_hub_abstraction.py` extended with `pool_alloc()`,
  `pool_free()`, `pool_reset()` API and `PoolRegion` dataclass.

## Results Summary

| Prototype | ShimDMA Used | Cores | Key Pattern |
|-----------|-------------|-------|-------------|
| 01 Broadcast/Split | 1 MM2S + 1 S2MM | 4 | `object_fifo_link` split/join |
| 02 Packet Mux | 1 MM2S + 1 S2MM | 2 | `packetflow` multiplexing |
| 03 Time Mux | 1 MM2S + 1 S2MM (reused) | 4 | `dma_wait` phase barriers |
| 04 BD Chain | 1 MM2S + 1 S2MM | 2 | `iter_count` + `repeat_count` |
| 05 Multi-Column | 4 MM2S + 4 S2MM | 4 | Parallel columns |
| 06 MemTile Hub | 1 MM2S + 1 S2MM | 2 | Runtime `dma_configure_task` on MemTile |
| 07 Pool Alloc | 1 MM2S + 1 S2MM | 2 | 1 pool buffer, offset-based BDs |

All prototypes verified on NPU Strix Halo hardware.

## Proposed Abstraction: MemTileHub

See `06_runtime_memtile_hub/memtile_hub_abstraction.py` for the full design.

The core idea is a `MemTileHub` class that manages MemTile L2 buffer allocation,
lock pairs, flow pre-wiring, and runtime DMA task generation:

```python
hub = MemTileHub(tile(0, 1), shim=tile(0, 0))

# Compile time: allocate L2 buffers and pre-wire routes
buf_a = hub.alloc("data_a", shape=(256,), dtype=np.uint8)
buf_b = hub.alloc("data_b", shape=(256,), dtype=np.uint8)
hub.connect(tile(0, 2))   # Core A
hub.connect(tile(0, 3))   # Core B

# Runtime: program data movement by coordinate
with rt.sequence(in_ty, out_ty) as (inp, out):
    hub.load(buf_a, inp, offset=0, size=256)       # DDR → MemTile
    hub.send(buf_a, target=tile(0, 2))              # MemTile → Core A (unicast)
    hub.broadcast(buf_b, targets=[tile(0,2), tile(0,3)])  # broadcast
    hub.recv(res, source=tile(0, 2))                # Core A → MemTile
    hub.drain(res, out, offset=0, wait=True)        # MemTile → DDR
```

### Other Proposed Abstractions

1. **PacketObjectFifo**: ObjectFIFO variant using packet switching, allowing
   multiple logical FIFOs to share a single physical DMA channel.

2. **PhasedRuntime**: Higher-level runtime sequence API for multi-phase
   pipelines with automatic barrier insertion.

3. **ColumnGroup**: Abstraction that replicates a single-column ObjectFIFO
   pattern across N columns with independent shimDMA channels.

## Environment Setup

```bash
source env.sh  # Sets up ironenv, XRT, PEANO_INSTALL_DIR, MLIR_AIE_DIR
```

## Building and Running

```bash
cd 01_broadcast_split_baseline
source ../env.sh
make NPU2=1 all
make NPU2=1 run_py
```
