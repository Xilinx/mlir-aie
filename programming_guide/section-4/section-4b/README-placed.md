<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//-->

# <ins>Trace for Placed Designs</ins>

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * [Section 4b - Trace](../section-4b)
    * [Section 4c - Kernel Vectorization and Optimization](../section-4c)

-----

In [section-4b](../section-4b), we introduced how trace is enabled in IRON python designs. These designs are unplaced and relies on the tools to decide on a placement for the tiles. Because of this, certain assumptions and limitation are introduced to help simplify the trace configuration. These include the following:

For placed designs using explicit tile declarations, we have more direct control over trace configuration. Tracing uses a two-phase declarative API where:
1. **`configure_trace()`** - called outside `runtime_sequence` to declare which tiles and events to trace. This emits `aie.trace` ops into the MLIR.
2. **`start_trace()`** - called inside `runtime_sequence` to configure the trace buffer and activate tracing. This emits `aie.trace.host_config` and `aie.trace.start_config` ops.

The compiler then automatically handles the remaining work through trace lowering passes (described in [How Trace Lowering Works](#5-how-trace-lowering-works)).

## <u>1. Enable and configure AIE trace units</u>

```python
import aie.utils.trace as trace_utils

# Outside runtime_sequence: declare which tiles to trace
tiles_to_trace = [ComputeTile, MemTile, ShimTile]
trace_utils.configure_trace(tiles_to_trace)

# Inside runtime_sequence: activate tracing
@runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
def sequence(A, F, C):
    trace_utils.start_trace(trace_size=opts.trace_size)
```

### <u>Tracing both core and memory on same tile</u>

Core tiles have two trace units: one for core events (instructions, stalls) and one for memory events (DMA). To trace both, list the tile twice:

```python
# First occurrence: core trace, Second occurrence: memory trace
tiles_to_trace = [tile_0_2, tile_0_2, mem_tile_0_1, shim_tile_0_0]
trace_utils.configure_trace(tiles_to_trace)
```

## <u>2. Customizing Trace Events (`configure_trace` parameters)</u>

The `configure_trace()` function accepts parameters to customize which events each tile type monitors:

```python
from aie.utils.trace.events import (
    PortEvent, MemTilePortEvent,
    CoreEvent, MemEvent, MemTileEvent, ShimTileEvent
)
from aie.dialects.aie import WireBundle

trace_utils.configure_trace(
    tiles_to_trace,
    start_broadcast=15,      # Broadcast channel for trace start (default: 15)
    stop_broadcast=14,       # Broadcast channel for trace stop (default: 14)
    coretile_events=[        # Events for core tile trace (max 8)
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        CoreEvent.INSTR_VECTOR,
        CoreEvent.MEMORY_STALL,
        CoreEvent.STREAM_STALL,
        CoreEvent.LOCK_STALL,
        PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
        PortEvent(CoreEvent.PORT_RUNNING_1, WireBundle.DMA, 0, False),
    ],
    coremem_events=[         # Events for core memory trace (max 8)
        MemEvent.DMA_S2MM_0_START_TASK,
        MemEvent.DMA_MM2S_0_START_TASK,
        MemEvent.DMA_S2MM_0_FINISHED_TASK,
        MemEvent.DMA_MM2S_0_FINISHED_TASK,
        MemEvent.DMA_S2MM_0_STREAM_STARVATION,
        MemEvent.DMA_S2MM_1_STREAM_STARVATION,
        MemEvent.CONFLICT_DM_BANK_0,
        MemEvent.CONFLICT_DM_BANK_1,
    ],
    memtile_events=[         # Events for mem tile trace (max 8)
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, 0, False),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_1, WireBundle.DMA, 1, False),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_2, WireBundle.DMA, 0, True),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_3, WireBundle.DMA, 1, True),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_4, WireBundle.DMA, 2, True),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_5, WireBundle.DMA, 3, True),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_6, WireBundle.DMA, 4, True),
        MemTilePortEvent(MemTileEvent.PORT_RUNNING_7, WireBundle.DMA, 5, True),
    ],
    shimtile_events=[        # Events for shim tile trace (max 8)
        ShimTileEvent.DMA_S2MM_0_START_TASK,
        ShimTileEvent.DMA_S2MM_1_START_TASK,
        ShimTileEvent.DMA_MM2S_0_START_TASK,
        ShimTileEvent.DMA_S2MM_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_1_FINISHED_TASK,
        ShimTileEvent.DMA_MM2S_0_FINISHED_TASK,
        ShimTileEvent.DMA_S2MM_0_STREAM_STARVATION,
        ShimTileEvent.DMA_S2MM_1_STREAM_STARVATION,
    ],
)
```

### <u>PortEvent API</u>

Port events monitor activity on stream switch ports. A **physical port** on the stream switch is identified by three components: the **bundle** (which interface on the tile, e.g., DMA, North, South), the **channel** (which channel within that bundle), and the **direction** (master for input/S2MM, slave for output/MM2S).

The hardware provides 8 port monitor slots (0-7), each configured to watch a specific physical port. The slot number is determined by the suffix of the event name (e.g., `PORT_RUNNING_0` uses slot 0).

```python
from aie.dialects.aie import WireBundle

# Core tile port monitoring
PortEvent(CoreEvent.PORT_RUNNING_0, port=WireBundle.DMA, channel=0, master=True)

# Parameters:
# - code: PORT_RUNNING_N, PORT_IDLE_N, PORT_STALLED_N, or PORT_TLAST_N (N=0-7)
# - bundle: WireBundle.DMA, WireBundle.North, WireBundle.South, etc.
# - channel: Channel number within the bundle (e.g., 0 for DMA channel 0)
# - master: Direction - True for input to tile (S2MM), False for output from tile (MM2S)

# Mem tile port monitoring
MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.DMA, channel=0, master=True)

# Shim tile port monitoring
ShimTilePortEvent(ShimTileEvent.PORT_RUNNING_0, WireBundle.South, channel=2, master=True)
```

**Sharing a monitor slot across event types:** Multiple event types can share the same slot to observe different conditions on the same port. The event name suffix determines the slot number (`PORT_RUNNING_0` and `PORT_TLAST_0` both use slot 0). When events share a slot, each event type independently triggers in the trace whenever its condition is met on the monitored port - `PORT_RUNNING_0` fires while data is flowing and `PORT_TLAST_0` fires on the last beat of a transfer. They must be configured to monitor the same physical port:

```python
# From chaining_channels example - both use slot 0 on the same port
memtile_events=[
    MemTilePortEvent(MemTileEvent.PORT_RUNNING_0, WireBundle.South, 3, True),
    MemTilePortEvent(MemTileEvent.PORT_TLAST_0, WireBundle.South, 3, True),
    # ... other events
]
```

See the [chaining_channels](../../../programming_examples/basic/chaining_channels/) example for a complete design using shared port monitor slots.

If two events sharing a slot specify different port configurations, an error is raised.

## <u>3. Customizing Host-side Trace Settings (`start_trace()` parameters)</u>

```python
trace_utils.start_trace(
    trace_size=8192,              # Buffer size in bytes (default: 8192)
    ddr_id=4,                     # XRT buffer index 0-4 → group_id 3-7 (default: 4)
    trace_after_last_tensor=False  # Append trace after last tensor (default: False)
)
```

* `trace_size` - Size of the trace buffer in bytes. This must match the buffer size allocated on the host side.
* `ddr_id` - XRT buffer index (0-4) mapping to group_id (3-7). Default is 4 (group_id 7). Ignored when `trace_after_last_tensor` is True.
* `trace_after_last_tensor` - If True, the compiler automatically appends trace data after the last `runtime_sequence` tensor argument, computing the buffer index and byte offset. The "last tensor" is determined by argument order in `runtime_sequence(A, B, C)` - typically `C` is the output buffer since users conventionally list outputs last, but it is simply whichever argument comes last. This is useful when sharing an XRT buffer between output data and trace data.

> **NOTE**: Currently, all trace data is routed to a single shim tile at column 0 (tile 0,0). Per-column routing (where each column's traces route to its own shim) is planned for future support.

## <u>4. Complete Examples</u>

See [aie2_placed.py](./aie2_placed.py) for a complete working example.

For additional examples with both Python and MLIR syntax, see:
- [Event Trace Example](../../../programming_examples/basic/event_trace/)

### <u>Exercises</u>
1. Build the placed design with trace: `make clean; make trace`. This compiles the placed design, generates a trace data file, and runs `parse.py` to generate the `trace_4b.json` waveform file.

## <u>5. How Trace Lowering Works in MLIR</u>

The `configure_trace()` and `start_trace()` calls generate high-level `aie.trace` MLIR ops. The compiler then runs several lowering passes to transform these into hardware configuration:

### <u>Pass 1: `-aie-insert-trace-flows`</u>

This pass does the bulk of the trace setup work:

1. **Assigns packet IDs** - Each trace source gets a unique packet ID (starting from 1 by default). Packet IDs are used to multiplex multiple trace streams over a shared connection.

2. **Infers packet types** - Determines the type based on the tile: `core` (0), `mem` (1), `shimtile` (2), or `memtile` (3). The packet type is used by the trace parser to identify which tile generated each trace packet.

3. **Creates packet flows** - Inserts `aie.packet_flow` ops that route trace data from each tile's trace port through the stream switch network to the shim tile at column 0 (tile 0,0).

4. **Configures shim DMA** - Programs the shim tile's buffer descriptor (BD) to receive trace packets and write them to a host-accessible DDR buffer via the specified XRT buffer index (`ddr_id`).

5. **Synchronizes timers** - Configures broadcast events so all traced tiles start and stop simultaneously. The shim tile generates a `USER_EVENT_1` that is broadcast (default channel 15) to all traced tiles to reset their timers and start tracing. At the end of the runtime sequence, a `USER_EVENT_0` is broadcast (default channel 14) to stop tracing.

### <u>Pass 2: `-aie-trace-to-config`</u>

Converts the `aie.trace` ops (events, mode, port monitors) into register write operations (`npu_write32`) targeting the trace control registers in each tile.

### <u>Pass 3: `-aie-trace-pack-reg-writes`</u>

Optimizes register write sequences by packing multiple small writes into fewer operations.

### <u>Pass 4: `-aie-inline-trace-config`</u>

Inlines the trace configuration into the runtime sequence so it executes as part of the normal instruction stream.

-----
[[Prev]](../section-4a) [[Up]](../) [[Next]](../section-4c)
