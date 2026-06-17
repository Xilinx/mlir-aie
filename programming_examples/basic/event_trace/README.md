# MLIR Trace Example

Vector × scalar AIE design with custom hardware-event tracing on AMD NPU devices.

## Contents

- `aie_trace.py` — IRON (`@iron.jit`) design that attaches custom hardware-event lists via `Worker(trace=TileTrace(events=[...]))` for the compute tile and `Program(trace_tiles=[TileTrace(tile=..., events=[...])])` for the mem and shim tiles, with the trace sink configured by `Program(trace=TraceBuffer(...))`.  The AIE compute kernel is the library `kernels.scale` (scalar variant) — `event0()` / `event1()` markers are already baked into the library source.
- `test.cpp` / `test.py` — host runners (C++ via `make run_trace`, Python via `make run_trace_py`).
- `visualize_trace.py` — renders a PNG timeline from parsed trace JSON.
- `run_makefile.lit` / `run_strix_makefile.lit` — lit test definitions for NPU1 and NPU2.

### Run with trace

```bash
make run_trace        # C++ testbench
make run_trace_py     # Python testbench
```

## Outputs

After `make run_trace` or `make run_trace_py`:
- `trace.txt` — raw trace dump
- `trace.json` — parsed trace events
- `trace_timeline.png` — timeline visualization

## How the trace is wired

The whole design lives in [`aie_trace.py`](./aie_trace.py).  Custom event lists are attached per tile via `TileTrace`, and the trace sink is the `TraceBuffer` passed to `Program`:

```python
import aie.iron as iron
from aie.iron import CompileTime, In, Out, ObjectFifo, Program, Worker
from aie.iron import TileTrace, TraceBuffer
from aie.utils.trace.events import (
    CoreEvent, MemEvent, MemTileEvent, ShimTileEvent, PortEvent, WireBundle,
)

@iron.jit
def aie_trace(A: In, F: In, C: Out, *, tensor_size: CompileTime[int] = 4096, ...):
    of_in = ObjectFifo(...)
    ...
    # A single mixed event list configures both hardware units of the compute
    # tile: TileTrace infers the unit from each event's type (CoreEvent.* -> core
    # unit, MemEvent.* -> core-memory unit). The old coretile_events /
    # coremem_events lists combine into this one events=[...] list.
    worker = Worker(
        core_fn,
        fn_args=[...],
        trace=TileTrace(
            events=[
                CoreEvent.INSTR_EVENT_0,             # core-unit events
                CoreEvent.INSTR_EVENT_1,
                MemEvent.DMA_S2MM_0_START_TASK,      # core-memory-unit events
                ...,
            ],
        ),
    )

    def runtime_sequence(a_in, f_in, c_out):
        of_in.prod().fill(a_in)
        ...

    # Non-worker trace sources (mem tile / shim tile) go through trace_tiles, and
    # the trace sink is the TraceBuffer (trace_size / ddr_id / trace_file live on
    # it). The worker's compute tile is traced via Worker(trace=...) above.
    return Program(
        iron.get_current_device(),
        runtime_sequence,
        arg_types=[tensor_ty, scalar_ty, tensor_ty],
        workers=[worker],
        trace_tiles=[
            TileTrace(tile=mem_tile, events=[MemTileEvent...]),
            TileTrace(tile=shim_tile, events=[ShimTileEvent...]),
        ],
        trace=TraceBuffer(trace_size=8192),
    ).resolve_program()
```

`TileTrace` / `TraceBuffer` forward to the same `aie.utils.trace.configure_trace` machinery the lower-level dialect API uses; the difference is you no longer have to talk to `@device` / `tile()` / `object_fifo` / `configure_trace` directly.

## Lowering reference

For direct MLIR usage, the declarative `aie.trace` syntax that
`TileTrace` / `TraceBuffer` ultimately lower to looks like:

```mlir
aie.trace @core_trace(%tile_0_2) {
  aie.trace.mode "Event-Time"
  aie.trace.packet id=1 type=core
  aie.trace.event<"INSTR_EVENT_0">
  aie.trace.event<"INSTR_EVENT_1">
  aie.trace.port<0> port=DMA channel=0 direction=S2MM
  aie.trace.start broadcast=15
  aie.trace.stop broadcast=14
}

aie.runtime_sequence(...) {
  aie.trace.start_config @core_trace
}
```

## Compiler Pipeline

Compiler lowering pipeline for declarative trace:
1. `-aie-insert-trace-flows`
2. `-aie-trace-to-config`
3. `-aie-trace-pack-reg-writes`
4. `-aie-inline-trace-config`

Inspect intermediate IR for a `make`-built design:

```bash
aie-opt -aie-insert-trace-flows build/final.prj/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config build/final.prj/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes build/final.prj/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config build/final.prj/input_with_addresses.mlir
```

## Example Visualization

Generate visualization from existing parsed output:

```bash
python3 visualize_trace.py -i trace.json -o trace_timeline.png -t "Trace Timeline"
```

Copyright 2026 Advanced Micro Devices, Inc.
