# MLIR Trace Example

Vector × scalar AIE design with custom hardware-event tracing on AMD NPU devices.

## Contents

- `aie_trace.py` — Iron (`@iron.jit`) design that wires custom `coretile_events` / `coremem_events` / `memtile_events` / `shimtile_events` lists straight through `rt.enable_trace()`.
- `vector_scalar_mul.cc` — AIE kernel, built into the JIT work dir via `ExternalFunction(source_file=…)`.
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

The whole design lives in [`aie_trace.py`](./aie_trace.py).  Custom event lists go straight on the iron `Runtime`:

```python
import aie.iron as iron
from aie.iron import Compile, In, Out, ObjectFifo, Runtime, Worker
from aie.utils.trace.events import CoreEvent, MemEvent, MemTileEvent, ShimTileEvent

@iron.jit
def aie_trace(A: In, F: In, C: Out, *, tensor_size: Compile[int] = 4096, ...):
    of_in = ObjectFifo(...)
    ...
    # `trace=1` flags the worker for hardware tracing.
    worker = Worker(core_fn, fn_args=[...], trace=1)

    rt = Runtime()
    with rt.sequence(...) as (a_in, f_in, c_out):
        rt.enable_trace(
            trace_size=8192,
            workers=[worker],
            coretile_events=[CoreEvent.INSTR_EVENT_0, ...],   # up to 8
            coremem_events=[MemEvent.DMA_S2MM_0_START_TASK, ...],
            memtile_events=[...],
            shimtile_events=[...],
        )
        rt.start(worker)
        ...
```

Iron's `rt.enable_trace()` forwards the four event lists to the same `aie.utils.trace.configure_trace` machinery the lower-level dialect API uses; the difference is you no longer have to talk to `@device` / `tile()` / `object_fifo` / `configure_trace` directly.

## Compiler Pipeline

Compiler lowering pipeline for declarative trace:
1. `-aie-insert-trace-flows`
2. `-aie-trace-to-config`
3. `-aie-trace-pack-reg-writes`
4. `-aie-inline-trace-config`

Inspect intermediate IR for a `make`-built design:

```bash
aie-opt -aie-insert-trace-flows build/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config build/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes build/input_with_addresses.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config build/input_with_addresses.mlir
```

## Example Visualization

Generate visualization from existing parsed output:

```bash
python3 visualize_trace.py -i trace.json -o trace_timeline.png -t "Trace Timeline"
```

Copyright 2026 Advanced Micro Devices, Inc.
