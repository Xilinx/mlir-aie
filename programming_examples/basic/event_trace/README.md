# MLIR Trace Example

Standalone MLIR example for tracing on AMD NPU devices using a vector-scalar multiply kernel.

## Contents

- `aie_trace.mlir` - declarative trace configuration using low-level MLIR ops
- `aie_trace.py` - declarative trace configuration using Python `configure_trace()` API
- `vector_scalar_mul.cc` - AIE kernel
- `test.cpp` / `test.py` - host runners
- `visualize_trace.py` - renders a PNG timeline from parsed trace JSON
- `run_makefile.lit` / `run_strix_makefile.lit` - lit test definitions for NPU1 and NPU2

### Run with trace

```bash
make run_trace
```

### Run with trace (Python)

```bash
make run_trace_py
```

## Outputs

After `make run_trace` or `make run_trace_py`:
- `trace.txt` - raw trace dump
- `trace.json` - parsed trace events
- `trace_timeline.png` - timeline visualization

## Declarative Trace APIs

### Python API (`aie_trace.py`)

The recommended approach uses `configure_trace()` and `configure_trace_output()`:

```python
import aie.utils.trace as trace_utils
from aie.utils.trace.events import PortEvent, CoreEvent, MemEvent

# Outside runtime_sequence - configure which tiles to trace
tiles_to_trace = [tile_0_2, tile_0_2, mem_tile_0_1, shim_tile_0_0]
trace_utils.configure_trace(
    tiles_to_trace,
    coretile_events=[
        CoreEvent.INSTR_EVENT_0,
        CoreEvent.INSTR_EVENT_1,
        PortEvent(CoreEvent.PORT_RUNNING_0, WireBundle.DMA, 0, True),
        # ... up to 8 events
    ],
)

# Inside runtime_sequence - activate tracing
@runtime_sequence(...)
def sequence(...):
    trace_utils.configure_trace_output()
    # ... data transfers
```

**Note:** To trace both core and memory events on a core tile, list it twice in
`tiles_to_trace`. The first occurrence configures core trace, the second
configures memory trace.

### MLIR Syntax (`aie_trace.mlir`)

For direct MLIR usage:

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

Inspect intermediate IR:

```bash
aie-opt -aie-insert-trace-flows aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes aie_trace.mlir
aie-opt -aie-insert-trace-flows -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_trace.mlir
```

## Example Visualization

Generate visualization from existing parsed output:

```bash
python3 visualize_trace.py -i trace.json -o trace_timeline.png -t "Trace Timeline"
```

Copyright 2026 Advanced Micro Devices, Inc.