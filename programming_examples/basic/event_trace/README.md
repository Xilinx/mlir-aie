# MLIR Trace Example

Standalone MLIR example for tracing on AMD NPU devices using a vector-scalar multiply kernel.

## Contents

- `aie_trace.mlir` - declarative trace configuration (`aie.trace`, `aie.trace.event`, `aie.trace.start_config`)
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

## Declarative trace syntax

From `aie_trace.mlir`:

```mlir
aie.trace @core_trace(%tile_0_2) {
  aie.trace.mode "Event-Time"
  aie.trace.packet id=1 type=core
  aie.trace.event<"INSTR_EVENT_0">
  aie.trace.event<"INSTR_EVENT_1">
  aie.trace.port<0> port=DMA channel=0 direction=S2MM
  aie.trace.start event=<"BROADCAST_15">
  aie.trace.stop event=<"BROADCAST_14">
}

aie.runtime_sequence(...) {
  aie.trace.start_config @core_trace
}
```

Compiler lowering pipeline for declarative trace:
1. `-aie-trace-to-config`
2. `-aie-trace-pack-reg-writes`
3. `-aie-inline-trace-config`

Inspect intermediate IR:

```bash
aie-opt -aie-trace-to-config aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_trace.mlir
```

## Example Visualization

Generate visualization from existing parsed output:

```bash
python3 visualize_trace.py -i trace.json -o trace_timeline.png -t "Trace Timeline"
```

Copyright 2026 Advanced Micro Devices, Inc.