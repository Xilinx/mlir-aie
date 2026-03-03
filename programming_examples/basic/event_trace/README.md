# MLIR Trace Example

Standalone MLIR example for tracing on AMD NPU devices using a vector-scalar multiply kernel.

## What this example contains

- `aie_trace.mlir` - declarative trace configuration (`aie.trace`, `aie.trace.event`, `aie.trace.start_config`)
- `vector_scalar_mul.cc` - AIE kernel
- `test.cpp` / `test.py` - host runners
- `visualize_trace.py` - renders a PNG timeline from parsed trace JSON
- `run_makefile.lit` / `run_strix_makefile.lit` - lit test definitions for NPU1 and NPU2

## Prerequisites

- AMD NPU device (for execution)
- MLIR-AIE toolchain on PATH (`aiecc.py`, `aie-opt`)
- XRT runtime
- Peano compiler toolchain

## Quick start

```bash
make clean
make -j"$(nproc)"
```

### Run with trace

```bash
make run_new_trace
```

### Run with trace (Python)

```bash
make run_new_trace_py
```

## Outputs

After `make run_new_trace` or `make run_new_trace_py`:
- `trace.txt` - raw trace dump
- `trace_new.json` - parsed trace events
- `trace_new_timeline.png` - timeline visualization

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
4. `-aiex-config-to-npu` (when generating final NPU operations)

Inspect intermediate IR:

```bash
aie-opt -aie-trace-to-config aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes aie_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_trace.mlir
```

## Visualization

Generate visualization from existing parsed output:

```bash
python3 visualize_trace.py -i trace_new.json -o trace_new_timeline.png -t "Trace Timeline"
```

## Common customizations

- Change traced events: edit `aie.trace.event<"...">` entries in `aie_trace.mlir`
- Add traced tiles: add additional `aie.trace @name(%tile)` blocks plus packet flow routing
- Adjust trace buffer size: update `trace_size` in `Makefile` and/or BD fields in MLIR

## Troubleshooting

**`aiecc.py` not found**
- Ensure MLIR-AIE environment is sourced and tools are on PATH.

**No device found / runtime failure**
- Verify NPU device presence and XRT installation.
- Check permissions for accelerator device nodes.

**No trace output**
- Confirm trace-enabled run target was used.
- Check packet flow and event configuration.
- Increase trace buffer size if data is truncated.

## Related references

- `../programming_guide/section-4/section-4b` (Python-based trace flow)
- `../test/create-packet-flows/trace_packet_routing.mlir` (packet-flow example)
- `../utils/events_database.json` (event database)
