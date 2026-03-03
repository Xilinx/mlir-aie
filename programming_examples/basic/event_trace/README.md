# MLIR Trace Example

Standalone MLIR example for tracing on AMD NPU devices using a vector-scalar multiply kernel.
This directory now has a single guide that combines the old README and QUICKSTART content.

## What this example contains

- `aie_no_trace.mlir` - baseline design with no trace setup
- `aie_trace.mlir` - low-level/manual trace configuration (`aiex.npu.write32`, `aiex.npu.writebd`, etc.)
- `aie_new_trace.mlir` - declarative trace configuration (`aie.trace`, `aie.trace.event`, `aie.trace.start_config`)
- `vector_scalar_mul.cc` - AIE kernel
- `test.cpp` / `test.py` - host runners
- `visualize_trace.py` - renders a PNG timeline from parsed trace JSON

## Prerequisites

- AMD NPU device (for execution)
- MLIR-AIE toolchain on PATH (`aiecc.py`, `aie-opt`)
- XRT runtime
- Peano compiler toolchain

## Quick start

```bash
cd /work/acdc/aie/mlir_trace_example
make clean
make -j"$(nproc)"
make run
```

### Run with trace (legacy low-level MLIR path)

```bash
make trace
```

This target currently uses `aie_trace.mlir`.

### Run with declarative trace syntax

```bash
make run_new_trace
```

This target uses `aie_new_trace.mlir`.

## Outputs

After `make trace`:
- `trace.txt` - raw trace dump
- `trace_mlir.json` - parsed trace events
- `trace_timeline.png` - timeline visualization

After `make run_new_trace`:
- `trace.txt`
- `trace_new.json`
- `trace_new_timeline.png`

## Two trace styles

### 1) Declarative syntax (recommended)

From `aie_new_trace.mlir`:

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
aie-opt -aie-trace-to-config aie_new_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes aie_new_trace.mlir
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes -aie-inline-trace-config aie_new_trace.mlir
```

### 2) Low-level/manual syntax

From `aie_trace.mlir`:

```mlir
aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, row = 2 : i32, value = ... : ui32}
```

This is useful for debugging and understanding exact hardware register programming.

## Compare files quickly

```bash
diff -u aie_no_trace.mlir aie_new_trace.mlir
diff -u aie_new_trace.mlir aie_trace.mlir
```

## Visualization

Generate visualization from existing parsed output:

```bash
make visualize
```

Or directly:

```bash
python3 visualize_trace.py -i trace_mlir.json -o trace_timeline.png -t "Trace Timeline"
```

## Common customizations

- Change traced events: edit `aie.trace.event<"...">` entries in `aie_new_trace.mlir`
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
