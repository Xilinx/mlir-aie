# Quick Start Guide - MLIR Trace Example

This guide helps you quickly get started with the MLIR trace example.

## Prerequisites

- AMD NPU device (Ryzen AI or Phoenix Point)
- MLIR-AIE toolchain installed
- XRT runtime installed
- Peano compiler available

## Quick Build and Run

```bash
cd /work/acdc/aie/mlir_trace_example

# Build everything
make clean
make -j$(nproc)

# Run the design (if you have NPU hardware)
make run

# Run with trace enabled (uses aie_new_trace.mlir by default)
make trace

# Or run with the new declarative trace explicitly
make run_new_trace
```

**Note:** The default `make trace` now uses the declarative syntax (`aie_new_trace.mlir`). 
For the legacy low-level syntax, see `aie_trace.mlir`.

## Trace Output

After running `make trace`, you'll get:

- **`trace.txt`** - Raw trace data from NPU
- **`trace_mlir.json`** - Parsed trace events in JSON format  
- **`trace_timeline.png`** - Visual timeline showing trace events
- **Console output** - Trace summary with statistics

The **trace_timeline.png** provides a visual representation of:
- Kernel execution intervals (event0 to event1)
- Lock acquisition and release patterns
- DMA operations and timing
- Port utilization
- Stream starvation events

Open `trace_timeline.png` in any image viewer to see the timeline visualization.

## Understanding the Design

The example implements: **Output[i] = Input[i] × ScaleFactor**

- Input: 4096 integers
- Scale factor: 3 (single integer)
- Output: 4096 integers (each multiplied by 3)

The computation is split across 4 sub-vectors of 1024 elements each, processed by a single AIE compute tile.

## Key Files to Study

1. **Start here:** `aie_no_trace.mlir` - Basic design without trace
2. **Declarative syntax:** `aie_new_trace.mlir` - Design with high-level trace declarations (recommended)
3. **Low-level example:** `aie_trace.mlir` - Same design with explicit register writes
4. **Compare approaches:** 
   - `diff -u aie_no_trace.mlir aie_new_trace.mlir | less` - See high-level trace additions
   - `diff -u aie_new_trace.mlir aie_trace.mlir | less` - Compare declarative vs low-level

## Two Ways to Configure Trace

### New Declarative Syntax (Recommended)

The `aie_new_trace.mlir` file uses high-level trace declarations that are easier to read and maintain:

```mlir
aie.trace(%tile_0_2) {
  aie.trace.mode event_time
  aie.trace.start_event <INSTR_EVENT_0>
  aie.trace.stop_event <INSTR_EVENT_1>
  aie.trace.packet_id 1
  aie.trace.packet_type core
  aie.trace.event <INSTR_EVENT_0>  slot=0
  aie.trace.event <INSTR_EVENT_1>  slot=1
  // ... more events
}
```

This gets automatically lowered to register writes by the compiler passes:
- `-aie-trace-to-config` - Converts to trace.config operations
- `-aie-trace-pack-reg-writes` - Optimizes register writes
- `-aiex-inline-trace-config` - Generates final npu.write32 operations

### Legacy Low-Level Syntax

The `aie_trace.mlir` file uses explicit register writes (still supported but verbose):

```mlir
aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, 
                  row = 2 : i32, value = 2038038528 : ui32}
```

**Recommendation:** Use the declarative syntax in `aie_new_trace.mlir` as your starting point.

## Trace Components in MLIR

### Declarative Trace (aie_new_trace.mlir)

The new syntax uses intuitive trace declarations:

```mlir
aie.trace(%tile_0_2) {
  // Set trace mode (event_time, event_pc, or execution)
  aie.trace.mode event_time
  
  // Define start/stop conditions
  aie.trace.start_event <INSTR_EVENT_0>
  aie.trace.stop_event <INSTR_EVENT_1>
  
  // Packet routing configuration
  aie.trace.packet_id 1
  aie.trace.packet_type core
  
  // Events to capture (up to 8 slots)
  aie.trace.event <INSTR_EVENT_0>  slot=0
  aie.trace.event <INSTR_EVENT_1>  slot=1
  aie.trace.event <INSTR_VECTOR>   slot=2
  // ... more events
}
```

The compiler automatically:
- Validates event names against the architecture database
- Checks that events are valid for the tile type (core/shim/mem)
- Encodes events into register values
- Optimizes multiple field writes into single register writes
- Generates the appropriate npu.write32 operations

### Low-Level Components (aie_trace.mlir)

For reference, the low-level implementation uses:

### 1. Packet Flows (lines 70-80)
Routes trace packets from tiles to shim DMA:
```mlir
aie.packet_flow(1) {
  aie.packet_source<%tile_0_2, Trace : 0>
  aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
} {keep_pkt_header = true}
```

### 2. Trace Control Registers (lines 90-120)
Configure what events to capture:
```mlir
aiex.npu.write32 {address = 213200 : ui32, column = 0 : i32, 
                  row = 2 : i32, value = 2038038528 : ui32}
```

### 3. Trace Buffer Descriptor (lines 130-170)
Sets up DDR buffer for trace data:
```mlir
aiex.npu.writebd {
  bd_id = 15 : i32, 
  buffer_length = 8192 : i32,
  enable_packet = 1 : i32,
  ...
}
```

### 4. Trace Flush (lines 210-215)
Ensures all trace data is written:
```mlir
aiex.npu.write32 {address = 213064 : ui32, ...}
```

## Trace Output

After running `make trace`, you'll get:

- **`trace.txt`** - Raw trace data from NPU
- **`trace_mlir.json`** - Parsed trace events in JSON format
- **Console output** - Trace summary with statistics

The trace shows:
- When the kernel starts/stops (event0/event1)
- Memory access patterns
- Stream stalls
- Lock contention
- Active vs idle time

## Modifying the Example

### Using Declarative Syntax (Recommended)

**Change traced events:**
Edit the `aie.trace` block in `aie_new_trace.mlir`:
```mlir
aie.trace(%tile_0_2) {
  // ... existing config ...
  aie.trace.event <LOCK_STALL>        slot=4  // Add lock stall events
  aie.trace.event <MEMORY_STALL>      slot=5  // Add memory stall events
}
```

**Trace multiple tiles:**
Add more trace blocks:
```mlir
aie.trace(%tile_0_3) {
  aie.trace.mode event_time
  aie.trace.start_event <INSTR_EVENT_0>
  aie.trace.stop_event <INSTR_EVENT_1>
  aie.trace.packet_id 3  // Different packet ID
  aie.trace.packet_type core
  aie.trace.event <INSTR_EVENT_0> slot=0
}
```

**Change trace mode:**
```mlir
aie.trace(%tile_0_2) {
  aie.trace.mode event_pc  // Capture program counter instead of time
  // ... rest of config
}
```

**Available events by tile type:**
- Core tiles: `INSTR_EVENT_0`, `INSTR_EVENT_1`, `INSTR_VECTOR`, `MEMORY_STALL`, `LOCK_STALL`, etc.
- Shim tiles: `DMA_S2MM_0_START_TASK`, `DMA_MM2S_0_START_TASK`, `DMA_S2MM_0_STREAM_STARVATION`, etc.
- Mem tiles: Check `utils/events_database.json` for complete list

### Using Low-Level Syntax (Advanced)

### Change trace buffer size:
Edit line 131 in `aie_trace.mlir`:
```mlir
buffer_length = 16384 : i32,  // Change from 8192 to 16384
```

### Trace additional tiles:
Add more packet flows (after line 80):
```mlir
aie.packet_flow(3) {
  aie.packet_source<%tile_0_3, Trace : 0>
  aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
} {keep_pkt_header = true}
```

### Change traced events:
Modify register values at lines 105-110 (see NPU documentation for event codes)

## Troubleshooting

**Build fails with "aiecc.py not found":**
- Ensure MLIR-AIE tools are in your PATH
- Source the appropriate setup script

**Runtime fails with "No device found":**
- Check `/dev/accel/accel0` exists
- Verify XRT is properly installed
- Check device permissions

**No trace data captured:**
- Ensure trace buffer size is sufficient
- Check that trace registers are configured correctly
- Verify packet flows are set up properly

## Next Steps

1. **Study the code:** 
   - Start with `aie_new_trace.mlir` to see declarative trace syntax
   - Compare with `aie_no_trace.mlir` to see what trace adds
   - Examine `aie_trace.mlir` to understand low-level implementation
2. **Modify events:** Add different events to the trace configuration
3. **Add tiles:** Trace multiple compute tiles simultaneously
4. **Analyze output:** Study the JSON trace file and timeline visualization
5. **Optimize:** Use trace data to find performance bottlenecks

## Compiler Pass Pipeline

When using declarative syntax, the compiler applies these passes:

1. **`-aie-trace-to-config`** - Converts `aie.trace` blocks to `aie.trace.config` with register operations
2. **`-aie-trace-pack-reg-writes`** - Optimizes by merging multiple field writes into single register writes
3. **`-aiex-inline-trace-config`** - Generates final `aiex.npu.write32` operations in runtime sequence

You can see the intermediate steps:
```bash
# After conversion to config
aie-opt -aie-trace-to-config aie_new_trace.mlir

# After register packing optimization  
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes aie_new_trace.mlir

# Final lowering (what actually runs)
aie-opt -aie-trace-to-config -aie-trace-pack-reg-writes -aiex-inline-trace-config aie_new_trace.mlir
```

## Related Documentation

- [Programming Guide Section 4b](../programming_guide/section-4/section-4b) - Python trace examples
- [README.md](./README.md) - Detailed explanation of trace components
- [AGENTS.md](../AGENTS.md) - Guide for LLM agents working with trace code
- [AIE Architecture Manual] - Hardware trace unit documentation
- `utils/events_database.json` - Complete list of available trace events by architecture
