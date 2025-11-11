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

# Run with trace enabled
make trace
```

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
```

## Understanding the Design

The example implements: **Output[i] = Input[i] × ScaleFactor**

- Input: 4096 integers
- Scale factor: 3 (single integer)
- Output: 4096 integers (each multiplied by 3)

The computation is split across 4 sub-vectors of 1024 elements each, processed by a single AIE compute tile.

## Key Files to Study

1. **Start here:** `aie_no_trace.mlir` - Basic design without trace
2. **Then compare:** `aie_trace.mlir` - Same design with trace added
3. **Run diff:** `diff -u aie_no_trace.mlir aie_trace.mlir | less`

## Trace Components in MLIR

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

1. **Study the code:** Compare `aie_no_trace.mlir` vs `aie_trace.mlir`
2. **Modify events:** Change which events are traced
3. **Add tiles:** Trace multiple compute tiles
4. **Analyze output:** Study the JSON trace file
5. **Optimize:** Use trace data to find performance bottlenecks

## Related Documentation

- [Programming Guide Section 4b](../programming_guide/section-4/section-4b) - Python trace examples
- [README.md](./README.md) - Detailed explanation of trace components
- [AIE Architecture Manual] - Hardware trace unit documentation
