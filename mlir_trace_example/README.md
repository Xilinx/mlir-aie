# MLIR Trace Example

This is a standalone MLIR example demonstrating how to enable and configure trace functionality on AMD NPU devices. It implements a simple vector-scalar multiplication with trace enabled.

## Quick Comparison: With and Without Trace

This directory contains two MLIR files to illustrate what trace adds to a design:

- **`aie_no_trace.mlir`** - Basic design without trace (minimal, ~90 lines)
- **`aie_trace.mlir`** - Same design WITH trace enabled (~220 lines)

**Trace adds:**
1. Packet flows to route trace data (`aie.packet_flow`)
2. Trace control register configuration (`aiex.npu.write32` operations)
3. Buffer descriptor for trace data capture (`aiex.npu.writebd`)
4. Trace completion/flush operations

Use `diff aie_no_trace.mlir aie_trace.mlir` to see exactly what changes.

## Overview

This example shows how to write trace configuration directly in MLIR without using Python wrapper utilities. It's useful for understanding the low-level trace mechanisms and for creating custom trace configurations.

## Key Components

### 1. Trace Packet Flows (`aie.packet_flow`)

Packet flows route trace data from source tiles to destination DMA channels:

```mlir
aie.packet_flow(1) {
  aie.packet_source<%tile_0_2, Trace : 0>
  aie.packet_dest<%shim_noc_tile_0_0, DMA : 1>
} {keep_pkt_header = true}
```

This creates a packet-switched route from compute tile (0,2)'s trace unit to the shim tile (0,0) DMA channel 1.

### 2. Trace Control Register Configuration (`aiex.npu.write32`)

These operations configure the trace control registers for each tile:

- **Address 213200 (0x34110)**: Trace Control 0 - enables trace with event selection
- **Address 213204 (0x34114)**: Trace Control 1 - configures trace mode and packet generation
- **Address 213216 (0x34120)**: Trace Event 0 - selects events 0-3 to trace
- **Address 213220 (0x34124)**: Trace Event 1 - selects events 4-7 to trace

### 3. Trace Buffer Descriptor (`aiex.npu.writebd`)

Configures a buffer descriptor (BD 15) to capture trace packets to DDR memory:

```mlir
aiex.npu.writebd {
  bd_id = 15 : i32, 
  buffer_length = 8192 : i32,      // 8KB trace buffer
  burst_length = 64 : i32,         // 64-byte DMA bursts
  enable_packet = 1 : i32,         // Enable packet mode for trace
  ...
}
```

### 4. Trace Completion

At the end of execution, write trace done events to flush any buffered trace data:

```mlir
aiex.npu.write32 {address = 213064 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
aiex.npu.write32 {address = 213000 : ui32, column = 0 : i32, row = 0 : i32, value = 126 : ui32}
```

## Files

- `aie_trace.mlir` - Main MLIR file with trace configuration
- `aie_no_trace.mlir` - Same design WITHOUT trace (for comparison)
- `vector_scalar_mul.cc` - AIE kernel that multiplies a vector by a scalar
- `test.cpp` - Host application to run the design
- `Makefile` - Build system
- `CMakeLists.txt` - CMake configuration for building the host application
- `test.sh` - Automated test script
- `visualize_trace.py` - Python script to visualize trace data as PNG timeline

## Trace Output Files

After running `make trace`, you'll have:
- **`trace.txt`** - Raw trace data from NPU
- **`trace_mlir.json`** - Parsed trace events in JSON format
- **`trace_timeline.png`** - Visual timeline of trace events (PNG image)

## Building and Running

### Quick Start (if you have NPU hardware):

```bash
./test.sh
```

This automated script will clean, build, and run the example with trace enabled.

### Manual Build:

#### Build the design:

```bash
make
```

This will:
1. Compile the AIE kernel (`vector_scalar_mul.cc`) using Peano compiler
2. Compile the MLIR design to generate xclbin and NPU instructions

#### Build the host executable:

```bash
make mlir_trace_example.exe
```

This builds the C++ host application that runs on the x86 CPU.

### Run the design with trace:

```bash
make trace
```

This will:
1. Execute the design on the NPU
2. Capture trace data to `trace.txt`
3. Parse the trace data to generate `trace_mlir.json`
4. Display a trace summary
5. **Generate a PNG visualization (`trace_timeline.png`)**

### Visualize existing trace data:

If you already have `trace_mlir.json`, you can regenerate just the visualization:

```bash
make visualize
```

Or run the script directly with custom options:

```bash
python3 visualize_trace.py -i trace_mlir.json -o my_trace.png -t "My Custom Title"
```

### Run without trace processing:

```bash
make run
```

This runs the design but doesn't process trace data (though trace is still captured).

### Build without NPU hardware:

You can build all artifacts without running on hardware:
```bash
make              # Builds xclbin and insts.bin
make mlir_trace_example.exe   # Builds host executable
```

The build artifacts can then be copied to a machine with NPU hardware for execution.

## Understanding Trace Events

The kernel uses two trace events:
- `event0()` - Marks the start of the computation
- `event1()` - Marks the end of the computation

These events, along with other system events (memory stalls, stream stalls, etc.), are captured in the trace and can be visualized using the trace JSON output.

### Trace Visualization

The `visualize_trace.py` script creates a PNG timeline showing:
- **Multiple processes** - Separate rows for core trace and shim trace
- **Thread activity** - Each thread (event type) gets its own lane
- **Event intervals** - Color-coded rectangles showing when events occur
- **Event names** - Labels on larger intervals
- **Legend** - Shows top event types with occurrence counts

The timeline makes it easy to see:
- When the kernel executes (INSTR_EVENT_0 to INSTR_EVENT_1)
- Lock contention patterns (LOCK_STALL events)
- DMA activity (DMA_START_TASK, DMA_FINISHED_TASK)
- Port utilization (PORT_RUNNING_0, PORT_RUNNING_1)
- Stream starvation issues

You can open the PNG in any image viewer or Chrome trace viewer.

## Trace Buffer Configuration

The example uses:
- **Trace buffer size**: 8KB (8192 bytes)
- **DMA burst length**: 64 bytes
- **Buffer location**: 5th XRT buffer (arg_idx = 4)

You can modify these by editing the `aiex.npu.writebd` operation in the MLIR file.

## Customizing Trace

To trace different events or tiles:

1. **Add more tiles to trace**: Add additional `aie.packet_flow` declarations
2. **Change traced events**: Modify the values in the trace event registers (addresses 213216 and 213220)
3. **Adjust buffer size**: Change `buffer_length` in the `aiex.npu.writebd` operation

## Comparison with Python Trace Utilities

This example shows the low-level MLIR equivalent of what the Python trace utilities (`aie.utils.trace`) generate automatically. The Python utilities are recommended for most use cases, but understanding the MLIR representation is valuable for:

- Debugging trace issues
- Creating custom trace configurations
- Understanding the underlying hardware mechanisms
- Integrating trace with custom MLIR passes

## Related Examples

- [Section 4b - Trace (Python)](../programming_guide/section-4/section-4b) - High-level Python trace example
- [Trace Packet Routing Test](../test/create-packet-flows/trace_packet_routing.mlir) - Simple packet flow test

## Hardware Details

For more information about NPU trace units and configuration registers, refer to:
- NPU Architecture Manual
- AIE API Documentation
- Trace control register specifications in the hardware documentation
