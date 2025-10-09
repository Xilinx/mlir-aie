<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Chaining Channels</ins>

This reference design demonstrates low-level DMA control and channel chaining on a Ryzen™ AI NPU (npu2_1col).

## Overview

This [design](./chaining_channels_placed.py) showcases advanced DMA programming techniques:

1. **MemTile Buffer Initialization**: A 1kB buffer in MemTile (row 1) is initialized with values 1-256 using the `initial_value` parameter.

2. **Explicit DMA Control**: The MemTile DMA is controlled via explicit locks rather than automatic ObjectFIFO management. The DMA waits for a lock release triggered by `npu_write32` in the runtime sequence.

3. **Low-Level Runtime Sequence**: Uses explicit DMA buffer descriptor operations:
   - `npu_writebd` - Configure buffer descriptors with lock parameters
   - `npu_address_patch` - Patch buffer addresses into BDs
   - `npu_push_queue` - Push BDs to DMA queues
   - `npu_sync` - Wait for DMA completion

4. **Lock-Based Sequencing**: DMA operations are sequenced using locks in the buffer descriptors rather than tokens, providing explicit control over data flow timing.

5. **Data Flow**:
   - **Write Path**: MemTile (1KB initialized data) → DDR buffer A
   - **Read Path**: DDR buffer B (4KB) → ComputeTile (data discarded, locks toggled)

## Architecture

The design uses:
- **ShimTile** (row 0) - Interfaces with external DDR memory
- **MemTile** (row 1) - Stores initialized data, controlled by lock at address 0xC0000
- **ComputeTile** (row 2) - Receives data and toggles locks in an infinite loop
- **Explicit Flows** - Connect MemTile→ShimTile and ShimTile→ComputeTile DMAs

## Building and Running

To compile and run the design:
```shell
make        # Build without tracing
make run    # Run test
```

### With Tracing

To enable ShimTile tracing for debugging DMA channels and data traffic:
```shell
make TRACE=1            # Build with tracing enabled
make run TRACE=1        # Run with tracing
make parse_trace        # Parse and analyze trace data
```

The trace will capture DMA channel activity, lock operations, and data traffic on the ShimTile. Trace data is written to `trace.txt` and can be analyzed with the parse_trace target, which generates `trace_chaining_channels.json` and displays a summary.

### Cleaning

```shell
make clean              # Clean all build artifacts and trace files
make clean_trace        # Clean only trace files
```

## Verification

The test verifies that buffer A contains the initialized pattern (values 1-256) that was written from the MemTile.

