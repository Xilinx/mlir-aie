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

This reference design can be run on a Ryzen™ AI NPU.

In this [design](./chaining_channels_placed.py), data is initialized in a MemTile buffer (row 1) using ObjectFIFO `initValues`, written to external DDR memory (1kB), and then a larger buffer (4kB) is read back into a ComputeTile. This demonstrates:

1. **MemTile Buffer Initialization**: A 1kB single-buffered ObjectFIFO is allocated in the MemTile and initialized with a pattern (values 1 to N) using the `initValues` parameter (feature from PR #1813).

2. **Write to DDR**: The initialized 1kB data is transferred directly from the MemTile to external DDR memory (buffer A) using ObjectFIFO DMA operations.

3. **Read from DDR**: A 4kB buffer (buffer B) is then read back from DDR into a ComputeTile via a double-buffered ObjectFIFO (for concurrency between core and tileDMA).

4. **Optional Verification Path**: When enabled, the ComputeTile can write the read data to buffer C for host verification. Otherwise, data is acquired and released (discarded) without processing.

5. **DMA Task Operations**: The runtime sequence uses `shim_dma_single_bd_task`, `dma_start_task`, and `dma_await_task` operations instead of `dma_memcpy_nd` for explicit DMA task management.

6. **Buffer Initialization**: 
   - Buffer A (1KB): Write buffer, initialized with 0xDEADBEEF (overwritten by MemTile data)
   - Buffer B (4KB): Read buffer, initialized with increasing values (0, 1, 2, ...)
   - Buffer C (4KB, optional): Verification buffer for host-side checking

The design uses:
- A **ShimTile** (row 0) for interfacing with external memory
- A **MemTile** (row 1) for data storage with initialized values
- A **ComputeTile** (row 2) that processes data from the double-buffered read ObjectFIFO
- **ObjectFIFOs** with `initValues` to manage data movement and initialization

This example is written in the lower-level placed version of IRON, using explicit tile placement.

To compile and run the design for NPU:
```shell
make
make run
```

To enable verification path:
```shell
make run VERIFY=1
```

The test verifies that buffer A was correctly written with the initialized pattern (values 1 to N). When verification is enabled, it also verifies that buffer C contains the data read from buffer B.
