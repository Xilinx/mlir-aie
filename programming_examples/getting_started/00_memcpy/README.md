<!-- Copyright (C) 2025-2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Getting Started: Measure and Analyze Peak NPU Memory Bandwidth with `memcpy`

This example is a highly parallel design that uses shim DMAs in every NPU column. It measures memory bandwidth across the full NPU, not just a single AI Engine tile with the goal to evaluate how well a design utilizes available memory bandwidth across multiple columns and channels.

For a version of the memcpy design with customizable parameters, please see [here](../../basic/memcpy/).

## Objective

* Understand the structure of a full NPU dataflow application using IRON
* Measure and report the peak memory bandwidth achieved

## Overview

This design consists of the following:

* `memcpy.py`: The NPU design and host driver. Describes which cores are used,
  how data is routed between them, and the per-core program. The design uses
  the IRON `@iron.jit` decorator to compile to an NPU binary on first call.
  The host driver warms up the JIT cache, runs a 5-iteration benchmark via
  `aie.utils.benchmark.run_iters`, reports dispatch, NPU and end-to-end
  latency, and computes effective bandwidth from the NPU time, the dispatch
  less an empty dispatch's overhead (the honest number for a
  memory-bandwidth microbenchmark).
* `run.lit`: lit tests that run the design on different NPU devices.

## Step-by-Step Instructions

Run and verify the design:

```shell
python3 memcpy.py
```

1. **Configure Your Run Using Parameters:**

   In memcpy.py, `length` in `main()` sets the size of the full transfer in
   `int32_t`. It must be a multiple of 1024 × the device's column count ×
   `num_channels`, the workers per column set by the `num_channels=2`
   argument of the `transform_parallel(...)` call. The design uses every
   column of the device (4 on `npu`, 8 on `npu2`).

2. **Report Your Findings:**

   * Run experiments across different `num_columns`, `num_channels`, and `length` settings
   * Record latency and bandwidth
   * Identify the configuration that delivers the highest bandwidth
   * Understand the runtime sequence operations
