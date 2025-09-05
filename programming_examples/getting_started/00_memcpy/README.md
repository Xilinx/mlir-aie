# Getting Started: Measure and Analyze Peak NPU Memory Bandwidth with `memcpy`

This example is a highly parallel design that uses shim DMAs in every NPU column. It measures memory bandwidth across the full NPU, not just a single AI Engine tile with the goal to evaluate how well a design utilizes available memory bandwidth across multiple columns and channels.

For a version of the memcpy design with customizable parameters, please see [here](../../basic/memcpy/).

## Objective

* Understand the structure of a full NPU dataflow application using IRON
* Measure and report the peak memory bandwidth achieved

## Overview

This design consists of the following:

* `memcpy.py`: The NPU design for this application,
  which describes which cores of the NPU we will use, how to route data between
  cores, and what program to run on each core. This design leverages the IRON
  JIT decorator to compile the design into a binary to run on the NPU, as well as 
  to describe the program that runs on the CPU (host) that calculates a correct 
  reference output, verifies and times our NPU design's execution.
* `passThrough.cc`: A C++ vectorized kernel that exposes efficient 
  vector operations on the AI Engine using the 
  [AIE API](https://xilinx.github.io/aie_api/index.html).
* `run.lit`: lit tests that run the design on different NPU devices.

## Step-by-Step Instructions

Run and verify the design:

```shell
python3 memcpy.py
```

1. **Configure Your Run Using Parameters:**

   Use the memcpy.py variables to control:

   * `length`: Size of the full transfer in `int32_t` (must be divisible by `columns * channels` and a multiple of 1024)
   * `num_columns`: Number of NPU columns (≤ 4 for `npu`, ≤ 8 for `npu2`)
   * `num_channels`: 1 or 2 DMA channels per column

2. **Report Your Findings:**

   * Run experiments across different `num_columns`, `num_channels`, and `length` settings
   * Record latency and bandwidth
   * Identify the configuration that delivers the highest bandwidth
   * Understand the runtime sequence operations
