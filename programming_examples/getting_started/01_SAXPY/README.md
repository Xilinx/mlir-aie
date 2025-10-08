# Getting Started: SAXPY

This example implements a SAXPY kernel, which given single-precision (i.e. `bfloat16`) inputs `a`, `X` and `Y` performs the $Z = a*X + Y$ operation. In this design `X` and `Y` are `4096` element-wide tensors, and `a` is a scalar with set value `3`. A single AI Engine core is used to compute the kernel.

## Overview

This design consists of the following:

* `saxpy.py`: The NPU design for this application,
  which describes which cores of the NPU we will use, how to route data between
  cores, and what program to run on each core. This design leverages the IRON
  JIT decorator to compile the design into a binary to run on the NPU, as well as 
  to describe the program that runs on the CPU (host) that calculates a correct 
  reference output, verifies and times our NPU design's execution.
* `saxpy.cc`: Contains both a scalar C++ SAXPY kernel (saxpy_scalar) and a vectorized kernel (saxpy) that exposes efficient 
  vector operations on the AI Engine using the 
  [AIE API](https://xilinx.github.io/aie_api/index.html).
* `run.lit`: lit test that runs the design on different NPU devices.

## Data Movement and Compute

Input tensors `X` and `Y` are moved from DRAM to the AI Engine compute core using ObjectFIFOs `of_x` and `of_y`. The output tensor `Z` is moved from the AI Engine compute core to DRAM using ObjectFIFO `of_z`. The AI Engine program calls the external kernel `saxpy_kernel` defined in `saxpy.cc`.

## Ryzen™ AI Usage

Run and verify the design:

```shell
python3 saxpy.py
```
