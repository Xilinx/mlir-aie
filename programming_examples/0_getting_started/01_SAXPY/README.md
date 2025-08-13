# Getting Started: SAXPY

This example implements a SAXPY kernel, which given single-precision (i.e. `bfloat16`) inputs `a`, `X` and `Y` performs the $Z = a*X + Y$ operation. In this design `X` and `Y` are `4096` element-wide tensors, and `a` is a scalar with set value `3.141`. A single AI Engine core is used to compute the kernel.

## Overview

This design consists of the following:

* `saxpy.py`: The NPU design for this application,
  which describes which cores of the NPU we will use, how to route data between
  cores, and what program to run on each core.
* `saxpy_scalar_baseline.cc`: A C++ scalar SAXPY kernel.
* `saxpy_vector.cc`: A C++ vectorized kernel that exposes efficient 
  vector operations on the AI Engine using the 
  [AIE API](https://xilinx.github.io/aie_api/index.html).
* `test.cpp`: A program that runs on the CPU (host) to dispatch our design to 
  run on the NPU, calculates a correct reference output, verifies and times
  our NPU design's execution.
* `Makefile`: Contains the compilation instructions for the constituent
  parts of this design. Study it to see how the pieces are assembled together.

## Data Movement and Compute

Input tensors `X` and `Y` are moved from DRAM to the AI Engine compute core using ObjectFIFOs `of_x` and `of_y`. The output tensor `Z` is moved from the AI Engine compute core to DRAM using ObjectFIFO `of_z`. The AI Engine program calls the external kernel `saxpy_kernel` defined in `saxpy_vector.cc`.

## Ryzen™ AI Usage

### Compilation

To compile the design:

```shell
make
```

### C++ Testbench

To run the design:

```shell
make run
```
