# Getting Started: SAXPY

This example implements a SAXPY kernel, which given single-precision (i.e. `bfloat16`) inputs `a`, `X` and `Y` performs the $Z = a*X + Y$ operation. In this design `X` and `Y` are `4096` element-wide tensors, and `a` is a scalar with set value `3`. A single AI Engine core is used to compute the kernel.

## Overview

This design consists of the following:

* `saxpy.py`: The NPU design and host driver. Uses the IRON `@iron.jit`
  decorator to compile to an NPU binary on first call. The host driver
  computes a CPU reference and verifies the NPU output.
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
