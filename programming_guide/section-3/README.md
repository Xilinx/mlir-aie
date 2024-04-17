<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3 - My First Program</ins>

<img align="right" width="500" height="250" src="../assets/binaryArtifacts.svg">

This section creates a first program that will run on the AIE-array. As shown in the figure on the right, we will have to create both binaries for the AIE-array (device) and CPU (host) parts. For the AIE-array, a structural description and kernel code is compiled into the AIE-array binaries. The host code loads the AIE-array binaries and contains the test functionality.

For the AIE-array structural description we will combine what you learned in [section-1](../section-1) for defining a basic structural design in python with the data movement part from [section-2](../section-2).

For the AIE kernel code, we will start with non-vectorized code that will run on the scalar processor part of an AIE. [section-4](../section-4) will introduce how to vectorize a compute kernel to harvest the compute density of the AIE.

The host code can be written in either C++ (as shown in the figure) or in Python. We will also introduce some convenience utility libraries for typical test functionality and to simplify context and buffer creation when the [Xilinx RunTime (XRT)](https://github.com/Xilinx/XRT) is used, for instance in the [AMD XDNA Driver](https://github.com/amd/xdna-driver) for Ryzen AI devices. 

* Introduce example of first simple program (vector scalar multiply)
    * Walk through syntax of aie2.py, test.cpp, test_utils.h, maybe CMakeLists.txt and Makefile/ makefile-common as well
    * need to remove trace parts from test.cpp for now and move it to Section-4

* Illustrate how built-in simulation of single core design
* Illustrate how to run designs on Ryzen AI enabled hardware
