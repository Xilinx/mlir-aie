<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>MLIR-AIE Programming Guide</ins>

MLIR-AIE is an MLIR-based representation for AI Engine design. It provides a foundation from which complex and performant AI Engine designs can be defined and is supported by simulation and hardware impelemenation infrastructure. To better understand how AI Engine designs are defined at the MLIR level, it is recommended that you spend some time going through the [tutorial course](../tutorials/). However, this programming guide is intended to lead you through a higher level abstraction (python) of the underlying MLIR-AIE framework and provide design examples and programming tips to allow users to build designs directly. Keep in mind also that MLIR-AIE is a foundational layer in a AI Engine software development framework and while this guide provides a programmer's view for using AI Engines, it also serves as a lower layer for higher abstraction MLIR layers such as [MLIR-AIR](https://github.com/Xilinx/mlir-air).

## Outline
<details><summary><a href="./section-1">Section 1 - Basic AI Engine building blocks (tiles and buffers)</a></summary>

* Introduce AI Engine building blocks with references to Tutorial material
* Give example of python binded MLIR source for defining tiles and buffers
</details>
<details><summary><a href="./section-2">Section 2- My First Program</a></summary>

* Introduce example of first simple program (Bias Add)
* Illustrate how built-in simulation of single core design
</details>
<details><summary><a href="./section-3">Section 3 - Data Movement (Object FIFOs)</a></summary>

* Introduce topic of objectfifos and how they abstract connections between objects in the AIE array
* Point to more detailed objectfifo material in Tutorial
* Introduce key objectfifo connection patterns (link/ broadcast, join/ distribute)
</details>
<details><summary><a href="./section-4">Section 4 - Vector programming & Peformance Measurement</a></summary>

* Discuss topic of vector programming at the kernel level
* Introduce performance measurement (trace) and how we measure cycle count and efficiency
* Vector Scalar design example
</details>
<details><summary><a href="./section-5">Section 5 - Example Vector Designs</a></summary>

* Introduce additional vector design examples with exercises to measure performance on each
    * Eltwise Unary ReLU
    * Eltwise Unary e^x
    * Eltwise Binary: Vector Addition
    * Eltwise Binary: Vector Multiplication
</details>
<details><summary><a href="./section-6">Section 6 - Larger Example Designs</a></summary>

* Introduce larger design examples with performance measured over multiple cores
    * GEMM
    * CONV2D
    * Edge Detect
    * Resnet
</details>



