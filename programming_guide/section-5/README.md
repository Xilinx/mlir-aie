<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 5 - Example Vector Designs</ins>

The [programming examples](../../programming_examples) are a number of sample designs which further help explain many of the unique features of AI Engines and the NPU array in Ryzen™ AI.

## Simplest

#### Passthrough

The [passthrough](../../programming_examples/basic/passthrough_kernel/) example is the simplest "getting started" example.  It copies 4096 bytes from the input to output using vectorized loads and stores.  The design example shows a typical project organization which is easy to reproduce with other examples.  There are only really 4 important files here.
1. [`aie2.py`](../../programming_examples/basic/passthrough_kernel/aie2.py) The AIE structural design which includes the shim tile connected to the external memory, and a single AIE core for performing the copy.  It also shows a simple use of the ObjectFIFOs described in [section 2](../section-2).
1. [`passthrough.cc`](../../aie_kernels/generic/passThrough.cc)  This is a C++ file which performs the vectorized copy operation.
1. [`test.cpp`](../../programming_examples/basic/passthrough_kernel/test.cpp) or [`test.py`](../../programming_examples/basic/passthrough_kernel/test.py) A C++ or Python main application for exercising the design, and comparing against a CPU reference
1. [`Makefile`](../../programming_examples/basic/passthrough_kernel/Makefile) A Makefile documenting (and implementing) the build process for the various artifacts.

The [passthrough DMAs](../../programming_examples/basic/passthrough_dmas/) example shows an alternate method of performing a copy without involving the cores, and instead performing a loopback.

## Basic

| Design name | Data type | Description |
|-|-|-|
| [Vector Scalar Add](../../programming_examples/basic/vector_scalar_add/) | i32 | Adds 1 to every element in  vector | 
| [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) | i32 | Returns a vector multiplied by a scale factor | 
| [Vector Reduce Add](../../programming_examples/basic/vector_reduce_add/) | bfloat16 | Returns the sum of all elements in a vector | 
| [Vector Reduce Max](../../programming_examples/basic/vector_reduce_max/) | bfloat16 | Returns the maximum of all elements in a vector | 
| [Vector Reduce Min](../../programming_examples/basic/vector_reduce_min/) | bfloat16 | Returns the minimum of all elements in a vector | 
| [Vector Exp](../../programming_examples/basic/vector_exp/) | bfloat16 | Returns a vector representing $e^x$ of the inputs | 

## Machine Kearning Kernels

| Design name | Data type | Description | 
|-|-|-|
| [Eltwise Add](../../programming_examples/ml/eltwise_add/) | bfloat16 | An element by element addition of two vectors | 
| [Eltwise Mul](../../programming_examples/ml/eltwise_mul/) | i32 | An element by element multiplication of two vectors | 
| [ReLU](../../programming_examples/ml/relu/) | bfloat16 | Rectified linear unit (ReLU) activation function on a vector| 
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  | 
| [Single core GEMM](../../programming_examples/basic/matrix_multiplication/single_core) | bfloat16 | A single core matrix-matrix multiply | 
| [Multi core GEMM](../../programming_examples/basic/matrix_multiplication/whole_array) | bfloat16 | A matrix-matrix multiply using 16 AIEs with operand broadcast.  Uses a simple "accumulate in place" strategy | 
| [GEMV](../../programming_examples/basic/matrix_multiplication/matrix_vector) | bfloat16 | A vector-matrix multiply returning a vector
| [Conv2D](../../programming_examples/ml/conv2d) | i8 | A single core 2D convolution for CNNs |
| [Conv2D+ReLU](../../programming_examples/ml/conv2d_fused_relu) | i8 | A Conv2D with a ReLU fused at the vector register level |

## Exercises

1. Can you modify the [passthrough](../../programming_examples/basic/passthrough_kernel/) design to copy more (or less) data? <img src="../../mlir_tutorials/images/answer1.jpg" title="Check the Makefile...PASSTHROUGH_SIZE" height=25>

1. Take a look at the testbench in our [Vector Exp](../../programming_examples/basic/vector_exp/) example [test.cpp](../../programming_examples/basic/vector_exp/test.cpp). Take note of the data type and the size of the test vector. What do you notice? <img src="../../mlir_tutorials/images/answer1.jpg" title="We are testing 65536 values or 2^16, therefore testing all possible bfloat16 values through the approximation." height=25>

1. What is the communication to computation ratio in [ReLU](../../programming_examples/ml/relu/)? <img src="../../mlir_tutorials/images/answer1.jpg" title="~6 as reported by the Trace. This is why it is a good candiate for kernel fusion with Conv2D or GEMMs for ML." height=25>

1. **HARD** Which basic example is a component in [Softmax](../../programming_examples/ml/softmax/)? <img src="../../mlir_tutorials/images/answer1.jpg" title="[Vector Exp](../../programming_examples/basic/vector_exp/)" height=25>

-----
[[Prev - Section 4](../section-4/)] [[Top](..)] [[Next - Section 6](../section-6/)]
