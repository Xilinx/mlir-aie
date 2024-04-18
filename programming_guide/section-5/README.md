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

There are a number of example designs available [here](../../programming_examples/) which further help explain many of the unique features of the NPU.

## Simplest

### Pass through

The [passthrough](../../programming_examples/basic/passthrough_kernel/) example is the simplest "getting started" example.  It copies 4096 bytes from the input to output using vectorized loads and stores.  The design example shows a typical project organization which is easy to reproduce with other examples.  There are only really 4 important files here.
    * [`aie2.py`](../../programming_examples/basic/passthrough_kernel/aie2.py) The AIE structural design which includes the shim tile connected to the external memory, and a single AIE core for performing the copy.  It also shows a simple use of the ObjectFIFOs described in [section 2](../section-2)
    * [`passthrough.cc`](../../aie_kernels/generic/passThrough.cc)  This is a C++ file which performs the vectorized copy operation.
    * [`test.cpp`](../../programming_examples/basic/passthrough_kernel/test.cpp) or [`test.py`](../../programming_examples/basic/passthrough_kernel/test.py) A C++ or Python main application for exercising the design, and comparing against a CPU reference
    * [`Makefile`](../../programming_examples/basic/passthrough_kernel/Makefile) A Makefile documenting (and implementing) the build process for the various artifacts.

The [passthrough DMAs](../../programming_examples/basic/passthrough_dmas/) example shows an alternate method of performing a copy without involving the cores, and instead performing a loopback.

## Simple

| Design name | Data type | Description |
|-|-|-|
| [Vector Scalar Add](../../programming_examples/basic/vector_scalar_add/) | i32 |Adds 1 to every element in  vector | 
| [Vector Scalar Mul](../../programming_examples/basic/vector_scalar_mul/) | i32 | Does something more complicated | 
| [Vector Reduce Add](../../programming_examples/basic/vector_reduce_add/) | bfloat16 | Returns the sum of all elements in a vector | 
| [Vector Reduce Max](../../programming_examples/basic/vector_reduce_max/) | bfloat16 | Returns the maximum of all elements in a vector | 
| [Vector Reduce Min](../../programming_examples/basic/vector_reduce_min/) | bfloat16 | Returns the minimum of all elements in a vector | 
| [Vector $e^x$](../../programming_examples/basic/vector_exp/) | bfloat16 | Returns a vector representing $e^x$ of the inputs | 

## Machine learning kernels

| Design name | Data type | Description | 
|-|-|-|
| [Eltwise Add](../../programming_examples/ml/eltwise_add/) | bfloat16 | An element by element addition of two vectors | 
| [Eltwise Mul](../../programming_examples/ml/eltwise_mul/) | i32 | An element by element multiplication of two vectors | 
| [ReLU](../../programming_examples/ml/relu/) | bfloat16 | Rectified linear unit (ReLU) activation function on a vector| 
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  | 
| [Single core GEMM](../../programming_examples/basic/matrix_multiplication/single_core) | bfloat16 | A single core matrix-matrix multiply | 
| [Multi core GEMM](../../programming_examples/basic/matrix_multiplication/whole_array) | bfloat16 | A matrix-matrix multiply using 16 AIEs with operand broadcast.  Uses a simple "accumulate in place" strategy | 
| [GEMV](../../programming_examples/basic/matrix_multiplication/matrix_vector) | bfloat16 | A vector-matrix multiply returning a vector
| [Conv2D](../../programming_examples/basic/vector_exp/) | i8 | A Conv2D | 


