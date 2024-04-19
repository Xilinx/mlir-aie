<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Softmax

The softmax function is a mathematical function commonly used in machine learning, especially in classification tasks. It transforms a vector of real-valued scores (often called logits) into a probability distribution. The resulting probabilities are positive and sum up to 1, making them suitable for representing categorical distributions.

## Key Characteristics
* Exponential Normalization: The softmax function applies the exponential function to each element of the input vector and then normalizes these values by dividing by the sum of all these exponentials. This has the effect of amplifying the differences between the elements of the input vector, making the highest values stand out more prominently.

* Formula: For a vector,

    ```math
    \mathbf{z} = \begin{bmatrix} z_1 & z_2 & \cdots & z_n \end{bmatrix} 
    ```

    the softmax function for each element is,

    ```math
    \sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}} 
    ```

    where e is the base of the natural logarithm.

* Output as Probabilities: The output of the softmax function is a vector where each component is between 0 and 1, and the sum of all components is 1. This makes it useful for interpreting the outputs as probabilities.


## Compilation details

The softmax function employs the exponential function $e^x$, similar to the example found [here](../../basic/eltwise_exp/). Again to efficiently implement softmax, a lookup table approximation is utilized.

In addition, and unlike any of the other current design examples, this example uses MLIR dialects as direct input, including the `vector`,`affine`,`arith` and `math` dialects.  This is shown in the [source](./bf16_softmax.mlir).  This is intended to be generated from a higher level description, but is shown here as an example of how you can use other MLIR dialects as input.

The compilation process is different from the other design examples, and is shown in the [Makefile](./Makefile).

1. The input MLIR is first vectorized into chunks of size 16, and a C++ file is produced which has mapped the various MLIR dialects into AIE intrinsics, including vector loads and stores, vectorized arithmetic on those registers, and the $e^x$ approximation using look up tables
1. This generated C++ is compiled into a first object file
1. A file called `lut_based_ops.cpp` from the AIE2 runtime libary is compiled into a second object file.  This file contains the look up table contents to approximate the $e^x$ function.
1. A wrapper file is also compiled into an object file, which prevents C++ name mangling, and allows the wrapped C function to be called from the strucural Python
1. These 3 object files and combined into a single .a file, which is then referenced inside the aie2.py structural Python.

This is a slightly more complex process than the rest of the examples, which typically only use a single object file containing the wrapped C++ function call, but is provided to show how a library based flow can also be used.

## Usage

### C++ Testbench

To compile the design and C++ testbench:

```
make
```

To run the design:

```
make run
```