<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
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

The softmax function employs the exponential function $e^x$, similar to the example found [here](../../basic/vector_exp/). Again to efficiently implement softmax, a lookup table approximation is utilized.

In addition, and unlike any of the other current design examples, this example uses MLIR dialects as direct input, including the `vector`,`affine`,`arith` and `math` dialects.  This is shown in the [source](./bf16_softmax.mlir).  This is intended to be generated from a higher-level description but is shown here as an example of how you can use other MLIR dialects as input.

The compilation process is different from the other design examples, and is shown in the [Makefile](./Makefile).

1. The input MLIR is first vectorized into chunks of size 16, and a C++ file is produced which has mapped the various MLIR dialects into AIE intrinsics, including vector loads and stores, vectorized arithmetic on those registers, and the $e^x$ approximation using look up tables
1. This generated C++ is compiled into a first object file
1. A file called `lut_based_ops.cpp` from the AIE2 runtime library is compiled into a second object file.  This file contains the look up table contents to approximate the $e^x$ function.
1. A wrapper file is also compiled into an object file, which prevents C++ name mangling, and allows the wrapped C function to be called from the strucural Python
1. These 3 object files are combined into a single .a file, which is then referenced inside the `softmax.py` structural Python.

This is a slightly more complex process than the rest of the examples, which typically only use a single object file containing the wrapped C++ function call, but is provided to show how a library-based flow can also be used.

1. `softmax.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using aiecc.py to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI).

2. `softmax_placed.py`: An alternative version of the design in softmax.py, that is expressed in a lower-level version of IRON.

3. `softmax_whole_array_placed.py`: This Python script extends the design to utilize the entire AIE array, scaling up from the use of two cores in `softmax_placed.py`. The number of cores of the AIE array (`n_cores`) is configurable via the `n_col` and `n_cores_per_col` variables.

## Usage

### C++ Testbench

To compile the design and C++ testbench:
```shell
make
```

To compile the placed design:
```shell
env use_placed=1 make
```

To compile the design on whole array:
```shell
env use_whole_array=1 make
```

To compile the design on whole array with custom columns and cores per column:
```shell
env use_whole_array=1 whole_array_cols=2 whole_array_rows=2 make
```

To run the design:
```shell
make run
```

To generate a [trace file](../../../programming_guide/section-4/section-4b/README.md):
```shell
env use_placed=1 make trace
```
