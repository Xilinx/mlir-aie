<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
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


## Design Overview

Softmax is computed independently per 1024-element tile (no cross-tile reduction), so the design scales like `ml/eltwise_unary` — the body delegates to `iron.algorithms.transform_parallel_typed` with `num_channels=2`. The per-tile kernel comes from `aie.iron.kernels.softmax`, which uses a lookup-table approximation of `e^x` (similar to [`basic/vector_exp`](../../basic/vector_exp/)).


## Source Files Overview

1. `softmax.py`: IRON design driven by `@iron.jit`; one `transform_parallel_typed` call over `kernels.softmax(tile_size=1024)`.

1. `test.cpp`: C++ testbench that loads the compiled XCLBIN, runs the kernel, and verifies the output against a CPU reference.


## Usage

### Build and Run

Build and run with default settings (npu1):
```shell
make run
```

Target Strix (npu2):
```shell
make run devicename=npu2
```

Build and run with custom runtime parameters:
```shell
make run size=524288 n_iterations=100 n_warmup=20
```

### Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `size` | 262144 | Input data size (number of elements) |
| `chans` | 2 | Shim DMA channels per column (1 or 2) |
| `n_iterations` | 20 | Number of benchmark iterations |
| `n_warmup` | 10 | Number of warmup iterations |
| `devicename` | npu | Target device (`npu` or `npu2`) |
