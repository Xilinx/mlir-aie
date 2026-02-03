<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Algorithms

Pre-built algorithms for common dataflow patterns on AIE. These abstractions handle Workers, ObjectFifos, and data movement automatically. The examples shown leverage IRON JIT and passes simple lambda functions to the algorithm. The algorithms can be furthered used with C++ kernels.

## Usage with C++ Kernels
[TODO] Generalize kernel usage

To use with C++ kernels, they are expected to have the form:
```cpp
void my_kernel(int32_t *input, int32_t *output, [extra_args...], int32_t tile_size)
```
and its equivalent IRON ExternalFunction handle:
```python
my_kernel_func = ExternalFunction(
    "my_kernel",
    arg_types=[tile_ty, tile_ty, [extra_arg_types...], np.int32],  # [input, output, extra_args..., tile_size]
    source="my_kernel.cc"
)
```
`tile_size` is automatically provided by the algorithm and additional kernel arguments should be passed(if any):
```python
iron.jit(is_placed=False)(transform)(input, output, my_kernel_func, extra_args...)
```

For a complete example, see [vector_scalar_mul_jit.py](../basic/vector_scalar_mul/vector_scalar_mul_jit.py).

## Available Algorithms

| Algorithm | Description | 
|-----------|-------------| 
| `transform` | Apply a unary function to each element (1 input → 1 output) | 
| `transform_binary` | Apply a binary function to pairs of elements (2 inputs → 1 output) | 
| `transform_parallel` | Parallel `transform` across multiple AIE tiles |
| `transform_parallel_binary` | Parallel `transform_binary` across multiple AIE tiles | 
| `for_each` | Apply a function to each element in-place |

## Ryzen™ AI Usage

Run and verify a design:

```shell
python3 transform.py
```
