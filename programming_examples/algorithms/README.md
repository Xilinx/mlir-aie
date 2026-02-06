<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Algorithms

Pre-built algorithms for common dataflow patterns on AIE. These abstractions handle Workers, ObjectFifos, and data movement automatically. 

## Available Algorithms

| Algorithm | Description | ExternalFunction Support |
|-----------|-------------|--------------------------|
| `transform` | Unary transformation with tiled processing | Yes |
| `transform_binary` | Binary transformation with tiled processing | Yes |
| `for_each` | In-place transformation (same tensor for input/output) | Yes |
| `transform_parallel` | Parallel `transform` across multiple AIE tiles | No |
| `transform_parallel_binary` | Parallel `transform_binary` across multiple AIE tiles |  No |

## Using C++ Kernels (ExternalFunction)

For C++ kernels, the kernel signature and `ExternalFunction` must match the algorithm's expected format.

| Algorithm | C++ Kernel Signature | ExternalFunction arg_types |
|-----------|---------------------|---------------------------|
| `transform` | `void kernel(T* in, T* out, params...)` | `[tile_ty, tile_ty, *param_types]` |
| `transform_binary` | `void kernel(T* in1, T* in2, T* out, params...)` | `[tile_ty, tile_ty, tile_ty, *param_types]` |
| `for_each` | `void kernel(T* in, T* out, params...)` | `[tile_ty, tile_ty, *param_types]` |

### Example: transform with ExternalFunction
```python
tile_ty = np.ndarray[(16,), np.dtype[np.int16]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

my_kernel = ExternalFunction(
    "my_kernel",
    source_file="my_kernel.cc",
    arg_types=[tile_ty, tile_ty, np.int32],  # [in, out, param_0]
)

# Pass in full size tensors, algorithm will tile it to tile_ty
iron.jit(is_placed=False)(transform)(
    my_kernel, input, output, factor, np.int32
)
```

For a complete example, see [vector_scalar_mul_jit.py](../basic/vector_scalar_mul/vector_scalar_mul_jit.py).

## Ryzen™ AI Usage

Run and verify a design:

```shell
python3 transform.py
```
