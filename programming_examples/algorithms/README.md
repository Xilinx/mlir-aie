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

| Algorithm | Description | 
|-----------|-------------|
| `transform` | Unary transformation with tiled processing |
| `transform_binary` | Binary transformation with tiled processing |
| `for_each` | In-place transformation (same tensor for input/output) |
| `transform_parallel` | Parallel `transform` across multiple AIE tiles |
| `transform_parallel_binary` | Parallel `transform_binary` across multiple AIE tiles |

## Using C++ Kernels (ExternalFunction)

For C++ kernels, the kernel signature and `ExternalFunction` must match the algorithm's expected format.

**Important:** The last argument in every kernel must be `tile_size` (`np.int32`), which receives the tile size at runtime.

| Algorithm | C++ Kernel Signature | ExternalFunction arg_types |
|-----------|---------------------|---------------------------|
| `transform` | `void kernel(T* in, T* out, params..., int32_t tile_size)` | `[tile_ty, tile_ty, *param_types, np.int32]` |
| `transform_binary` | `void kernel(T* in1, T* in2, T* out, params..., int32_t tile_size)` | `[tile_ty, tile_ty, tile_ty, *param_types, np.int32]` |
| `for_each` | `void kernel(T* in, T* out, params..., int32_t tile_size)` | `[tile_ty, tile_ty, *param_types, np.int32]` |
| `transform_parallel` | `void kernel(T* in, T* out, params..., int32_t tile_size)` | `[tile_ty, tile_ty, *param_types, np.int32]` |
| `transform_parallel_binary` | `void kernel(T* in1, T* in2, T* out, params..., int32_t tile_size)` | `[tile_ty, tile_ty, tile_ty, *param_types, np.int32]` |

### Example: transform with ExternalFunction
```python
TILE_SIZE = 16
tile_ty = np.ndarray[(TILE_SIZE,), np.dtype[np.int16]]
scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]

# Input and output should still be declared as tiled shapes in arg_types
# TODO: remove this discrepancy
my_kernel = ExternalFunction(
    "my_kernel",
    source_file="my_kernel.cc",
    arg_types=[tile_ty, tile_ty, np.int32, np.int32],  # [in, out, factor, tile_size]
)

# Pass in full size tensors, algorithm will tile it to tile_size
# tile_size must be passed as a keyword argument
iron.jit(is_placed=False)(transform)(
    my_kernel, input, output, factor, tile_size=TILE_SIZE
)
```

For a complete example, see [vector_scalar_mul_jit.py](../basic/vector_scalar_mul/vector_scalar_mul_jit.py).

## Ryzen™ AI Usage

Run and verify a design:

```shell
python3 transform.py
```
