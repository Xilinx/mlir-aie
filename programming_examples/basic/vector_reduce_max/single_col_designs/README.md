<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

## Column-Wide Reduction Designs

This folder contains three designs for performing column-wide reductions across AIE cores. These designs support both BF16 and INT32 data types and utilize the kernels from `reduce_max.cc`.

## Source Files Overview

1. `vector_reduce_max_cascade.py`: Implements a cascading reduction where intermediate results are passed between adjacent tiles in the column.

2. `vector_reduce_max_shared.py`: Utilizes shared memory between neighboring tiles to perform the final reduction.

3. `vector_reduce_max_memtile.py`: Leverages memory tiles to aggregate partial results from the column, which is then sent to one of the AIE cores for the final reduction step.

## Ryzen™ AI Usage

### Compilation

To compile the design (default is the cascade design):

```shell
make
```

To compile the shared memory-based design:

```shell
env use_shared=1 make
```

To compile the memory tile-based design:

```shell
env use_memtile=1 make
```

To compile the C++ testbench:

```shell
make vector_reduce_max.exe
```
### C++ Testbench

To run the design:

```shell
make run
```

### Trace

To generate a [trace file](../../../programming_guide/section-4/section-4b/README.md) for the default cascade design:

```shell
make trace
```

To generate a trace file for the shared memory-based design:

```shell
env use_shared=1 make trace
```

To generate a trace file for the memory tile-based design:

```shell
env use_memtile=1 make trace
```