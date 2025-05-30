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

This folder contains three extended designs where reduction is performed across an entire column of AIE cores:

1. **vector_reduce_max_cascade**: Implements a cascading reduction where intermediate results are passed between adjacent tiles in the column.

2. **vector_reduce_max_shared**: Utilizes shared memory between neighboring tiles to perform the final reduction.

3. **vector_reduce_max_memtile**: Leverages memory tiles to aggregate partial results from the column, which is then sent to one of the AIE cores for the final reduction step.

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
