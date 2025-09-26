<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->
## Multi-Column Reduction Designs

Partial reductions are performed within each row using shared memory between neighboring tiles, followed by a final reduction step across the column. The number of columns (`num_columns`) can be configured from 1 to 8 and number of shim DMAs (`num_channels`) can be configured from 1 to 2. 

Both BF16 and INT32 data types are supported, leveraging kernels from `reduce_max.cc`.

## Source Files Overview

### Design Source Files
1. `col_wise_vector_reduce_max.py`: A Python script that defines the AIE array structural design using MLIR-AIE operations. This generates MLIR that is then compiled using `aiecc.py` to produce design binaries (ie. XCLBIN and inst.bin for the NPU in Ryzen™ AI). The Column-Limited placer is used as the placement algorithm instead of the sequential placer. 

<br><img src="assets/Multi-col.png" alt="Multi-column Design" width="1250"/>

2. `row_wise_vector_reduce_max.py`: An alternative design `col_wise_vector_reduce_max.py` using the Sequential placer instead of the Column-Limited placer. 

3. `row_wise_vector_reduce_max_placed.py`: A version of `row_wise_vector_reduce_max.py`, expressed in a lower-level version of IRON. 

<br><img src="assets/Multi-col-row-wise.png" alt="Multi-column Design" width="500"/>

## Ryzen™ AI Usage

### Compilation

To compile the design (default is the cascade design):

```shell
make
```

To compile the placed design:

```shell
env use_placed=1 make
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