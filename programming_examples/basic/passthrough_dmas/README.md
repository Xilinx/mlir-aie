<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Passthrough DMAs</ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./passthrough_dmas.py) data is brought from external memory to a compute tile and back, without modification from the tile, by using an implicit copy via the compute tile's Direct Memory Access (DMA). The data is read from and written to external memory through a shim tile.

The implicit copy is performed using the ObjectFifo `forward()` function that specifies how input data arriving via `of_in` should be sent further via `of_out` by leveraging the fowarding tile's DMA. 

The single [passthrough_dmas.py](./passthrough_dmas.py) design uses `@iron.jit` and runs on both NPU and VCK5000 (the latter via the legacy print-MLIR + `aiecc` flow).

To compile and run the design for NPU:
```shell
make
make run
```

To run the standalone Python JIT + verify directly (no Makefile, no C++ testbench):
```shell
python3 passthrough_dmas.py
```

To target VCK5000:
```shell
make vck5000
make run_vck5000
```