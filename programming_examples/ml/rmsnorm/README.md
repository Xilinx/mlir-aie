<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# RMSNorm

This design implements a `bfloat16` based Root Mean Square Normalization (RMSNorm) technique commonly used in machine learning, especially in transformer models. It normalizes each row of the input matrix by its root mean square (RMS) value. The computation is parallelized across multiple AIE cores, with each core processing a block of rows.

## Source Files Overview

1. `rmsnorm.py`: Python script to set up and run the RMSNorm kernel on the AIE device using IRON operations. This script also includes a `TensorAccessPattern` visualizer to display the column distribution across AIE cores, illustrating how the workload is partitioned. It generates MLIR, which is then compiled using `aiecc.py` to produce design binaries (e.g., XCLBIN and inst.bin for the NPU).

2. `rms_norm.cc`: A C++ implementation of a RMSNormfor AIE cores. The code uses the AIE API, which is a C++ header-only library providing types and operations that get translated into efficient low-level intrinsics.  The source can be found [here](../../../aie_kernels/aie2p/rms_norm.cc).

3. `test.cpp`: This C++ code is a testbench for the design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and executing the AIE design on the NPU. After executing, the script verifies the memcpy results and optionally outputs trace data.

## Usage

### Python
To generate the MLIR for the design, run:

```shell
python rmsnorm.py -d <device> -r <rows> -c <cols> [-t <trace_size>]
```

#### Arguments
- `-d`, `--dev`: AIE device to use (`npu` or `npu2`)
- `-r`, `--rows`: Number of rows in the input matrix
- `-c`, `--cols`: Number of columns in the input matrix
- `-t`, `--trace_size`: (Optional) Trace buffer size (default: 0)

#### Example
```shell
python rmsnorm.py -d npu -r 128 -c 128
```

### C++ Testbench
To compile the design and C++ testbench:
```shell
make
```

To run the design:
```shell
make run
```

## Notes
- The input and output matrices are stored in row-major order.
- The computation is parallelized across multiple AIE cores, with each core processing a block of rows.
- On AIE2 architecture, the `rms_norm.cc` kernel requires a custom inverse square root (invsqrt) function, as native hardware support for sqrt is not available.