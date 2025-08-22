<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Compute Repeat</ins>

This reference design can be run on a Ryzen™ AI NPU.

In the [design](./aie2.py) data is brought from external memory via the `ShimTile` to the `ComputeTile` and back. Furthermore, the input data is repeated by the `ComputeTile` four times which results in the output data consisting of four instances of the input data.

The repeat count is specified as follows:
```python
of_out.set_repeat_count(repeat_count)
```
Specifically, the instruction above specifies the number of repetitions that the producer side of the `of_out` objectfifo should do.
