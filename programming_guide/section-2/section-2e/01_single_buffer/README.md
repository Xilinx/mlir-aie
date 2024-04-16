<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Single Buffer</ins>

The design in [single_buffer.py](./single_buffer.py) uses an Object FIFO to transfer data from producer tile `ComputeTile2` to consumer tile `ComputeTile3`. The Object FIFO has a depth of `1` which describes a single buffer between the two tiles as shown in the figure below.

TODO: add drawing

Both the producer and the consumer processes in this design have trivial tasks. The producer process running on `ComputeTile2` acquires the single buffer and writes `1` into all its entries before releasing it for consumption. The producer process running on `ComputeTile3` acquires the single buffer and immediately releases it back for the producer.
