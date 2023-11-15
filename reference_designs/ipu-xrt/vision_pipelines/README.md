<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Vision Pipelines</ins>

The vision pipeline reference designs show how to complex vision pipelines can be constructed from basic vision kernel building blocks. Those building blocks can be found in [./vision_kernels](./vision_kernels) and contain example kernels written for AI engines in both scalar and unoptimized vector format. 

## <ins>Edge Detect</ins>

The [Edge Detect pipeline design](./edge_detect/) consists of the following blocks arranged in a pipeline fashion for the detection of edges in a sequence of images (rgba2gray, filter2D, threshold, addWeighted, gray2rgba).