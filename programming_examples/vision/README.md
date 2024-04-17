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

The vision pipeline reference designs show how complex vision pipelines can be constructed from basic vision kernel building blocks. Those building blocks can be found in [aie_kernels/aie2](../../aie_kernels/aie2) and contain example kernels written for AI engines in both scalar and unoptimized vector format. 

## <ins>[Color Detect](./color_detect/)</ins>

The [Color Detect pipeline design](./color_detect/) consists of the following blocks arranged in a pipeline fashion for the detecting of 2 colors in a sequence of images : `rgba2hue`, `threshold`, `threshold`, `bitwiseOR`, `gray2rgba`, `bitwiseAND`.

## <ins>[Edge Detect](./edge_detect/)</ins>

The [Edge Detect pipeline design](./edge_detect/) consists of the following blocks arranged in a pipeline fashion for the detection of edges in a sequence of images: `rgba2gray`, `filter2D`, `threshold`, `gray2rgba`, `addWeighted`.

## <ins>[Color Threshold](./color_threshold/)</ins>

The [Color Threshold pipeline design](./color_threshold/) consists of 4 threshold blocks in separate tiles that process a different region of an input image. The results are then merged back together and sent to the output.