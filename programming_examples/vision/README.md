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


| Design name | Data type | Description | 
|-|-|-|
| [Vision Passthrough](../../programming_examples/vision/vision_passthrough/) | i8 | A simple pipeline with just one `passThrough` kernel. This pipeline mainly aims to test whether the data movement works correctly to copy a greyscale image. | 
| [Color Detect](../../programming_examples/vision/color_detect/) | i32 | This multi-kernel, multi-core pipeline detects colors in an RGBA image. The design consists of the following blocks arranged in a pipeline fashion for the detecting of 2 colors in a sequence of images : `rgba2hue`, `threshold`, `threshold`, `bitwiseOR`, `gray2rgba`, `bitwiseAND`.| 
| [Edge Detect](../../programming_examples/vision/edge_detect/) | i32 | A multi-kernel, multi-core pipeline that detects edges in an image and overlays the detection on the original image. The design consists of the following blocks arranged in a pipeline fashion for the detection of edges in a sequence of images: `rgba2gray`, `filter2D`, `threshold`, `gray2rgba`, `addWeighted`.| 
| [Color Threshold](../../programming_examples/vision/color_threshold/) | i32 | A multi-core data-parallel implementation of color thresholding of a RGBA image. The design consists of 4 threshold blocks in separate tiles that process a different region of an input image. The results are then merged back together and sent to the output.| 
