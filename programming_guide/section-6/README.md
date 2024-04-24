<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 6 - Larger Example Designs</ins>

There are a number of example designs available [here](../../programming_examples/) which further help explain many of the unique features of AI Engines and the NPU array in Ryzen™ AI. This section contains more complex application designs for both vision and machine learning use cases. In particular we will describe a ResNet implementation on for Ryzen™ AI.

## Vision Kernels

| Design name | Data type | Description | 
|-|-|-|
| [Vision Passthrough](../../programming_examples/vision/vision_passthrough/) | i8 | A simple pipeline with just one `passThrough` kernel. This pipeline's main purpose is to test whether the data movement works correctly to copy a greyscale image. | 
| [Color Detect](../../programming_examples/vision/color_detect/) | i32 | This multi-kernel, multi-core pipeline detects colors in an RGBA image.  | 
| [Edge Detect](../../programming_examples/vision/edge_detect/) | i32 | A mult-kernel, multi-core pipeline that detects edges in an image and overlays the detection on the original image. | 
| [Color Threshold](../../programming_examples/vision/color_threshold/) | i32 | A mult-core data-parallel implementation of color thresholding of a RGBA image. | 


## Machine Learning Designs

| Design name | Data type | Description | 
|-|-|-|
|[bottleneck](../../programming_examples/ml/bottleneck/)|ui8|A Bottleneck Residual Block is a variant of the residual block that utilises three convolutions, using 1x1, 3x3 and 1x1 filter sizes, respectively. The use of a bottleneck reduces the number of parameters and computations.|
|[resnet](../../programming_examples/ml/resnet/)|ui8|ResNet with offloaded conv2_x bottleneck blocks. The implementation features kernel fusion and dataflow optimizations highlighting the unique architectural capabilties of AI Engines.|

## Exercises

1. In [bottleneck](../../programming_examples/ml/bottleneck/) design following a dataflow approach, how many elements does the 3x3 convolution operation require to proceed with its computation? <img src="../../mlir_tutorials/images/answer1.jpg" title="3. This allows for the necessary neighborhood information required by the convolutional kernel to be available for processing." height=25>
2. Suppose you have a bottleneck block with input dimensions of 32x32x256. After passing through the 1x1 convolutional layer, the output dimensions become 32x32x64. What would be the output dimensions after the subsequent 3x3 convolutional layer, assuming a stride of 1 and no padding and output channel of 64? <img src="../../mlir_tutorials/images/answer1.jpg" title="30×30×64. Without padding, the spatial dimensions would shrink by two pixels in each dimension due to the 3x3 convolution operation." height=25>

-----
[[Prev - Section 5](../section-5/)] [[Top](..)]

