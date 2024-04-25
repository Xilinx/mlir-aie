<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Machine Learning Examples</ins>

| Design name | Data type | Description | 
|-|-|-|
| [Eltwise Add](../../programming_examples/ml/eltwise_add/) | bfloat16 | An element by element addition of two vectors | 
| [Eltwise Mul](../../programming_examples/ml/eltwise_mul/) | i32 | An element by element multiplication of two vectors | 
| [ReLU](../../programming_examples/ml/relu/) | bfloat16 | Rectified linear unit (ReLU) activation function on a vector| 
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  | 
| [Conv2D](../../programming_examples/ml/conv2d) | i8 | A single core 2D convolution for CNNs |
| [Conv2D+ReLU](../../programming_examples/ml/conv2d_fused_relu) | i8 | A Conv2D with a ReLU fused at the vector register level |
|[Bottleneck](../../programming_examples/ml/bottleneck/)|ui8|A Bottleneck Residual Block is a variant of the residual block that utilizes three convolutions, using 1x1, 3x3, and 1x1 filter sizes, respectively. The implementation features fusing of multiple kernels and dataflow optimizations, highlighting the unique architectural capabilities of AI Engines|
|[ResNet](../../programming_examples/ml/resnet/)|ui8|ResNet with offloaded conv2_x layers. The implementation features depth-first implementation of multiple bottleneck blocks across multiple NPU columns.|

