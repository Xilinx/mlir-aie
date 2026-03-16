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
| [GeLU](../../programming_examples/ml/gelu/) | bfloat16 | Gaussian Error Linear Unit (GeLU) activation function on a vector |
| [SiLU](../../programming_examples/ml/silu/) | bfloat16 | Sigmoid Linear Unit (SiLU) activation function on a vector |
| [SwiGLU](../../programming_examples/ml/swiglu/) | bfloat16 | Swish-Gated Linear Unit (SwiGLU) activation function on a vector |
| [ReLU](../../programming_examples/ml/relu/) | bfloat16 | Rectified linear unit (ReLU) activation function on a vector|
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  |
| [LayerNorm](../../programming_examples/ml/layernorm/) | bfloat16 | Layer normalization on a matrix |
| [RMSNorm](../../programming_examples/ml/rmsnorm/) | bfloat16 | Root Mean Square layer normalization on a matrix |
| [RoPE](../../programming_examples/ml/rope/) | bfloat16 | Rotary Position Embedding on a matrix |
| [Scale Shift](../../programming_examples/ml/scale_shift/) | bfloat16 | Element-wise scale (multiply) and shift (add) on vectors |
| [Conv2D](../../programming_examples/ml/conv2d) | i8 | A single core 2D convolution for CNNs |
| [Conv2D 14x14](../../programming_examples/ml/conv2d_14x14) | i8 | A multi-core 2D convolution for 14x14 feature maps |
| [Conv2D+ReLU](../../programming_examples/ml/conv2d_fused_relu) | i8 | A Conv2D with a ReLU fused at the vector register level |
|[Bottleneck](../../programming_examples/ml/bottleneck/)|ui8|A Bottleneck Residual Block is a variant of the residual block that utilizes three convolutions, using 1x1, 3x3, and 1x1 filter sizes, respectively. The implementation features fusing of multiple kernels and dataflow optimizations, highlighting the unique architectural capabilities of AI Engines|
|[ResNet](../../programming_examples/ml/resnet/)|ui8|ResNet with offloaded conv2_x layers. The implementation features depth-first implementation of multiple bottleneck blocks across multiple NPU columns.|
|[Magika](../../programming_examples/ml/magika/)|bfloat16|Magika file-type detection model inference on the NPU.|
|[Block Datatypes](../../programming_examples/ml/block_datatypes/)|various|Examples demonstrating block floating point and other block datatypes on the NPU.|

