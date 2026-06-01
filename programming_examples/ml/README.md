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
| [Eltwise (Add / Mul)](../../programming_examples/ml/eltwise/) | bfloat16 | Element-wise addition or multiplication of two vectors (`op={add,mul}` knob). |
| [Eltwise Unary (ReLU / SiLU / GELU)](../../programming_examples/ml/eltwise_unary/) | bfloat16 | Element-wise ReLU, SiLU, or GELU activation on a vector (`op={relu,silu,gelu}` knob). |
| [SwiGLU](../../programming_examples/ml/swiglu/) | bfloat16 | Swish-Gated Linear Unit (SwiGLU) activation function on a vector |
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  |
| [Norm (RMS / Layer)](../../programming_examples/ml/norm/) | bfloat16 | Row-wise RMSNorm or LayerNorm on a matrix (`op={rms,layer}` knob). |
| [RoPE](../../programming_examples/ml/rope/) | bfloat16 | Rotary Position Embedding on a matrix |
| [Scale Shift](../../programming_examples/ml/scale_shift/) | bfloat16 | Element-wise scale (multiply) and shift (add) on vectors |
| [Conv2D (optional fused ReLU)](../../programming_examples/ml/conv2d) | i8 | 1x1 Conv2D for CNNs; `fuse_relu=1` swaps the output to uint8 saturation, fusing ReLU at the vector register level. |
| [Conv2D 14x14](../../programming_examples/ml/conv2d_14x14) | i8 | A multi-core 2D convolution for 14x14 feature maps |
|[Bottleneck](../../programming_examples/ml/bottleneck/)|ui8|A Bottleneck Residual Block is a variant of the residual block that utilizes three convolutions, using 1x1, 3x3, and 1x1 filter sizes, respectively. The implementation features fusing of multiple kernels and dataflow optimizations, highlighting the unique architectural capabilities of AI Engines|
|[ResNet](../../programming_examples/ml/resnet/)|ui8|ResNet with offloaded conv2_x layers. The implementation features depth-first implementation of multiple bottleneck blocks across multiple NPU columns.|
|[Magika](../../programming_examples/ml/magika/)|bfloat16|Magika file-type detection model inference on the NPU.|
|[Block Datatypes](../../programming_examples/ml/block_datatypes/)|various|Examples demonstrating block floating point and other block datatypes on the NPU.|

