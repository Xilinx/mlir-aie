<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# <ins>Machine Learning Examples</ins>

| Design name | Data type | Description |
|-|-|-|
| [Eltwise (Add / Mul)](../../programming_examples/ml/eltwise/) | bfloat16 | Element-wise addition or multiplication of two vectors (`op={add,mul}` option). |
| [Eltwise Unary (ReLU / SiLU / GELU)](../../programming_examples/ml/eltwise_unary/) | bfloat16 | Element-wise ReLU, SiLU, or GELU activation on a vector (`op={relu,silu,gelu}` option). |
| [SwiGLU](../../programming_examples/ml/swiglu/) | bfloat16 | Swish-Gated Linear Unit (SwiGLU) activation function on a vector |
| [Softmax](../../programming_examples/ml/softmax/) | bfloat16 | Softmax operation on a matrix  |
| [Norm (RMS / Layer)](../../programming_examples/ml/norm/) | bfloat16 / float32 | Row-wise RMSNorm or LayerNorm on a matrix (`op={rms,layer,layer_f32,layer_affine_cast}` option); `layer_affine_cast` fuses a real per-column affine and a `float32 -> bfloat16` cast into the same dispatch. |
| [RoPE](../../programming_examples/ml/rope/) | bfloat16 | Rotary Position Embedding on a matrix |
| [Scale Shift](../../programming_examples/ml/scale_shift/) | bfloat16 | Element-wise scale (multiply) and shift (add) on vectors |
| [Cast f32 -> bf16](../../programming_examples/ml/cast_f32_bf16/) | float32 -> bfloat16 | Row-wise element-wise narrowing cast, round-to-nearest-even. |
| [Depthwise Conv1d](../../programming_examples/ml/dwconv1d/) | bfloat16 | Depthwise (per-channel) 1D convolution, 'same' padding, stride 1, with an optional per-channel bias (`kernel_size`, `bias` options). |
| [Conv2D (optional fused ReLU)](../../programming_examples/ml/conv2d) | i8 | 1x1 Conv2D for CNNs; `fuse_relu=1` swaps the output to uint8 saturation, fusing ReLU at the vector register level. |
| [Conv2D 14x14](../../programming_examples/ml/conv2d_14x14) | i8 | A multi-core 2D convolution for 14x14 feature maps |
|[Bottleneck](../../programming_examples/ml/bottleneck/)|ui8|A Bottleneck Residual Block is a variant of the residual block that utilizes three convolutions, using 1x1, 3x3, and 1x1 filter sizes, respectively. The implementation features fusing of multiple kernels and dataflow optimizations, highlighting the unique architectural capabilities of AI Engines|
|[ResNet](../../programming_examples/ml/resnet/)|ui8|ResNet with offloaded conv2_x layers. The implementation features depth-first implementation of multiple bottleneck blocks across multiple NPU columns.|
|[Magika](../../programming_examples/ml/magika/)|bfloat16|Magika file-type detection model inference on the NPU.|
|[Block Datatypes](../../programming_examples/ml/block_datatypes/)|various|Examples demonstrating block floating point and other block datatypes on the NPU.|
