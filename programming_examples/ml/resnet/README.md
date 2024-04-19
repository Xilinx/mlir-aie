<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>ResNet with Offloaded Conv2_x Bottleneck Blocks</ins>

## Introduction
ResNet [[1]](#1) is a convolutional neural network architecture that has gained significant popularity for various computer vision tasks, including image classification, object detection, and image segmentation. It is renowned for its depth and efficiency in training very deep networks.

This README focuses on a specific optimization technique applied to ResNet, specifically targeting the offloading of the conv2_x part of the bottleneck blocks. By offloading computations to dedicated hardware accelerators or specialized processors, we aim to improve the overall efficiency and speed of the network, especially when deploying it on resource-constrained devices or in scenarios where real-time processing is critical.


## ResNet Architecture Overview
ResNet consists of several key components:

1. Input Layer: Accepts input image data with dimensions typically set to 224x224x3 (width, height, RGB channels).
2. Convolutional Layers: The initial layers perform convolution operations to extract basic features from the input image.
3. Bottleneck Blocks:
    * ResNet is composed of multiple bottleneck blocks grouped into different stages (conv2_x, conv3_x, conv4_x, conv5_x).
    * Each bottleneck block contains convolutional layers and shortcut connections that facilitate the learning of residual mappings.
    * The conv2_x stage is particularly targeted for offloading computations in this optimization.
4. Pooling Layers: Max pooling layers reduce the spatial dimensions of the feature maps.
5. Fully Connected Layer: Produces the final output predictions, typically followed by a softmax activation for classification tasks.


## Offloading Conv2_x Bottleneck Blocks
The conv2_x stage of ResNet comprises a series of bottleneck blocks, each containing convolutional layers responsible for learning more complex features from the input data. By offloading the computations within these blocks to AI Engine, we aim to:

* Reduce the computational burden on the main processing unit (e.g., CPU or GPU).
* Improve overall inference speed and efficiency, especially in scenarios where real-time processing is crucial.
* Enable deployment on resource-constrained devices with limited computational resources.

##  Usage and Deployment
To leverage the optimized ResNet with offloaded conv2_x bottleneck blocks:
* [IRON Programming](https://github.com/Xilinx/mlir-aie/tree/gagan_asplos_resnet/programming_examples/ml/resnet/layers_conv2_x): Demonstrates the IRON flow for offloading conv2_x to AIE.


## Acceleration Techniques
1. Depth-First/Layer-Fused Implementation: Spatial architectures provide coarse-grained flexibility that allows for tailoring of the dataflow to optimize data movement. By tailoring the dataflow, we implement depth-first schedule for a bottleneck block  routing the output of one convolutional operation on an AIE core directly to another convolutional operation on a separate AIE core, all without the need to transfer intermediate results off-chip. This approach effectively minimizes the memory footprint associated with intermediate data, mitigating the overhead of costly off-chip accesses leading to increase in the overall performance.


2. Data Layout: Optimize activation and weight layout to enhance memory access patterns and enables effective utilization of AIE parallel processing units, ultimately improving the performance of 2D convolution operations. 

3. Kernel Optimzation: To optimize convolution operations on AIE, we vectorize the code using AIE vector intrinsics. We load 8 elements of the input channel into vector registers using vector load intrinsic. We apply the convolution operation on this loaded data, utilizing for enhanced computational efficiency. To ensure accurate convolution results, particularly at the edges of feature maps, we implement zero-padding to handle boundary conditions. This comprehensive approach optimizes convolution processing on AIE, facilitating efficient and accurate feature extraction in neural network applications. Input is 4x8 matrix corresponding to 4 element of row and 8 input channels.

4. Quantization: We use int8 precision for activationa and weights. At int8 precision, AIE offers the highest compute density with 256 MAC/cycle.  

5. Layer Fused: We perform two levels of fusion. First, we fuse ReLU in convolution using SRS capabilities of AIE. Second, we fuse BatchNorm into convolution weights. 


## Data Layout
We need to ensure that the data layout is compatible with efficient SIMD processing and rearrange the input data into a format where contiguous elements represent consecutive X-dimension values for each channel. For more efficient processing, we adopt a channels-last memory ordering, denoted as NYCXC8, to ensure that channels become the densest dimension. Operating on 8 elements simultaneously, we process 8 channels with the same width at once. Subsequently, we traverse the entire width dimension, handling the remaining channels in batches of 8. This process continues row-wise, resulting in our final data layout pattern: NYCXC8. This optimized layout enhances memory access patterns and enables effective utilization of parallel processing units, ultimately improving the performance of 2D convolution operations. This transformation ensures that data can be efficiently loaded into SIMD registers and processed in parallel. 

YCXC8 Input/Output Data Layout:

In the YCXC8 (with N=1) data layout, the data is organized in memory as follows:

* Y: Represents the output feature map dimension.
* C: Denotes the number of channels.
* X: Represents the input feature map dimension.
* C8: Indicates that 8 elements of the input channel are processed together.

OIYXI8O8 Weight Layout:

We align the weight layout as specified: O,I,Y,X,I8,O8, to match the input image processing. We first load the weight tensor, organizing it to match this layout, where dimensions represent: output channels, input channels, kernel height, kernel width, input channel groups of 8, and output channel groups of 8. By aligning the weight layout in this manner, we enable seamless integration with the input data layout, maximizing parallelism and minimizing memory access overhead. 

In the OIYXI8O8 data layout, the data is organized in memory as follows:

* O: Denotes the number of output channels.
* I: Denotes the number of input channels.
* Y: Represents the kernel height.
* X: Represents the kernel weight.
* I8: Indicates that 8 elements of the input channel are processed together.
* O8: Indicates that 8 elements of the output channel are processed together.

## Fusing Convolution and Batch Normalization

We assume the BatchNorm layer is fused into Convoluion Layer. Fusing BatchNorm into convolution involves incorporating the normalization step directly into the convolution operation. This is achieved by modifying the weights of the convolutional filters to include the scaling and shifting factors. Specifically, the weights are adjusted such that the convolution operation performs the normalization, scaling, and shifting in a single step.

## Fusing ReLU

Fusing ReLU into the convolution operation can further optimize the implementation by reducing memory bandwidth requirements and computational overhead. ReLU activation function introduces non-linearity by setting negative values to zero and leaving positive values unchanged. Utilize SIMD instructions to efficiently compute ReLU activation in parallel with convolution. After performing the convolution operation, apply ReLU activation function at vector register level. 
We use `aie::set_rounding()` and `aie::set_saturation()` to set the rounding and saturation modes for the computed results in the accumulator. Seeting round mode `postitive_inf` rounds halfway towards positive infinity while setting saturation to `aie::saturation_mode::saturate` saturation rounds an uint8 range (0, 255). 

```
::aie::set_saturation(
      aie::saturation_mode::saturate); // Needed to saturate properly to uint8
::aie::set_rounding(
      aie::rounding_mode::positive_inf); // Needed to saturate properly to uint8
```
After convolution and ReLU fusion, the output data is generate in YCXC8 layout. Ensure that the output data layout is compatible with subsequent layers or processing steps in the neural network architecture.

## Compilation
To compile the design:
```
make
```

To run the design:
```
make run_py
```

### Prerequisites

To install the dependencies, run the following command:
```
pip install -r requirements.txt

```

## References
<a id="1">[1]</a> 
He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).

