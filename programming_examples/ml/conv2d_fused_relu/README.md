<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Convolution with Fused ReLU</ins>

## Introduction
Convolution is a crucial part of various machine learning and computer vision tasks, such as image recognition, object detection, and image segmentation.  ReLU (Rectified Linear Unit ) is one of the most commonly used activation functions due to its simplicity and effectiveness. This README provides instructions for implementing convolution with ReLU activation function on AI Engine. 


At its core, convolution is a mathematical operation that combines an input image and a filter to produce an output image. The input data is represented as a multi-dimensional matrix, such as an image with height, width, and channels (e.g., RGB channels). The filter is also represented as a multi-dimensional matrix with filter height, width, input and output channels (the same number of channels as the input data). The filter is systematically applied to different regions of the input data. At each step, the filter is element-wise multiplied with the overlapping region of the input data. The element-wise products are summed up to produce a single value, which represents the result of the convolution operation for that region. This process is repeated for all possible regions of the input data, producing an output matrix called the feature map.

The process of applying the filter to different regions of the input data is often visualized as a sliding window moving across the input data. The size of the sliding window corresponds to the size of the filter, and it moves with a certain stride (the number of pixels it moves at each step). The convolution operation consists of seven nested loops, iterating over the input height, input lenght, input channel, output channel, filter height, filter length, and the batch size, each loop corresponding to different aspect of the operation. This systematic process extracts features from the input image, yielding the output feature map, illustrating the computational intricacies of convolution. 

## Acceleration Techniques
1. Kernel Optimzation: To optimize convolution operations on AIE, we vectorize the code using AIE vector intrinsics. We load 8 elements of the input channel into vector registers using vector load intrinsic. We apply the convolution operation on this loaded data, utilizing for enhanced computational efficiency. To ensure accurate convolution results, particularly at the edges of feature maps, we implement zero-padding to handle boundary conditions. This comprehensive approach optimizes convolution processing on AIE, facilitating efficient and accurate feature extraction in neural network applications. Input is 4x8 matrix corresponding to 4 element of row and 8 input channels.

2. Quantization: We use int8 precision for activationa and weights. At int8 precision, AIE offers the highest compute density with 256 MAC/cycle.  

3. Layer Fused: We perform two levels of fusion. First, we fuse ReLU in convolution using SRS capabilities of AIE. Second, we fuse BatchNorm into convolution weights. 

4. Data Layout: Optimize activation and weight layout to enhance memory access patterns and enables effective utilization of AIE parallel processing units, ultimately improving the performance of 2D convolution operations. 

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


### Benefits of ReLU Fusion:

1. Reduced Memory Bandwidth:
By fusing ReLU into the convolution operation, unnecessary memory accesses and data transfers associated with separate ReLU computation are eliminated, leading to reduced memory bandwidth requirements.

2. Improved Performance:
Fusing ReLU reduces the number of instructions executed per element, resulting in improved computational efficiency and overall performance of the convolution operation.

3. Simplified Code Structure:
Fusing ReLU into the convolution kernel simplifies the code structure and reduces the overhead associated with separate activation function calls, leading to cleaner and more maintainable code.

4. Enhanced Resource Utilization:
By combining convolution and ReLU operations, computational resources such as CPU cores or SIMD units are utilized more efficiently, maximizing throughput and achieving better resource utilization.

## Compilation
To compile the design:
```
make
```

To run the design:
```
make run_py
```