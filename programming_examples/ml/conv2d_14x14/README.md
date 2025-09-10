<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Convolution 2D (14x14)</ins>
## Introduction
This presents an optimized implementation of convolution 2D with a kernel size of 14x14, stride of 14 and borders of 0. This block is ideally suited to such uses as the image tokenizer in Gemma3.

This optimized design is currently targeting a single AIE core and uses memtile and coretile DMA to perform data layout transformations. All datatypes are all `int8` except for the input/activations which are `uint8`. Intermediate sub-kernel accumulator are `int32`.

## Data Layout and Sizes
The data layout at each stage of the design is as follows:

### <u>Sub-kernel ([conv2dk14.cc](../../../aie_kernels/aie2p/conv2dk14.cc))</u>
For each vector multiply (vmul) on a strix device for uint8/int8 datatypes, we perform a 8x8x8 matrix multiplcation. The format of the data for each vmul is as follows:
* Inputs/Activations - {T8}{P2}
* Weights - {P2}{C8}
* Outputs - {T8}{C8}

Defintions
* P2 - 2 pixels consisting of rgba. So that would be {r0 b0 g0 a0} and {r1 b1 g1 a1}
* T8 - 8 tiles. Tiles are a sequential notation we're using to number each of the 14x14 pixel blocks we're interating over. So in our 896 x 896 pixel image, we have 64 x 64 tiles. The first row of tiles are then indexed as t0 .. t63. The first tile in the second row is then t64, etc.
* C8 - 8 channels. This corresponds to output channels. We do techncially have 4 input channels but we're grouping all 4 of them into a single pixel in this notation

Then, within our sub-kernel ([conv2dk14.cc](../../../aie_kernels/aie2p/conv2dk14.cc)) we loop over the following inputs, weights and outputs:
* Inputs - {T/8}{P/P2}{T8}{P2} - 12,544 bytes
* Weights - {C/8}{P/P2}{P2}{C8} - 12,544 bytes
* Outputs - {C/8}{T/8}{T8}{C8} - 256 bytes

Defintions
* P/P2 - Total pixels divided by 2 or the remaining pixels. In this design, we have a kernel size of 14x14 or 196 pixels. Since the lowest dimenioni is every 2 pixels, on the outer dimension, we iterate of 196/2 or 98
* T/8 - Remaining tiles. Our sub-kernel operates on 16 tiles so T/8 = 16/8 = 2.
* C/8 - Remaining channels. Our sub-kernel operates on 16 channels on C/8 = 2.

### <u>Kernel ([conv2dk14_placed.py](./conv2dk14_placed.py))/</u>
Our kernel then loops over a set of inputs and weights while calling our sub-kernel to process 16 channels, 16 tiles and 196 pixels (14x14x4). We loop over the tile row of our image which has 64 tiles, giving us a loop size of 4 (x_blocks) since we process 16 tiles at a time. Then we loop over the tile rows of our image which is a loop size of 64. That in turn is inside a infinite loop which allows us to compute as many output channels as needed. Given that we compute 16 output channels each iteration of the kernel body, we would iterate 72 times (1152/ 16) to compute the results for all output chanenls. 

### <u>Memtiles and Top-level</u>
We use the memtile primarily to buffer DDR reads but also to leverage the layout transformation of the memtile DMA.
* Inputs - an entire tile row or 4 x 12,544 bytes = 50,176 bytes. For the inputs, we assume the data is arranged as YXC where each pixel has 4 channels or rgba, then ordered by image width (896) and image height (896). We use the 2 levels of DMA layout transformation to arrange the data into the {T/8}{P/P2}{T8}{P2} format used by the kernel
* Weights - Not stored in memtile as weights are assumed to arranged in the correct layout format which is {C/8}{P/P2}{P2}{C8}
* Outputs - Full output size (64x64) for 16 channels or 65,536 bytes. The output has a partial layout transformation in that it transforms it into {C/16}YX{C16}. This means the lowest dimenion of C16 is 16 channels. In the case of the memtile, that's all we store so it's only YX{C16}. But since we continue to push data via the inputs and weights, we continue to compute results for additional channels, the output buffer format in DDR is currently {C/16}YX{C16}. We do a transformation in our testbench ([test.py](./test.py)) to get this back to CYX to compare it to the pytorch golden data. 

## Source Files Overview

```
.
+-- conv2dk14_placed.py        # A Python script that defines the AIE array structural design using MLIR-AIE operations using a lower-level version of IRON
+-- Makefile             # Contains instructions for building and compiling software projects.
+-- README.md            # This file.
+-- run_strix_makefile.lit # For LLVM Integrated Tester (LIT) of the placed design.
+-- test.py              # Python code testbench for the design example.
```

## Compilation
To compile and run the design:
```shell
make run_py
```


To build and run the design while generating trace
```shell
make trace_py
```

## Configure design
While the design was designed to be somewhat configurable, this is mostly tested in the output channel dimension as long as it's a multiple of 16. To configure the parameters of the convolution such as data width, height and the number of input and output channels, you can edit the top of the `Makefile` but this is largely untested. Choosing the scalar or vectorized version of the kernel can likewise be selected in the `Makefile` by modifying the `vectorized` variable but see limitations below on this feature.

## Limitation Notes
At the moment, the following limtations exist:
* The scalar kernel version of this design has some intermittent runtime issue (CMD_ABORT triggered) for the full output channel size. Reducing this to 256 channels from 1152 is a workaround at the moment but further investigation is needed to fully resolve this.
* Unplaced IRON version is in the works. At the moment, writing trace data to the 5th buffer which is the default for unplaced IRON seems to trigger a segfault. Further investgation needed.


