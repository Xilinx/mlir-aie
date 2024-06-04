<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Edge Detect</ins>

The Edge Detect pipeline design consists of the following blocks arranged in a pipeline fashion for the detection of edges in a sequence of images : `rgba2gray`, `filter2D`, `threshold`, `gray2rgba`, `addWeighted`.

The pipeline is mapped onto a single column of the npu device, with one Shim tile (0, 0), one Mem tile (0, 1) and four AIE compute tiles (0, 2) through (0, 5). As shown in the image below, the `rgba2gray`, `filter2D` and `threshold` kernels are each mapped onto one compute tile, while `gray2rgba` and `addWeighted` are mapped together on AIE tile (0, 5). 

<p align="center">
  <img
    src="./edge_detect_pipeline.png"
    width="1050">
</p>

The data movement of this pipeline is described using the OrderedObjectBuffer (OOB) primitive. Input data is brought into the array via the Shim tile. The data then needs to be broadcasted both to AIE tile (0, 2) and AIE tile (0, 5). However, tile (0, 5) has to wait for additional data from the other kernels before it can proceed with its execution, so in order to avoid any stalls in the broadcast, data for tile (0, 5) is instead buffered in the Mem tile. Because of the size of the data, the buffering couldn't directly be done in the smaller L1 memory module of tile (0, 5). This is described using two OOBs, one for the broadcast to tile (0, 2) and the Mem tile, and one for the data movement between the Mem tile and tile (0, 5). The two OOBs are linked to express that data from the first OOB should be copied to the second OOB implicitly through the Mem tile's DMA.

Starting from tile (0, 2) data is processed by each compute tile and the result is sent to the next tile. This is described by a series of one-to-one OOBs. As the two kernels `gray2rgba` and `addWeighted` are mapped together on AIE tile (0, 5), an OOB is also created with tile (0, 5) being both its source and destination to describe the data movement between the two kernels. Finally, the output is sent from tile (0, 5) to the Mem tile and finally back to the output through the Shim tile.

To compile desing in Windows:
```
make
make edgeDetect.exe
```

To run the design:
```
make run
```
