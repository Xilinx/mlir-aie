<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 4 - Communication (switchbox)</ins>

In the [flows](../flow) portion of `Tutorial-4`, we covered the topics of configuring switchbox connection via the `AIE.flow` operator as well as the specification of tileDMAs within the `AIE.mem` blocks. Flows is a logical abstraction to automate the route and physical switchbox connections between two endpoints, just like objectFifos can automate the the route along with the DMA configuration. But if more fine grained control is needed for the hardware switchboxes is needed, we can use `AIE.switchbox` to configure individual switchboxes. These connection use the `AIE.connect` operator to connect input and output ports and channels for a switchbox as shown below:

```
AIE.switchbox(%tile14) { AIE.connect<"DMA": 0, "East": 1> }
AIE.switchbox(%tile24) { AIE.connect<"West": 1, "East": 3> }
AIE.switchbox(%tile34) { AIE.connect<"West": 3, "DMA": 1> }
```
Multiple `AIE.connect` operators could be defined in a single switchbox and is only limited by the number of connections available in hardware. 

## <ins>Tutorial 4 Lab</ins>
1. Take a look at the [aie.mlir](./aie.mlir) to see how the previous `AIE.flow` operator was replaced with 3 `AIE.switchbox` declarations. If the ports between switchboxes in tile(2,4) and tile(3,4) are changed from 3 to 4, does this change the behavior of our design? <img src="../../images/answer1.jpg" title="No" height=25>
