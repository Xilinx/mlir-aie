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

In the [flows](../flow) portion of `Tutorial-4`, we covered the topics of configuring switchbox connection via the `AIE.flow` operator as well as the specification of tileDMAs within the `AIE.mem` blocks. A Flow is a logical abstraction to automate the route and physical switchbox connections between two endpoints, just like objectFifos can automate the the route along with the DMA configuration. But if more fine grained control is needed for the hardware switchboxes, we can use `AIE.switchbox` to configure individual switchboxes. These switchboxes, one in each AIE tile, are full crossbar, meaning that any 32b input port can be wired to any 32b output port using the `AIE.connect` operator as shown below:

```
AIE.switchbox(%tile14) { AIE.connect<"DMA": 0, "East": 1> }
AIE.switchbox(%tile24) { AIE.connect<"West": 1, "East": 3> }
AIE.switchbox(%tile34) { AIE.connect<"West": 3, "DMA": 1> }
```
The "East" output in %tile14 connects to the "West" input of %tile24, and so these three `AIE.connect` operations implement a flow from DMA-0 in %tile14 to DMA-1 in %tile34. A single switchbox can have many connect operations and is only limited by the number of connections available in hardware. 

## <ins>Tutorial 4 Lab</ins>
1. Take a look at the aie.mlir to see how the previous AIE.flow operator was replaced with 3 AIE.switchbox declarations. Are these switchbox operations equivalent to the flow? <img src="../../images/answer1.jpg" title="Yes, the switchboxes implement the flow. (AIE.wire ops are added automatically)" height=25>
 
2. If the ports between switchboxes in tile(2,4) and tile(3,4) are both changed from 3 to 4, does this change the behavior of our design? <img src="../../images/answer1.jpg" title="No, the stream would still be connected." height=25>

3. Build the design with `make` and simulate with `make -C aie.mlir.prj/sim` to see that the design functions correctly after replacing the flow op.

## <ins>Advanced Topics - Pathfinder Routing and Visualizations</ins>
The lowering from abstract `AIE.flow` ops to physical `AIE.switchbox` and `AIE.wire` ops is accomplished with the `--aie-create-pathfinder-flows` pass. This pass uses an iterative congestion-aware algorithm to find legal routes for all flows in the AIE array. For each flow, Pathfinder creates a route using Djikstra's Shortest Path algorithm. When congestion occurs (too many routes want to use the same physical wires) then the "demand" for those high congestion wires is increased, and then Pathfinder runs another iteration. This repeats iteratively until all flows are legally mapped onto the available routing resources. 

1. Examine `./path/pathfinder_input.mlir`. Here we are only interested in the routing of flows, so the core and memory sections have been removed. Instead we have a list of tiles and four flows which we want routed.
2. Run `make pathfinder` to perform the pass. Examine `./path/pathfinder_output.mlir` to see the results. How many switchboxes did the Pathfinder algorithm use in routing? <img src="../../images/answer1.jpg" title="There are 8 switchbox ops with detailed routing information in the output." height=25>
3. Open `./path/pathfinder_output.json` and run the prviewer extension by pressing F1 and running the `Routing View` command. After activating the extension, you will see only route0 displayed. To see all routes, hover over `route_all` at the bottom of `./path/pathfinder_output.json` and click `Display route_all` in the textbox that appears. You can also do the same to view only specific routes.
> Note: recall that the `prviewer` extension uses a (row, col) format for tile coordinates instead of (col, row) used in mlir.
4. Add a new flow in `./path/pathfinder_input.mlir`. For example, you could add a flow from tile (0, 3) to tile (4, 1). Run `make pathfinder` again to view the new routing.
5. Change an existing flow to go to another tile. View the new routing.