<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 4 - communication (tile DMA, logical routing)</ins>

Even though shared local memory is ideal for communicating between adjacent tiles, for non-adjcent tiles or tiles that are far away, communication is instead done through the stream switch network with the tile DMAs serving as data movers and each tile's local switchbox serving as the data steering highway. Efficient movement of data through the AIE array is key to achieving high performance. The `mlir-aie` dialect has buit-in automatic routing capabilities and visulization tools for viewing the routing choices made. This is an great area for further research where additional transformation can be added to provide other optimized routing functionality.

## <ins>Streams</ins>
Streams are 32-bits wide and runs at the system clock rate (e.g. 1 GHz). There are 4 streams running horizontally and vertically in both into and out of the switchbox (except in the vertical direction going north where there are 6 streams in and out). The AIE core does have 2 input and 2 output stream ports that connect to its local swtichbox but the more common way of pushing large blocks of data into and out of the stream network is with the tile DMAs. Rather than having the core read and write streams directly, the core accesses local memory and the tile DMA moves the data from local memory to the swtichbox.

## <ins>Logical routing (aka flows)</ins>
The AIE switchboxes are very powerful configurable blocks for steering data between AI Engine tiles. Having 24 inputs and 24 outputs, the switchbox has connections to its neighbor, the tileDMA, the core, and an internal fifo. Each connection from input to output within the switchbox can operate in one of two modes: 
* circuit swtich mode
* packet swtich mode

We will discuss the circuit swtich mode first and take a look at the packet switch mode in [tutorial-6](../../tutorial-6). Circuit switch mode is where a given route between ports of the switchbox are fixed, allowing data to be passed from one input port to one or more output ports. The path is fixed after configuration and the data movers (DMAs) can push data as frequently as it wishes along that fixed route.

A set of circuit switched routes from a source port in one switchbox to a destination port in another switchbox (potentially far way) is called a `flow ` and can be automatically determined and configured using the `AIE.flow` operation. 
```
AIE.flow ($source, $sourceBundle : $sourceChannel, $dest, $destBundle : $destChannel)
```
An example of a flow is shown below:
```
AIE.flow(%tile71, "South" : 3, %tile73, "DMA"   : 0)
AIE.flow(%tile73, "DMA"   : 1, %tile71, "South" : 2)
```
Notice that the route from tile(7,1) to tile(7,3) will most likley involve tile(7,2) who's inclusion in the route is not explicitly stated. Instead, we just declare the end points and allow the logical routing algorihtm (aka pathfinder) to determine the optimized path for data to flow and configures the switchboxes to enable that flow.
> You can explore the lowering of logical `flows` to physical switchbox configrations in [here](./switchbox)

In the above flow syntax, valid bundle names and channels are listed below: 
| Bundle | Channels (In) | Channels (Out) |
|-------|---|---|
| DMA   | 2 | 2 |
| Core  | 2 | 2 |
| West  | 4 | 4 |
| East  | 4 | 4 |
| North | 4 | 6 |
| South | 6 | 4 |
| PLIO  | 2?| 2?|
>All channels function identically in behavior but the key is that the chosen dma channel needs to match the declared dma channel in the tile DMA definition that will be explained next

## <ins>Tile DMAs</ins>
There are 2 tile DMAs connected to the local switchbox, each with a read and write port, giving us 4 independent dma+channel data movers. Among all 4 data movers, we have 16 buffer descriptors (bd) describing the rules of the data movement. The definition of these bds are declared within an `AIE.mem` operation as shown below:
```
%mem14 = AIE.mem(%tile14) {
    %dma1 = AIE.dmaStart("MM2S", 0, ^bd0, ^end)
    ^bd0:
        AIE.useLock(%lock14_6, "Acquire", 1)
        AIE.dmaBd(<%buf14: memref<256xi32>, 0, 256>, 0)
        AIE.useLock(%lock14_6, "Release", 0)
        cf.br ^end
    ^end:
        AIE.end
}
```
> Note that bds can be associated with any DMA (both input and output). They can also be flexibly chained together.

The defintion of the body of the `AIE.mem` op is generally composed of labels which we use to chain bd definitions together using the `cf.br` op. These label names are arbitrary and do not necessarily correspond to a defined bd value. In our example, the `^bd0` label does define a bd definition but the `^end` label does not.  

In defining bds within the local memory (`AIE.mem`), we start with defining the DMA used (`AIE.dmaStart`) which is defined as:
```
AIE.dmaStart($channelDir, $channelIndex, $dest, $chain)
```
`$channelDir`: DMA block based on direction (MM2S, S2MM)
`$channelIndex`: DMA channel (0, 1). First generation tile DMA has 2 channels in each direction.
`$dest`: label of 1st bd in chain
`$chain`: label of next definition in our declaration chain. Generally point to the label where `AIE.end` is defined but can point to the next call to another `AIE.dmaStart`

### <ins>Buffer descriptors</ins>
Within a memory module, we can define up to 16 buffer descriptors, each which refrences the buffer in local memory that it will be accessing (either read from or write to).
The `AIE.dmaBd` syntax is:
```
AIE.dmaBd(<$buffer : type($buffer), $offset, $len>, $AB)
```
`$buffer : type($buffer)`: the usual buffer declaration like the one used in `memref` (e.g. %buf14: memref<256xi32>). 

`%offset`:  offset in bytes that the dma accesses from the beginning of the buffer. 

`%length`: length of the DMA operation in bytes

`%AB`: unique configuration where the bd is set to use either the A or B buffer. For now, just leave this as a 0.

Each bd operation can also optionally be gated by locks meaning a lock must be acquired before the bd operation starts and releaesd after the bd operation is complete. These lock acquires and releases can be used to time when bd transfers are done.
> The lock operations are optional and the lock ID used in acquire and release do not have to match each other though they usually do.

Below is a diagram that maps the components introduced with the physical block on the AI Engine tile.
<p><img src="../../images/diagram6.jpg?raw=true" width="800"><p>

## <ins>Tutorial 4 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. Which DMA direction and channel is used on tile(1,4) to send data to tile(3,4)? <img src="../../images/answer1.jpg" title="MM2S, channel 0" height=25>

2. What is the shared lock ID used in tile(1,4) to arbitrate access to its local memory? <img src="../../images/answer1.jpg" title="6" height=25>

3. What is the DMA direction/ channel and lock ID for the tile DMA used in tile(3,4) to receive data? <img src="../../images/answer1.jpg" title="S2MM, channel 1, lock ID=7" height=25>

4. Based on the lock status, who will successfully acquire a lock when the core are first enabled?
<img src="../../images/answer1.jpg" title="tile(1,4)'s core, tile(3,4)'s tile DMA (S2MM, ch#1)" height=25>

5. To explore and understand bd behavior further, make changes to [aie.mlir](aie.mlir) so the bd forever. Then change the testbench [test.cpp](test.cpp) so we force the two tileDMAs to run again. 
    >HINT: Acquire and release the locks for the tileDMAs in the right states. 

    What do you expect to happen to the output? <img src="../../images/answer1.jpg" title="Test will fail since the we overwrite the buf34 again." height=25>

6. *** Add timers to measure tiledma transfer time ***

7. *** Example that cannot route ***

8. *** View route results *** 
