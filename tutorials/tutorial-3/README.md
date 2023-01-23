<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 3 - communication (local memory), locks</ins>

After declaring the `core` and `buffer` dialect operations which map to the core and local memory respectively, and then defining the functionality within cores with either integrated dialect operations (arith, memref) or external kernel functions, the next major component of for AIE system design is communciation. As summarized briefly in the [Basic AI Engine Architecure](../README.md) section, communication via local memory is one of the most efficient ways to share data and can be done among up to 4 tiles adjacent to a local memory. In `mlir-aie`, all tiles have an associated local memory but adjacent tiles are able to read and write to that memory as well. 

In the diagram below, we see that the local memory for tile(1,4) is accessible to the core in tile(1,3). If we were to expand the diagram further, we would see that tile(0,4) and  tile(1,5) can also access that buffer. That is why the core in tile(1,3) can reference the buffer declared by tile(1,4).

<p><img src="../images/diagram4.jpg?raw=true" width="800"><p>


While the tile does naturally arbitrate between read and write requests, to avoid access conflicts, we use such hardware locks to gain exclusive access to the local memory. Bear in mind that these locks are not explicitly tied to the local memory and can be use for any purpose. But using them in this way helps with arbitration and performance.

## <ins>Locks</ins>
The `lock` operation, is actually a physical sub component of the `AIE.tile` but is declared within the `module` block. The syntax for declaring a lock is `AIE.lock(tileName, lockID)`. An example would be:
```
%lockName = AIE.lock(%tileName, %lockID)
```
Examples:
```
%lock13_4 = AIE.lock(%tile13, 4)
%lock13_11 = AIE.lock(%tile13, 11)
```
Each tile has 16 locks and each lock is in one of two states (acquired, released) and one of two values (0, 1).
> By default, we tend to assume (value=0 is a "write", value=1 is "read"). But there is no real definition of these values. The only key thing to remember is that lock value start and is reset into the release val=0 state. Which means an acquire=0 will always succeed first while an acquire=1 needs the lock state to be release=1 to succeed. Once acquired, a lock can be released to the 0 or 1 state. 

The 16 locks in a tile are accessible by it's the same 3 cardinal neighbors that can access the tile's local memory. This is to ensure all neighbors that can access the local memory can also access the locks. 

To use the lock, we call the `useLock` operation either inside a `core` operation or `mem/ shimDMA` operation. 
```
AIE.useLock(%lockName, "Acquire|Release", 0|1)
```
That code would look something like:
```
%core14 = AIE.core(%tile14) {
    AIE.useLock(%lock14_7, "Acquire", 0)
    ... core ops ...
    AIE.useLock(%lock14_7, "Release", 1)
}
```
Notice the familiar design pattern of:
* acquire lock in some value
* a set of operations
* release lock in some value (usually the other value)

The acquire value must match the current lock state in order for the acqure to succeed. The release value can be either 0 or 1. Below is another example of lock usage including a common state diagram of lock state transitions. Note that we can actually release to the same value if we choose.
<p><img src="../images/diagram5.jpg?raw=true" width="800"><p>

## <ins>Tutorial 3 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. Which tile's local memory is being shared between the two tiles? <img src="../images/answer1.jpg" title="tile(2,4)" height=25>

2. Can we share tile(1,4)'s local memory instead? Why or why not? <img src="../images/answer1.jpg" title="No, they do not both see tile(1,4) local memory" height=25>

3. What about in the vertical direction, say between tile(1,3) and tile(1,4). Which tiles' local memory can be shared between these two tiles? <img src="../images/answer1.jpg" title="both tile(1,3) and tile(1,4) can be shared" height=25>

4. Change the lock from belonging to tile(2,4) to tile(1,4). Does this change the behavior of our design? What does that say about the total number of locks available between two adjacent tiles? <img src="../images/answer1.jpg" title="No. Two adjacent tiles have up to 32 locks available to them." height=25>

5. Based on what you know about locks, which tile will execute its kernel code inside the lock calls first in this design? <img src="../images/answer1.jpg" title="tile(1,4)" height=25>

6. **Add simulation instructions here**

7. Change the design so that tile(2,4) runs first. What do you expect the value of buf[5] will be with this change? <img src="../images/answer1.jpg" title="100" height=25>

8. Change [test.cpp](test.cpp) so the testbench expects the correct result and passes again in simulation/ hardware. 

## <ins>Object FIFO Abstraction </ins>

In this tutorial the `objectFifo` abstraction is also introduced, see below. This is a higher-level abstraction which is used to establish communication accross the AI Engine array without explicit configuration of the involved `mlir-aie` components. The following tutorials will use this abstraction to introduce the `mlir-aie` dialect further.

[Link to higher level objectFifo write-up](./objectFifo_ver)
