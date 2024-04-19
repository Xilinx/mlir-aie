<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2f - Data Movement Without Object FIFOs</ins>

Not all data movement patterns can be described with Object FIFOs. This section goes into detail about how a user can express data movement using the Data Movement Accelerators (or `DMA`) on AIE tiles.

The AIE architecture currently has three different types of tiles: compute tiles referred to as `tile`, memory tiles reffered to as `Mem tile`, and external memory interface tiles referred to as `Shim tile`. Each of these tiles has its own attributes regarding compute capabilities and memory capacity, but the base design of their DMAs is the same. The different types of DMAs can be intialized using the constructors in [aie.py](../../../python/dialects/aie.py):
```python
@mem(tile) # compute tile DMA
@shim_dma(tile) # Shim tile DMA
@memtile_dma(tile) # Mem tile DMA
```

The DMA hardware component has a certain number of input and output `channels`, and each one has a direction and a port index. For the direction input channels are denoted with the keyword `SS2M` and output ones with `M2SS`. Port indices vary per tile, for example compute tiles have two input and two output ports, same as Shim tiles, whereas Mem tiles have six input and six output ports.

A channel in any tile's DMA can be initialized using the unified `dma` constructor:
```python
def dma(
    channel_dir,
    channel_index,
    *,
    num_blocks=1,
    loop=None,
    repeat_count=None,
    sym_name=None,
    loc=None,
    ip=None,
)
```

Finally, the data movement on each channel is described by a chain of Buffer Descriptors (or `BD`), where each BD describes what data is being moved and configures its synchornization mechanism.


