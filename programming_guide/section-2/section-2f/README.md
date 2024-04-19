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

Finally, the data movement on each channel is described by a chain of Buffer Descriptors (or `BD`), where each BD describes what data is being moved and configures its synchornization mechanism. The `dma` constructor already creates space for one such BD as can be seen by its `num_blocks=1` default valued input.

The code snippet below shows how to configure the DMA on `tile_a` such that data coming in on input channel 0 is written into `buff_in`:
```python
tile_a = tile(1, 3)

prod_lock = lock(tile_a, lock_id=0, init=1)
cons_lock = lock(tile_a, lock_id=1, init=0)
buff_in = buffer(tile=tile_a, shape=(256,), dtype=T.i32()) # 256xi32

@mem(tile_a)
def mem_body():
    @dma(S2MM, 0) # input channel, port 0
    def dma_in_0():
        use_lock(prod_lock, AcquireGreaterEqual)
        dma_bd(buff_in)
        use_lock(cons_lock, Release)
```
The locks `prod_lock` and `cons_lock` follow AIE2 architecture semantics. Their task is to mark synchronization points in the tile's and its DMA's execution: for example, if the tile is currently using `buff_in` it will only release the `prod_lock` when it is done and that is when the DMA will be allowed to overwrite the data in `buff_in` with new input. Similarly, the tile's core can query the `cons_lock` to know when the new data is ready to be read.

In the previous code the channel only had one BD in its chain. To add additional BDs to the chain, users can use the following constructor, which takes as input what would be the previous BD in the chain it should be added to:
```python
@another_bd(dma_bd)
```

This next code snippet shows how to extend the previous input channel with a double, or ping-pong, buffer using the previous constructor:
```python
tile_a = tile(1, 3)

prod_lock = lock(tile_a, lock_id=0, init=2) # note that the producer lock now has 2 tokens
cons_lock = lock(tile_a, lock_id=1, init=0)
buff_ping = buffer(tile=tile_a, shape=(256,), dtype=T.i32()) # 256xi32
buff_pong = buffer(tile=tile_a, shape=(256,), dtype=T.i32()) # 256xi32

@mem(tile_a)
def mem_body():
    @dma(S2MM, 0, num_blocks=2) # input channel, port 0
    def dma_in_0():
        use_lock(prod_lock, AcquireGreaterEqual)
        dma_bd(buff_ping)
        use_lock(cons_lock, Release)

    @another_bd(dma_in_0)
    def dma_in_1():
        use_lock(prod_lock, AcquireGreaterEqual)
        dma_bd(buff_pong)
        use_lock(cons_lock, Release)
```


