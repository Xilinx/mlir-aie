<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Tutorial 5 - Communication via flows (shim DMA, external memory aka DDR)</ins>

The [Tutorial-5 objectFifos](..) simplifies the process of declaring data communication across L1-L3 boundaries but understanding what components are needed at the flow level is very helpful in developing other abstractions of communications. As mentioned previously, the 3 additional blocks needed to support L1-L3 communication are:

* Shim DMA and External Buffers
* NOC configuration
* Host code for buffer allocation and virtual address mapping

As review, the diagram featuring the 3 blocks needed to connect L1 to L3 can be seen in the following diagram.
<p><img src="../../images/diagram9.png" width="1000"><p>

Here, we see the different components of the L1-L3 communication defined in MLIR. The shim DMA is the box labeled AI Engine Interface Tile while the external buffer is the smaller gray box within the blue DDR box. We see the NOC block represented by the light gray box labeled NOC. And the host code portion would be found in the host code [test.cpp](./test.cpp).

## <ins>Shim DMA and External Buffers</ins>
### <ins>shim_dma</ins>
We first need a component to move the data into and out of the AIE array and that component can be the shim DMA which is connected to the NoC block (the other option is the programmable logic or PL interfaces which we will not cover here). For this tutorial, we will focus on the shim DMA as that does not require PL blocks to move data to the DDR controller.

The shim DMA functions very similarly to the tile DMA when defined in MLIR. Rather than define the BD behavior inside an `AIE.mem` operator, we define the same set of BD behaviors inside the `AIE.shim_dma` operator as shown below:
```
%shimdma70 = AIE.shim_dma(%tile70) {
    AIE.dma_start("MM2S", 0, ^bd1, ^end)
    ^bd1:
        AIE.use_lock(%lock70_in, "Acquire", 1)
        AIE.dma_bd(%external_buf : memref<256xi32>, 0, 256>, A)
        AIE.use_lock(%lock70_in, "Release", 0)
        cf.br ^end
    ^end:
        AIE.end
}
```
Here, we see that the rules for bd and channel definitions are the same as in the tileDMA case.
> Note that shim_dma are defined for the shim tiles (row 0). In this example, tile(7,0). Also note that not every column in row 0 is shim_dma capable. The list of column values of capable tiles in the S70 device for example is `(2,3,6,7,10,11,18,19,26,27,34,35,42,43,46,47)`.

Much like the tile DMA, the shim DMA has 2 DMA units, each with a read and write port, giving us 4 independent dma+channel data movers. Among all 4 data movers, we again have 16 buffer descriptors (bd) describing the rules of the data movement. The definition of these bds are declared within an `AIE.shim_dma` operation in the same way as the tile DMA. Please review the tile DMA operations in [tutorial-4/flow](../../tutorial-4/flow) for more details.

### <ins>External_buffer</ins>
The second operator is the definition of the external buffer. Tile DMA moves data from the local memory of each AI Engine. But shim DMA moves data from external buffers (e.g. DDR). The `dmabBd` operator then needs to refer to this buffer in its definition. External buffers are defined with the `AIE.external_buffer` operation as shown below:
```
%ext_buf70_in  = AIE.external_buffer {sym_name = "ddr_test_buffer_in"}: memref<256xi32>
```
This looks very much like a local buffer definition except that it's not attached to any tile. Where this memory is physically located and how the shim_dma is able to connect to it is defined in the next two blocks.

> **NOTE - switchbox for shim Tile**: The shim tile switchbox has slightly different capabilities than a standard tile. For shim tiles, valid bundle names and channels are listed below: 

| Bundle | Channels (In) | Channels (Out) |
|-------|---|---|
| DMA   | 2 | 2 |
| West  | 4 | 4 |
| East  | 4 | 4 |
| North | 4 | 6 |
| South | 8 | 6 |
| FIFO  | 2 | 2 |
| Trace | 1 | 0 |

## <ins>NOC configuration</ins>

The next block to configure is the NOC interface that is connected to all shimDMAs to route to a valid external buffer. In the S80 device, for example, this can be to the DDR memory controller or other memory component connected to the NOC (e.g. BRAM controller). In our example platform, we have created a design where all NOC ports are able to route to the DDR memory controller but in practice, this step is done as part of the platform design. Future efforts to streamline the NOC configuration at run time is ongoing.

## <ins>Host code for buffer allocation and virtual address mapping</ins>

The last block to configure is the external buffer itself. Because our shim DMA is connected to a DDR memory controller, it can access any valid memory location therein. We then need to allocate a valid region of memory and pass that virtual address to the host code configuration functions so the shim DMA is configured correctly. For all tileDMAs, they are configured at runtime through the `mlir_aie_configure_dmas()` function. But this does not include the configuration of the shim DMAs. This is done as follows:
```
int *mem_ptr_in  = mlir_aie_mem_alloc(_xaie, 0, 256);
mlir_aie_external_set_addr_ddr_test_buffer_in((u64)mem_ptr_in);
mlir_aie_configure_shimdma_70(_xaie);
```
In this example, we first call `mlir_aie_mem_alloc` to allocate a region of DDR memory with a given offset and size and return a virtual address pointer. Then, in the `mlir_aie_external_set_addr_<bufname>(virtual_addr)`, we pass in the virtual address to MLIR defined external buffer. 
> Note that the `<bufname>` used here is the `sym_name` defined in the MLIR code. 

Finally, the `mlir_aie_configure_shimdma_<location>()` is called to configure the shim_dma given the shim DMA operators in MLIR and the virtual address defined at runtime in the host code. The `<location>` refers to the shim DMA defined in MLIR and is the concatenation of the column-row number, in this case column 7, row 0 or 70. 

Once these three functions are called, the shim DMA is configured properly with the runtime allocated memory region in DDR. Since the common use of shim_dma requires timing synchronization to start a transaction, we often use locks to do this just as we did in the tile DMA example. Here, we can acquire and release locks in the shim_dma using the following access functions:
```
mlir_aie_acquire_<sym_name>_lock(_xaie, 1, 100);
mlir_aie_release_<sym_name_lock(_xaie, 0, 100);
```
The `<sym_name>` used here is the same sym_name of the external buffer. The first argument is the lock value (0,1) and the second argument is the timeout duration in microseconds.

## <ins>Tutorial 5 Lab </ins>

1. Read through the [aie.mlir](aie.mlir) design. How many external buffers are defined and which direction are they? <img src="../../images/answer1.jpg" title="2 buffers. ext_buf70_in is for reading (DDR->L1). ext_buf70_out is for writing (L1->DDR)" height=25>

External buffers on their own cannot give any indication as to what they are used for but we can figure this out based on the bd description that the buffer is used in. For example, `ext_buf70_in` is defined in `bd1` which is itself defined for `dma_start("MM2S")` which tells us this is a MM2S connection. 
> Note that MM2S means memory map to stream. In this case, the MM is the external buffer side (e.g. DDR) and the stream is the AIE array side. So we are moving data from the external buffer to the AIE array. This is kind of the opposite to the tile DMA case where MM2S would be moving data from local memory to the stream which would be writing data out from the perspective of the AIE core.

2. Add a second read and write channel to the single shim_dma (tile(7,0)) that moves data to and from another tile. That tile can have the same function as the existing tile.

3. Can we add a third read or write channel to our shim_dma? <img src="../../images/answer1.jpg" title="No, DMAs have only 2 channels in first generation AI Engines" height=25>

4. Change the design so that the external buffer acts like a ping-pong buffer.

