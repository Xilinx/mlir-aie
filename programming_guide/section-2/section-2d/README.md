<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2d - Programming for multiple cores</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * Section 2d - Programming for multiple cores
    * [Section 2e - Practical Examples](../section-2e/)
    * [Section 2f - Data Movement Without Object FIFOs](../section-2f/)
    * [Section 2g - Runtime Data Movement](../section-2g/)

-----

This section will focus on the process of taking code written for a single core and transforming it into a design with multiple cores relatively quickly. For this we will start with the code in [aie2.py](./aie2.py) which contains a simple design running on a single compute tile, and progressively turn it into the code in [aie2_multi.py](./aie2_multi.py) which contains the same design that distributes the work to three compute tiles.

The first step in the design is the tile declaration. In the simple design we use one Shim tile to bring data from external memory into the AIE array inside of a Mem tile that will then send the data to a compute tile, wait for the output and send it back to external memory through the Shim tile. Below is how those tiles are declared in the simple design:
```python
ShimTile = tile(0, 0)
MemTile = tile(0, 1)
ComputeTile = tile(0, 2)
```
For our scale out design we will keep using a single Shim tile and a single Mem tile, but we will increase the number of compute tiles to three. We can do so cleanly and efficiently in the following way:
```python
n_cores = 3

ShimTile = tile(0, 0)
MemTile = tile(0, 1)
ComputeTiles = [tile(0, 2 + i) for i in range(n_cores)]
```
Each compute tile can now be accessed by indexing into the `ComputeTiles` array.

Once the tiles have been declared, the next step is to set up the data movement using Object FIFOs. The simple design has a total of four double-buffered Object FIFOs and two `object_fifo_links`. The Object FIFOs move objects of datatype `<48xi32>`. `of_in` brings data from the Shim tile to the Mem tile and is linked to `of_in0` which brings data from the Mem tile to the compute tile. For the output side, `of_out0` brings data from the compute tile to the Mem tile where it is linked to `of_out` to bring the data out through the Shim tile. The corresponding code is shown below:
```python
data_size = 48
buffer_depth = 2
data_ty = np.ndarray[(48,), np.dtype[np.int32]]


# Input data movement

of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
of_in0 = object_fifo("in0", MemTile, ComputeTile, buffer_depth, data_ty)
object_fifo_link(of_in, of_in0)


# Output data movement

of_out = object_fifo("out", MemTile, ShimTile, buffer_depth, data_ty)
of_out0 = object_fifo("out0", ComputeTile, MemTile, buffer_depth, data_ty)
object_fifo_link(of_out0, of_out)
```

<img src="../../assets/SingleDesign.svg" height="300" width="700">

We can apply the same method as in the tile declaration to generate the data movement from the Mem tile to the three compute tiles and back ([see distribute and join patterns](../section-2b/03_Link_Distribute_Join/README.md)). The `object_fifo_link` operations change from the 1-to-1 case to distributing the original `<48xi32>` data tensors to the three compute tiles as smaller `<16xi32>` tensors on the input side, and to joining the output from each compute tile to the Mem tile on the output side. Lists of Object FIFOs are used to keep track of the input and output Object FIFOs. With these changes the code becomes:
```python
n_cores = 3
data_size = 48
tile_size = data_size // 3

buffer_depth = 2
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

# Input data movement
inX_fifos = []

of_in = object_fifo("in", ShimTile, MemTile, buffer_depth, data_ty)
for i in range(n_cores):
    inX_fifos.append(object_fifo(
        f"in{i}", MemTile, ComputeTiles[i], buffer_depth, tile_ty
    ))

# Calculate the offsets into the input/output data for the join/distribute
if n_cores > 1:
    of_offsets = [16 * i for i in range(n_cores)]
else:
    of_offsets = []
object_fifo_link(of_in, inX_fifos, [], of_offsets)


# Output data movement
outX_fifos = []

of_out = object_fifo("out", ShimTile, MemTile, buffer_depth, data_ty)
for i in range(n_cores):
    outX_fifos.append(object_fifo(
        f"out{i}", ComputeTiles[i], MemTile, buffer_depth, tile_ty
    ))
object_fifo_link(outX_fifos, of_out, of_offsets, [])
```

<img src="../../assets/MultiDesign.svg" width="1000">

The core of this simple design acquires one object of each Object FIFO, adds `1` to each entry of the incoming data, copies it to the object of the outgoing Object FIFO, then releases both objects:
```python
@core(ComputeTile)
def core_body():
    # Effective while(1)
    for _ in range_(0xFFFFFFFF):
        elem_in = of_in0.acquire(ObjectFifoPort.Consume, 1)
        elem_out = of_out0.acquire(ObjectFifoPort.Produce, 1)
        for i in range_(data_size):
            elem_out[i] = elem_in[i] + 1
        of_in0.release(ObjectFifoPort.Consume, 1)
        of_out0.release(ObjectFifoPort.Produce, 1)
```
Once again we apply the same logic and use a `for`-loop over our three cores to write the code which will be executed on the three compute tiles. Each tile will index the `inX_fifos` and `outX_fifos` maps to retrieve the Object FIFOs it will acquire and release from. This process results in the following code:
```python
for i in range(n_cores):
    # Compute tile i
    @core(ComputeTiles[i])
    def core_body():
        for _ in range_(0xFFFFFFFF):
            elem_in = inX_fifos[i].acquire(ObjectFifoPort.Consume, 1)
            elem_out = outX_fifos[i].acquire(ObjectFifoPort.Produce, 1)
            for i in range_(tile_size):
                elem_out[i] = elem_in[i] + 1
            inX_fifos[i].release(ObjectFifoPort.Consume, 1)
            outX_fifos[i].release(ObjectFifoPort.Produce, 1)
```

-----
[[Prev - Section 2c](../section-2c/)] [[Up](..)] [[Next - Section 2e](../section-2e/)]
