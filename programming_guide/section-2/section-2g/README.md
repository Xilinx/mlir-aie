<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2g - Runtime Data Movement</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Programming for multiple cores](../section-2d/)
    * [Section 2e - Practical Examples](../section-2e/)
    * [Section 2f - Data Movement Without Object FIFOs](../section-2f/)
    * Section 2g - Runtime Data Movement

-----

In the preceding sections, we looked at how we can describe data movement between tiles *within* the AIE-array. However, to do anything useful, we need to get data from outside the array, i.e., from the "host", into the AIE-array and back. On NPU devices, we can achieve this with the operations described in this section. 

The operations that will be described in this section must be placed in a separate `aie.runtime_sequence` operation. The arguments to this function describe buffers that will be available on the host side; the body of the function describes how those buffers are moved into the AIE-array. [Section 3](../../section-3/) contains an example.

### Guide to Managing Runtime Data Movement to/from Host Memory

In high-performance computing applications, efficiently managing data movement and synchronization is crucial. This guide provides a comprehensive overview of how to utilize the `npu_dma_memcpy_nd` and `dma_wait` functions to manage data movement at runtime from/to host memory to/from the AIE array (for example, in the Ryzen™ AI NPU).

#### **Efficient Data Movement with `npu_dma_memcpy_nd`**

The `npu_dma_memcpy_nd` function is key for enabling non-blocking, multi-dimensional data transfers between different memory regions between the AI Engine array and external memory. This function is essential in developing real applications like signal processing, machine learning, and video processing.

**Function Signature and Parameters**:
```python
npu_dma_memcpy_nd(metadata, bd_id, mem, offsets=None, sizes=None, strides=None)
```
- **`metadata`**: This is a reference to the object FIFO or the string name of an object FIFO that records a Shim Tile and one of its DMA channels allocated for the host-side memory transfer. In order to associate the memcpy operation with an object FIFO, this metadata string needs to match the object FIFO name string.
- **`bd_id`**: Identifier integer for the particular Buffer Descriptor control registers used for this memcpy. A buffer descriptor contains all information needed for a DMA transfer described in the parameters below. 
- **`mem`**: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to. 
- **`offsets`** (optional): Start points for data transfer in each dimension. There is a maximum of four offset dimensions.
- **`sizes`**: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
- **`strides`** (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data. There is a maximum of three stride dimensions that can be expressed because dimension 0 is an implicit stride of 1 4B element. 

It is important to note that dimension 0 of the **`sizes`** and all **`strides`** are expressed in a 4B granularity. Higher dimensions of the **`sizes`** are integers to repeat the lower dimensions. The **`offsets`** are expressed in multiples of the **`sizes`**, however the dimension 0 offset is in a 4B granularity. The strides and wraps express data transformations analogously to those described in [Section 2C](../section-2c).

**Example Usage**:
```python
npu_dma_memcpy_nd(of_in, 0, input_buffer, sizes=[1, 1, 1, 30])
```

The example above describes a linear transfer of 30 data elements, or 120 Bytes, from the `input_buffer` in host memory into an object FIFO with matching metadata labeled "of_in". The `size` dimensions are expressed right to left where the right is dimension 0 and the left dimension 3. Higher dimensions not used should be set to `1`.


#### **Advanced Techniques for Multi-dimensional `npu_dma_memcpy_nd`**

For high-performance computing applications on AMD's AI Engine, mastering the `npu_dma_memcpy_nd` function for complex data movements is crucial. Here, we focus on using the `sizes`, `strides`, and `offsets` parameters to effectively manage intricate data transfers.

##### **Tiling a Large Matrix**

A common tasks such as tiling a 2D matrix can be implemented using the `npu_dma_memcpy_nd` operation. Here’s a simplified example that demonstrates the description.

**Scenario**: Tiling a 2D matrix from shape [100, 200] to [20, 20] and the data type `int16`. With the convention [row, col].

**1. Configuration to transfer one tile**:
```python
metadata = of_in
bd_id = 3
mem = matrix_memory  # Memory object for the matrix

# Sizes define the extent of the tile to copy
sizes = [1, 1, 20, 10]

# Strides set to '0' in the higher (unused) dimensions and to '100' (length of a row in 4B or "i32s") in the minor dimension
strides = [0, 0, 0, 100]  

# Offsets set to zero since we start from the beginning
offsets = [0, 0, 0, 0]

npu_dma_memcpy_nd(metadata, bd_id, mem, offsets, sizes, strides)
```

**2. Configuration to tile the whole matrix**:
```python
metadata = of_in
bd_id = 3
mem = matrix_memory  # Memory object for the matrix

# Sizes define the extent of the tile to copy.
# Dimension 0 is 10 to transfer 20 int16s for one row of the tile,
# Dimension 1 repeats that row transfer 20 times to complete a [20, 20] tile,
# Dimension 2 repeats that tile transfer 10 times along a row,
# Dimension 3 repeats the row of tiles transfer 5 times to complete.
sizes = [5, 10, 20, 10]

# Strides set to '0' in the highest (unused) dimension,
# '2000' for the next row of tile below the last (200 x 20 x 2B / 4B),
# '10' for the next tile to the 'right' of the last [20, 20] tile,
# and '100' (length of a row in 4B or "i32s") in dimension 0.
strides = [0, 2000, 10, 100]  

# Offsets set to zero since we start from the beginning
offsets = [0, 0, 0, 0]

npu_dma_memcpy_nd(metadata, bd_id, mem, offsets, sizes, strides)
```

#### **Host Synchronization with `dma_wait`**

Synchronization between DMA channels and the host is facilitated by the `dma_wait` operation, ensuring data consistency and proper execution order. The `dma_wait` operation waits until the BD associated with the ObjectFifo is complete, issuing a task complete token.

**Function Signature**:
```python
dma_wait(metadata)
```
- **`metadata`: The ObjectFifo python object or the name of the object fifo associated with the DMA option we will wait on.

**Example Usage**:

Waiting on DMAs associated with one object fifo:
```python
# Waits for the output data to transfer from the output object fifo to the host
dma_wait(of_out)  
```

Waiting on DMAs associated with more than one object fifo:
```python
dma_wait(of_in, of_out)  
```

#### **Best Practices for Data Movement and Synchronization**

- **Sync to Reuse Buffer Descriptors**: Each `npu_dma_memcpy_nd` is assigned a `bd_id`. There are a maximum of `16` BDs available to use in each Shim Tile. It is "safe" to reuse BDs once all transfers are complete, this can be managed by properly synchronizing taking into account the BDs that must have completed to transfer data into the array to complete a compute operation. And then sync on the BD that receives the data produced by the compute operation to write it back to host memory. 
- **Note Non-blocking Transfers**: Overlap data transfers with computation by leveraging the non-blocking nature of `npu_dma_memcpy_nd`.
- **Minimize Synchronization Overhead**: Synchronize/wait judiciously to avoid excessive overhead that might degrade performance.

#### **Conclusion**

The `npu_dma_memcpy_nd` and `dma_wait` functions are powerful tools for managing data transfers and synchronization with AI Engines in the Ryzen™ AI NPU. By understanding and effectively implementing applications leveraging these functions, developers can enhance the performance, efficiency, and accuracy of their high-performance computing applications.

-----
[[Prev - Section 2f](../section-2f/)] [[Up](..)] [[Next - Section 3](../../section-3/)]
