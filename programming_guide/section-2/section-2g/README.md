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
- **`tap`** (optional): A `TensorAccessPattern` is an alternative method of specifying `offset`/`sizes`/`strides` for determining an access pattern over the `mem` buffer.
- **`offsets`** (optional): Start points for data transfer in each dimension. There is a maximum of four offset dimensions.
- **`sizes`**: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
- **`strides`** (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.

The strides and sizes express data transformations analogously to those described in [Section 2C](../section-2c).

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

#### **Host Synchronization with `dma_wait` after one or more `npu_dma_memcpy_nd` operations**

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

#### **Best Practices for Data Movement and Synchronization with `npu_dma_memcpy_nd`**

- **Sync to Reuse Buffer Descriptors**: Each `npu_dma_memcpy_nd` is assigned a `bd_id`. There are a maximum of `16` BDs available to use in each Shim Tile. It is "safe" to reuse BDs once all transfers are complete, this can be managed by properly synchronizing taking into account the BDs that must have completed to transfer data into the array to complete a compute operation. And then sync on the BD that receives the data produced by the compute operation to write it back to host memory. 
- **Note Non-blocking Transfers**: Overlap data transfers with computation by leveraging the non-blocking nature of `npu_dma_memcpy_nd`.
- **Minimize Synchronization Overhead**: Synchronize/wait judiciously to avoid excessive overhead that might degrade performance.

#### **Efficient Data Movement with `dma_task` Operations**

As an alternative to `npu_dma_memcpy_nd` and `dma_wait`, there is a series of operations around **DMA tasks** that can serve a similar purpose.

There are two advantages of using the DMA task operations over using `npu_dma_memcpy_nd`:
* The user does not have to specify a BD number
* DMA task operations are capable of *chaining* BD operations; however, this is an advance use-case beyond the scope of this guide. 

All programming examples have an `*_alt.py` version that is written using DMA task operations.

**Function Signature and Parameters**:
```python
def shim_dma_single_bd_task(
    alloc,
    mem,
    tap: TensorAccessPatter | None = None,
    offset: int | None = None,
    sizes: MixedValues | None = None,
    strides: MixedValues | None = None,
    transfer_len: int | None = None,
    issue_token: bool = False,
)
```
- **`alloc`**: The `alloc` argument associates the DMA task with an ObjectFIFO. This argument is called `alloc` becuase the shim-side end of a data transfer (specifically a channel on a shim tile) is referenced through a so-called "shim DMA allocation". When an ObjectFIFO is created with a Shim Tile endpoint, an allocation with the same name as the ObjectFIFO is automatically generated.
- **`mem`**: Reference to a host buffer, given as an argument to the sequence function, that this transfer will read from or write to. 
- **`tap`** (optional): A `TensorAccessPattern` is an alternative method of specifying `offset`/`sizes`/`strides` for determining an access pattern over the `mem` buffer.
- **`offset`** (optional): Starting point for the data transfer. Default values is `0`.
- **`sizes`**: The extent of data to be transferred across each dimension. There is a maximum of four size dimensions.
- **`strides`** (optional): Interval steps between data points in each dimension, useful for striding-across and reshaping data.
- **`issue_token`** (optional): If a token is issued, one may call `dma_await_task` on the returned task. Default is `False`.

The strides and strides express data transformations analogously to those described in [Section 2C](../section-2c).

**Example Usage**:
```python
out_task = shim_dma_single_bd_task(of_out, C, sizes=[1, 1, 1, N], issue_token=True)
```

The example above describes a linear transfer of `N` data elements from the `C` buffer in host memory into an object FIFO with matching metadata labeled "of_out". The `sizes` dimensions are expressed right to left where the right is dimension 0 and the left dimension 3. Higher dimensions not used should be set to `1`.

#### **Host Synchronization with `dma_await_task`**

Synchronization between DMA channels and the host is facilitated by the `dma_await_task` operations, ensuring data consistency and proper execution order. The `dma_await_task` operation waits until all the BDs associated with a task have completed.

**Function Signature**:
```python
def dma_await_task(*args: DMAConfigureTaskForOp)
```
- `args`: One or more `dma_task` objects, where `dma_task` objects are the value returned by `shim_dma_single_bd_task`.

**Example Usage**:

Waiting on task completion of one DMA task:
```python
# Waits for the output task to complete
dma_await_task(out_task)  
```

Waiting on task completion of more than one DMA task:
```python
# Waits for the input task and then the output task to complete
dma_await_task(in_task, out_task)  
```

#### **Free BDs without Waiting with `dma_free_task`**

`dma_await_task` can only be called on a task created with `issue_token=True`. If `issue_token=False` (which is default), then `dma_free_task` should be called when the programmer knows that task if complete. `dma_free_task` allows the compiler to reuse the BDs of a task without synchronization. Using `dma_free_task(X)` before task `X` has completed will lead to a race condition and unpredictable behavior. Only use `dma_free_task(X)` in conjunction with some other means of synchronization. For example, you may issue `dma_free_task(X)` after a call to `dma_await_task(Y)` if you can reason that task `Y` can only complete after task `X` has completed.

**Function Signature**:
```python
def dma_free_task(*args: DMAConfigureTaskForOp)
```
- `args`: One or more `dma_task` objects, where `dma_task` objects are the value returned by `shim_dma_single_bd_task`.

**Example Usage**:

Release BDs belonging to DMAs associated with one task:
```python
# Allow compiler to reuse BDs of a a task. Should only be called if the programmer is sure the task is completed.
dma_free_task(out_task)  
```

Release BDs belonging to DMAs associated with more than one task:
```python
# Allow compiler to reuse BDs of more than one task. Should only be called if the programmer is sure all tasks are completed.
dma_free_task(in_task, out_task)  
```

#### **Best Practices for Data Movement and Synchronization with `dma_task` Operations**

- **Await or Free to Reuse Buffer Descriptors**: While the exact buffer descriptor (BD) used for each operation is not visible to the user with the `dma_task` operations, there are still a finite number (maximum of `16` on a Shim Tile). Thus, it is important to use `dma_await_task` or `dma_free_task` before the number of BDs are exhausted so that they may be reused. 
- **Note Non-blocking Transfers**: Overlap data transfers with computation by leveraging the non-blocking nature of `dma_start_task`.
- **Minimize Synchronization Overhead**: Synchronize/wait judiciously to avoid excessive overhead that might degrade performance.

#### **Conclusion**

The `npu_dma_memcpy_nd` and `dma_wait` functions are powerful tools for managing data transfers and synchronization with AI Engines in the Ryzen™ AI NPU. By understanding and effectively implementing applications leveraging these functions, developers can enhance the performance, efficiency, and accuracy of their high-performance computing applications.

-----
[[Prev - Section 2f](../section-2f/)] [[Up](..)] [[Next - Section 3](../../section-3/)]
