<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# Matrix Multiplication - Whole Array Design

The code in this directory showcases an example matrix multiplication design for a Ryzen AI device with an NPU (Neural Processing Unit). 
The NPU consists of an array of compute cores, called AI Engines (AIEs);
The example design configures each of those compute cores to perform multiplications of distinct sub-matrices in parallel.

At a high level, the code does the following (in order):

1. [**Defining Matrix Dimensions and Data Types:**](#1-defining-matrix-dimensions-and-data-types) We first specify the dimensions `M`, `K`, `N` for the input matrices `A` (`M`&times;`K`), and `B` (`K`&times;`N`), and the output matrix `C` (`M`&times;`N`), as well as their data type. To enable efficient computation, our design will split large input matrices into smaller sub-matrix blocks on two levels; we thus also define the sizes of those sub-matrices. At the first level, the constants `m`, `k`, and `n` define the size of the submatrices processed by each AIE core. At the second level, we further subdivide using smaller sizes `r`, `s` and `t` -- these are the sizes of required by the vector computation intrinsics of the AIEs. 

1. [**Constructing an AIE Array Configuration:**](#2-constructing-an-aie-array-configuration) The NPU hardware is comprised of components laid out in a two-dimensional grid of rows and columns. Based on the matrix sizes and tiling factors, we choose the number of rows, columns, and total number of compute cores of the AIE device that the design should utilize. We then configure the AI Engine array, memory tiles, and shim tiles.

1. [**Defining Data Movement Inside the NPU:**](#3-defining-data-movement-inside-the-npu) ObjectFIFOs are a data movement abstraction for buffering data and synchronizing between AIE components. We configure ObjectFIFOs for `A`, `B` and `C` to transfer and buffer data between AIE components in chunks of the previously defined sizes (`m`&times;`k`, `k`&times;`n` and `m`&times;`n`, respecively).

1. [**Defining Core Computations:**](#4-defining-core-computations) The `core_body()` function contains the code that will be loaded onto each AIE core. This code describes the matrix multiplication using the input submatrices `a` and `b` acquired through the ObjectFIFOs. The results are accumulated in the output submatrix `c`.

1. [**Defining External Data Transfer Sequences:**](#5-defining-external-data-transfer-sequences) The `sequence()` function sets up matrix data movement from the host into the AIE compute cores, and back to the host after computation. It initializes Data Movement Accelerator (DMA) transfers, sets memory access patterns, and performs synchronization.

1. **Generating the Design:** The `my_matmul()` function triggers the code generation process and represents the main entry point of the design. The final print statement outputs the MLIR representation of the AIE array configuration.

In summary, this design leverages an AI Engine accelerator to accomplish matrix multiplication efficiently by breaking large matrices into smaller, manageable submatrices. The design uses parallelism, pipelining, and efficient data movement strategies to minimize computation time on the AI Engine array.

## Building and Running the Design

As configured, this design will set up an array of AIEs to perform matrix-matrix multiplication on a `bfloat16` data type, with `A`, `B` and `C` matrices all of size `512 &times; 512 &times; 512`. The tiling size is configured as `64 &times; 64` for `a`, `b`, and `c`.

You will need C++23 for `bfloat16_t` support in the `test.cpp`, which can be found in `g++-13`: [https://lindevs.com/install-g-on-ubuntu](https://lindevs.com/install-g-on-ubuntu)

To compile the design:

```
make
make matrixMultiplication.exe
```

To run the design:

```
make run
```

## Detailed Design Explanation

The configuration of the AI Engine array is described in the `aie2.py` file.
It is linked against a compute microkernel which is implemented in C++.
The following sections elaborate on each of the steps outlined in the high-level summary above.

> Note: The term "tile" has two distinct meanings in the following discussion that should be distinguishable from context:
>  * AIE tiles are components of the hardware, specifically Shim, Memory and Compute tiles.
>  * Matrix tiles are smaller sub-matrices of the larger input and output matrices.

### 1. Defining Matrix Dimensions and Data Types

In the first section of the code in `aie2.py`, we define the following constants:

| Matrix        | Size      | Submatrix Size (1.) | Vector Intrinsic Size (2.) |
|---------------|-----------|---------------------|-----------------------|
| `A` (Input)   | `M`  &times;  `K` | `m`  &times;  `k`           | `r`  &times;  `s`             |
| `B` (Input)   | `K`  &times;  `N` | `k`  &times;  `n`           | `s`  &times;  `t`             |
| `C` (Output)  | `M`  &times;  `N` | `m`  &times;  `n`           | `r`  &times;  `t`             |


The input and output matrix sizes are given by the user. We subdivide the input matrices `A`, `B` and the output matrix `C` into smaller, manageable "tiles" (or submatrices) at two levels:

1. **Tiling to Compute Core Submatrix Chunks:** The input and output matrices stream to/from the AIE compute cores in chunks of size of `m`&times;`k`, `k`&times;`n` and `n`&times;`m`. Tiling into these chunks allows each of the computation cores to concurrently work on distinct sub-sections of the input matrices in parallel, which improves performance. This also reduces on-chip memory requirements. The final result is re-assembled using the sub-matrix results of all cores.

    > This tiling occurs in the `sequence()` function describing the host-to-memory-tile transfer.
We describe it further below, in section *"5. Defining External Data Transfer Sequences"*.

1. **Tiling to Vector Intrinsic Size:** The AIE compute cores calculate the matrix multiplication using efficient "multiply-accumulate" vector intrinsic instructions (`MAC` instructions). These hardware instructions process very small blocks of the matrix: size `r`&times;`s` blocks of `A` and size `s`&times;`t` blocks of  `B`, producing an output of size `r`&times;`t` (`C`). 
    > This tiling occurs in the inner-AIE data movements. We describe it in the section *"3. Defining Data Movement Inside the NPU"*.

    > The vector intrinsic size is dictated by the hardware and the compute microkernel.

### 2. Constructing an AIE Array Configuration

In the next section of the code, we obtain handles to the components of the hardware. 

The Neural Processing Unit (NPU) is physically structured as an array of 6 rows and 4 columns. The lower two rows contain so-called "shim" and "memory" tiles, and the upper four rows are made up of AIE compute cores (AIEs):

1. **Shim tiles:** A single row of shim tiles on the bottom of the core array is responsible for interfacing with the external host for data movement. In our code, they are represented by a list: `[_0_ShimTile, _1_ShimTile, _2_ShimTile, _3_ShimTile]`

1. **Memory tiles:** A row of memory tiles with scratchpad memory is located above the shim tiles. These memory cores are responsible for staging and distributing the data during processing. In our code, they are represented by a list: `[_0_MemTile, _1_MemTile, _2_MemTile, _3_MemTile]`

1. **Compute tiles:** In each of the four columns, there are 4 rows of computation tiles above the memory tiles. This makes for a total of 16 computation cores, which in this design are configured to perform the matrix multiplication. In our code, they are represented by a list of lists, `cores`, showing their two-dimensional arrangement.

### 3. Defining Data Movement Inside the NPU: 

We use "ObjectFIFOs" to abstractly describe the data movement and synchronization between AIE Compute, Memory and Shim tiles. ObjectFIFOs present an interface that behaves like a First-In-First-Out queue. To achieve this, they take care of DMA configuration, acquiring and releasing locks, and managing buffers. 

There are several ObjectFIFOs used in this design, which are created using the `object_fifo()` Python binding:

1. Host &rightarrow; Memory Tiles: `inA_fifos`, `inB_fifos` move the input matrices from the external host (via the shim tiles) in row 0 to the memory tiles in row 1.

2. Memory Tiles &rightarrow; Compute Tiles: `memA_fifos`, `memB_fifos` move input data from the memory tiles in row 1 to the compute tiles in rows 2-5.

3. Compute Tiles &rightarrow; Memory Tiles &rightarrow; Host: Analogously, `memC_fifos` and `OutC_fifos` move the output data out from the compute cores to the memory tiles (`memC_fifos`) and from there out to the external host via the shim tiles (`OutC_fifos`).

Each of `inA_fifos`, `inB_fifos`, `OutC_fifos`, `memA_fifos`, `memB_fifos` and `memC_fifos` are Python dictionaries, containing a separate ObjectFIFO instance for each column of AIE compute cores in the array. The respective `*_names` lists contain the names of these ObjectFIFOs.

Of note is the `object_fifo_link()` operation. This operation establishes a connection between the `mem*` FIFOs and the `in*` and `outC` FIFOs. By linking ObjectFIFOs, the output received at one end of the source FIFO is fed as input into the ObjectFIFO listed as the destination.

<!-- 2. Creation of Object Fifos for Matrix A:

    * The input matrix A is streamed from the host to the AIE array using object fifos. `inA_fifos` and `memA_fifos` are dictionaries created to store the object fifos for input matrix A. `inA_fifo_names` and `memA_fifo_names` are lists storing the names of corresponding object fifos.
    * For each column `i` in the AIE array:
        * An object fifo `inA_fifos[inA_fifo_names[i]]` is created to connect the shim tile to the memory tile. The matrix A data is sent from the shim tile to the memory tile using `inA_fifos`.
        *  An object fifo `memA_fifos[memA_fifo_names[i]]` is created to connect the memory tile to the cores in column `i`. The submatrices of A are sent from the memory tile to each core in column `i` using `memA_fifos`.
        *  Then, `object_fifo_link()` establishes the connection between those two FIFOs, creating a data movement pipeline. -->


<!--
1. Creation of Object Fifos for Matrix B:

    * The input matrix B is streamed from the host to the AIE array using object fifos. `inB_fifos` and `memB_fifos` are dictionaries created to store the object fifos for input matrix B. `inB_fifo_names` and `memB_fifo_names` are lists storing the names of corresponding object fifos.
    * For each column `i` in the AIE array:
        * An object fifo `inB_fifos[inB_fifo_names[i]]` is created to connect the shim tile to the memory tile. The matrix B data is sent from the shim tile to the memory tile using `inB_fifos`.
        * An object fifo `memB_fifos[memB_fifo_names[i]]` is created to connect the memory tile to the cores in row `i`. The submatrices of B are sent from the memory tile to each core in row `i` using `memB_fifos`.
        *  Then, `object_fifo_link()` establishes the connection between those two FIFOs, creating a data movement pipeline.
-->

<!--
2.	Creation of Object Fifos for Matrix C:
    * The output matrix C is streamed from the AIE array to the host using object fifos. `outC_fifos` is a dictionary created to store the object fifos for output matrix C. `outC_fifo_names` is a list storing the names of corresponding object fifos.
    * For each column `i` in the AIE array:
        * An object fifo `memC_fifos[i][memC_fifo_names[i][j]]` is created for each row `j` to connect the cores to the memory tile. The results of the matrix multiplication are sent from each core to the memory tile using `memC_fifos`.
        * An object fifo `outC_fifos[outC_fifo_names[i]]` is created to connect the memory tile to the shim tile. The output matrix C data is sent from the memory tile to the shim tile using `outC_fifos`.
    -->

#### Tiling and Data Layout Transformations

We assume our data are stored in **row-major format** in the host's memory. For processing on the AIE compute cores, we need to transform the data layouts, such the above listed *sub-matrix tiles* are laid out contiguously in AIE compute core memory. Thankfully, AIE hardware has extensive support for transforming data using the DMAs as it is received and sent with zero cost. In the following, we will explain how we make use of this hardware feature to transform our data.

##### Tiling to Vector Intrinsic Size

The `memA_fifos` and `memB_fifos` receive sub-matrices of size `m`&times;`k` and `k`&times;`n`, respectively. The FIFOs translate those matrices from a row-major format into the `r`&times;`s`-sized and `s`&times;`t`-sized blocks required by the hardware's vector instrinsics before sending them into the compute cores memory.

For matrix A (`memA_fifos`), this transformation is expressed using the following wraps and strides as a list of tuples `(wrap, stride)`, given as arguments to the `object_fifo()` operation:
(Note that `//` denotes integer floor-division in Python.)

    
```python
    [
        (m // r, r * k),   # Pair 1
        (k // s, s),       # Pair 2
        (r, k),            # Pair 3
        (s, 1),            # Pair 4
    ]
```

Let us break down each component of this pattern. We do so back-to-front for ease of understanding:

* Pair 4: `(s, 1)`
    * This dimension represents the transfer of a single row of a `r`&times;`s`-sized tile (our target tile size after the transformation).
    * Wrap: `s` is the length of a row of a `r`&times;`s`-sized block in units of 4 bytes (i32 elements).
    * Stride: A stride of `1` retrieves contiguous elements.
* Pair 3: `(r, k)`
    * Together with the previous dimension, this dimenison represents the transfer of a single `r`&times;`s`-sized tile.
    * Wrap: `r` is the number of rows of a `r`&times;`s`-sized tile.
    * Stride: `k` is the stride between first element of each consecutive row along the `m` dimension, i.e. adding this stride to a memory address points to the element in the matrix directly below the original address. 
* Pair 2: `(k // s, s)`
    * Together with the previous dimensions, this dimension represents the transfer of one row of `r`&times;`s`-sized tiles, i.e. the first `k`&times;`s` elements of the input array.
    * Wrap: `k // s` is the number of `r`&times;`s`-sized tiles along the `k` (columns) dimension.
    * Stride: `s` is the stride between starting elements of consecutive blocks along the `k` dimension, i.e. adding this stridde to a memory address points to the same element in the `r`&times;`s`-sized block directly to the right of the block of the original address.
* Pair 1: `(m // r, r * k)`
    * Together with the previous dimensions, this dimension transfers the entire `m`&times;`k`-sized matrix as blocks of `r`&times;`s`-sized tiles.
    * Wrap: `m // r` is the number of `r`&times;`s`-sized blocks along the `m` (rows) dimension.
    * Stride: `r * k` is the stride between starting elements of consecutive blocks along the `m` dimension, i.e. adding this stride to a memory address points to the same element in the `r`&times;`s`-sized block directly below the block of the original address.

> You can use this [data layout visualizer](http://andreroesti.com/data-layout-viz/data_layout.html) to better understand data layout transformations expressed as wraps and strides.

The matrix B transformation (`memB_fifos`) is equivalent after substituting the correct dimensions (`k`&times;`n` instead of `m`&times;`k` and `s`&times;`t` isntead of `r`&times;`s`).

Analogously, the output matrix C is transformed back from `r`&times;`t`-sized blocks back into a row-major matrix of contiguous rows with size `m`&times;`n`.


### 4. Defining Core Computations

The `core_body()` function defines the computation that each core will perform.
We define a `core_body()` function for each compute core `i`, inside of which we do the following:

 * We acquire a slot in the output buffer into which we will produce the next `m`&times;`n`-tile of output in `memC_fifos`. We name the acquired buffer `elem_out`.
 * We zero out the acquired output slot, since it may contain stale results using `call(zero [elem_out])`.
 * `K // k` times, we:
    * We acquire the next `m`&times;`k`-tile of `A`, and the next `k`&times;`n` tile of `B` from ObjectFIFOs `memA_fifos[i]` and `memB_fifos[i]`, respectively, as `elem_in_a` and `elem_in_b`.
    * We call our compute microkernel (implemented in C++ and linked against this design) to perform the matrix multiplication calculation, with `call(matmul, [elem_in_a, elem_in_b, elem_out])`.
    The result is summed element-wise in `elem_out` together with previous iterations.
    * We release `elem_in_a` and `elem_in_b`.
* After the complete result for the current `m`&times;`n`-block has been calculated, we can release `elem_out`.

### 5. Defining External Data Transfer Sequences

The function signature of the `sequence()` function lists as its arguments all the external buffers from the host that we wish to read from or write to on the AI Engine's shim tiles. The body of this function describes how these buffers are transfered from and to the host, including tiling the input matrices into `m`&times;`k` and `k`&times;`n`-sized sub-matrices, and combining the `m`&times;`n`-sized output tiles into the larger output `M`&times;`N` matrix buffer.

* The `tile_row_block` variable segments the M (rows of A) into smaller chunks, each containing `rows_per_block` tile rows. This is done so the buffer descriptors (BDs) can be reused for efficient DMA transfers.
* For each column `i`:
    * For each `tile_row` in the current row block:
        * The DMA transfer function `ipu_dma_memcpy_nd` loads a segment of matrix A and matrix B data (submatrix a, submatrix b) from the host into the corresponding `inA_fifos` for the respective column, maintaining the appropriate strides and offsets.
        * Analogously to the data layout transformations described [further above](#tiling-and-data-layout-transformations) to translate a `m`&times;`k` matrix into blocks of `r`&times;`s`-submatrices, this transfer translates the input `M`&times;`K` and `K`&times;`N` matrices into submatrices of size `m`&times;`k` and `k`&times;`n`.
           > Note that data layout transformations in the `ipu_dma_memcpy_nd` operation are expressed in units of 4 bytes. This is why you will see all strides and the lowest-dimension length multiplied by a factor of `word_size_in` or `word_size_out` (to get the size in bytes) and then divide by four (to get teh size in units of 4 bytes). This discrepancy will be streamlined in future versions.
    * The DMA transfer function `ipu_dma_memcpy_nd` sends a segment of matrix C data (submatrix c) from the corresponding `outC_fifos` for the respective column, back to the host while maintaining the appropriate strides and offsets.
    * After completing DMA transfers for each column, `ipu_sync` is used to synchronize their completion.

## Compute Microkernels

This C++ code demonstrates how to implement matrix multiplication for different data types and operations using AIE (AI Engine) API and templates. The AI Engine is designed for efficient computation and data movement, especially for matrix multiplication-intensive machine learning workloads. The code has the following main components:

1. `matmul_scalar`: A scalar function that performs matrix multiplication for input matrices `a` and `b` and adds the result to matrix `c`. This function iterates through each row in matrix `a` and each column in matrix `b`, performing the multiplication of the corresponding elements and accumulating their sum to populate matrix `c`.

1. `matmul_vectorized` and `matmul_vectorized_2x2`: Vectorized matrix multiplication functions for different block sizes and input/output types for the AI Engine. These functions use the AIE API for efficient vectorized matrix multiplication, with support for various input and output tensor data types (e.g., int16, bfloat16).

1. `matmul_vectorized_4x4x4_i16_i16`, `matmul_vectorized_4x8x4_bf16_bf16`, and `matmul_vectorized_4x8x4_bf16_f32`: Helper functions for calling the corresponding `matmul_vectorized` functions with specific input and output types and block sizes.

1. Extern "C" interface functions: These functions provide a C-compatible interface to the main matrix multiplication functions, making it easier to call these functions from other languages or environments.

1. Zeroing functions: Functions like `zero_vectorized` and `zero_scalar` initialize the output matrix (`c_out`) with all zero values.

This code showcases efficient performance in matrix multiplication-intensive workloads and can be adapted for other types of inputs and operations as needed.
