<!---//===- README.md --------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 3 - My First Program</ins>

<img align="right" width="500" height="250" src="../assets/binaryArtifacts.svg">

This section creates the first program that will run on the AIE-array. As shown in the figure on the right, we will have to create both binaries for the AIE-array (device) and CPU (host) parts. For the AIE-array, a structural description and kernel code is compiled into the AIE-array binaries: an XCLBIN file ("final.xclbin") and an instruction sequence ("inst.txt"). The host code ("test.exe") loads these AIE-array binaries and contains the test functionality.

For the AIE-array structural description we will combine what you learned in [section-1](../section-1) for defining a basic structural design in Python with the data movement part from [section-2](../section-2).

For the AIE kernel code, we will start with non-vectorized code that will run on the scalar processor part of an AIE. [Section-4](../section-4) will introduce how to vectorize a compute kernel to harvest the compute density of the AIE.

The host code can be written in either C++ (as shown in the figure) or in Python. We will also introduce some convenience utility libraries for typical test functionality and to simplify context and buffer creation when the [Xilinx RunTime (XRT)](https://github.com/Xilinx/XRT) is used, for instance in the [AMD XDNA Driver](https://github.com/amd/xdna-driver) for Ryzen™ AI devices.

<img align="right" width="410" height="84" src="../assets/vectorScalarMul.svg">

Throughout this section, a [vector scalar multiplication](../../programming_examples/basic/vector_scalar_mul/) (`c = a * factor`) will be used as an example. Vector scalar multiplication takes an input vector `a` and computes the output vector `c` by multiplying each element of `a` with a `factor`. In this example, the total vector size is set to 4096 (16b) that will processed in chunks of 1024.

This design is also available in the [programming_examples](../../programming_examples) of this repository. We will first introduce the AIE-array structural description, review the kernel code and then introduce the host code. Finally we will show how to run the design on Ryzen™ AI enabled hardware.

## AIE-array Structural Description

<img align="right" width="150" height="350" src="../assets/vectorScalarMulPhysicalDataFlow.svg">

The [aie2.py](../../programming_examples/basic/vector_scalar_mul/aie2.py) AIE-array structural description (see [section-1](../section-1)) deploys both a compute core (green) for the multiplication and a shimDMA (purple) for data movement of both input vector `a` and output vector `c` residing in external memory.

```python
# Device declaration - here using aie2 device NPU
@device(AIEDevice.npu1_1col)
def device_body():

    # Tile declarations
    ShimTile = tile(0, 0)
    ComputeTile2 = tile(0, 2)
```

We also need to declare that the compute core will run an external function: a kernel written in C++ that will be linked into the design as pre-compiled kernel (more details below). To get our initial design running on the AIE-array, we will run a generic version of the vector scalar multiply design here in this directory that is run on the scalar processor of the AIE. This local version will use `int32_t` datatype instead of the default `int16_t`for the [programming_examples version](../../programming_examples/basic/vector_scalar_mul/).

```python
        # Type declarations
        tensor_ty = np.ndarray[(4096,), np.dtype[np.int32]]
        tile_ty = np.ndarray[(1024,), np.dtype[np.int32]]
        scalar_ty = np.ndarray[(1,), np.dtype[np.int32]]
        
        # AIE Core Function declarations
        scale_scalar = external_func("vector_scalar_mul_aie_scalar",
            inputs=[tile_ty, tile_ty, scalar_ty, np.int32],
        )
```

Since the compute core can only access L1 memory, input data needs to be explicitly moved to (yellow arrow) and from (orange arrow) the L1 memory of the AIE. We will use the objectFIFO data movement primitive (introduced in [section-2](../section-2/)).

<img align="right" width="300" height="300" src="../assets/vector_scalar.svg">

This enables looking at the data movement in the AIE-array from a logical view where we deploy 3 objectFIFOs: `of_in` to bring in the vector `a`, `of_factor` to bring in the scalar factor, and `of_out` to move the output vector `c`, all using shimDMA. Note that the objects for `of_in` and `of_out` are declared to have the `tile_ty` type: 1024 int32 elements, while the `factor` is an object containing a single integer. All objectFIFOs are set up using a depth size of 2 to enable the concurrent execution to the Shim Tile and Compute Tile DMAs with the processing on the compute core.

```python
        # AIE-array data movement with object fifos
        of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tile_ty)
        of_factor = object_fifo("infactor", ShimTile, ComputeTile2, 2, scalar_ty)
        of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tile_ty)

```
We also need to set up the data movement to/from the AIE-array: configure n-dimensional DMA transfers in the shimDMAs to read/write to/from L3 external memory. For NPU, this is done with the `npu_dma_memcpy_nd` function (more details in [section 2-g](../section-2/section-2g)). Note that the n-dimensional transfer has a size of 4096 int32 elements and that the `metadata` argument  in the `npu_dma_memcpy_nd` needs by the corresponding objectFIFO python object or to match the `name` argument of the corresponding objectFIFO.
Note that for transfers into the AIE-array that we want to explicitly wait on with `dma_wait`, we must specify `issue_token=True` in order to ensure we have
a token to wait on. Tokens are generated automatically for `npu_dma_memcpy_nd`s in the opposite direction.

```python
        # To/from AIE-array data movement
        @runtime_sequence(tensor_ty, scalar_ty, tensor_ty)
        def sequence(A, F, C):
            npu_dma_memcpy_nd(metadata=of_in, bd_id=1, mem=A, sizes=[1, 1, 1, 4096], issue_token=True)
            npu_dma_memcpy_nd(metadata=of_factor, bd_id=2, mem=F, sizes=[1, 1, 1, 1], issue_token=True)
            npu_dma_memcpy_nd(metadata=of_out, bd_id=0, mem=C, sizes=[1, 1, 1, 4096])
            dma_wait(of_in, of_factor, of_out)
```

Finally, we need to configure how the compute core accesses the data moved to its L1 memory, in objectFIFO terminology: we need to program the acquire and release patterns of `of_in`, `of_factor` and `of_out`. Only a single factor is needed for the complete 4096 vector, while for every processing iteration on a sub-vector, we need to acquire an object of 1024 integers to read from `of_in` and one similar sized object from `of_out`. Then we call our previously declared external function with the acquired objects as operands. After the vector scalar operation, we need to release both objects to their respective `of_in` and `of_out` objectFIFO. Finally, after the 4 sub-vector iterations, we release the `of_factor` objectFIFO.

This access and execute pattern runs on the AIE compute core `ComputeTile2` and needs to get linked against the precompiled external function `scale.o`. We run this pattern in a very large loop to enable enqueuing multiple rounds of vector scalar multiply work from the host code.

```python
        @core(ComputeTile2, "scale.o")
        def core_body():
            # Effective while(1)
            for _ in range_(sys.maxsize):
                elem_factor = of_factor.acquire(ObjectFifoPort.Consume, 1)
                # Number of sub-vector "tile" iterations
                for _ in range_(4):
                    elem_out = of_out.acquire(ObjectFifoPort.Produce, 1)
                    elem_in = of_in.acquire(ObjectFifoPort.Consume, 1)
                    scale_scalar(elem_in, elem_out, elem_factor, 1024)
                    of_in.release(ObjectFifoPort.Consume, 1)
                    of_out.release(ObjectFifoPort.Produce, 1)
                of_factor.release(ObjectFifoPort.Consume, 1)
```

## Kernel Code

We can program the AIE compute core using C++ code and compile it with the selected single-core AIE compiler into a kernel object file. For our local version of vector scalar multiply, we will use a generic implementation of the `scale.cc` source (called [vector_scalar_mul.cc](./vector_scalar_mul.cc)) that can run on the scalar processor part of the AIE. The `vector_scalar_mul_aie_scalar` function processes one data element at a time, taking advantage of AIE scalar datapath to load, multiply and store data elements.

```c
void vector_scalar_mul_aie_scalar(int32_t *a_in, int32_t *c_out,
                                  int32_t *factor, int32_t N) {
  for (int i = 0; i < N; i++) {
    c[i] = *factor * a[i];
  }
}
```

[Section-4](../section-4/) will introduce how to exploit the compute dense vector processor.

Note that since the scalar factor is communicated through an object, it is provided as an array of size one to the C++ kernel code and hence needs to be dereferenced.

## Host Code

The host code acts as an environment setup and testbench for the Vector Scalar Multiplication design example. The code is responsible for loading the compiled XCLBIN file, configuring the AIE module, providing input data, and kick off the execution of the AIE design on the NPU. After running, it verifies the results and optionally outputs trace data (to be covered in [section-4c](../section-4/section-4c/)). Both C++ [test.cpp](./test.cpp) and Python [test.py](./test.py) variants of this code are available.

For convenience, a set of test utilities support common elements of command line parsing, the XRT-based environment setup and testbench functionality: [test_utils.h](../../runtime_lib/test_lib/test_utils.h) or [test.py](../../python/utils/test.py).   

The host code contains the following elements:

1. *Parse program arguments and set up constants*: the host code typically requires the following 3 arguments: 
    * `-x` the XCLBIN file
    * `-k` kernel name (with default name "MLIR_AIE")
    * `-i` the instruction sequence file as arguments 
    
    This is because it is its task to load those files and set the kernel name. Both the XCLBIN and instruction sequence are generated when compiling the AIE-array structural description and kernel code with `aiecc.py`.

1. *Read instruction sequence*: load the instruction sequence from the specified file in memory

1. *Create XRT environment*: so that we can use the XRT runtime

1. *Create XRT buffer objects* for the instruction sequence, inputs (vector `a` and `factor`) and output (vector `c`). Note that the `kernel.group_id(<number>)` needs to match the order of `def sequence(A, F, C):` in the data movement to/from the AIE-array of python AIE-array structural description, starting with ID number 2 for the first sequence argument and then incrementing by 1. This mapping is described as well in the [python utils documentation](../../python/utils/README.md#configure-shimdma). 

1. *Initialize and synchronize*: host to device XRT buffer objects

1. *Run on AIE and synchronize*: Execute the kernel, wait to finish, and synchronize device to host XRT buffer objects

1. *Run testbench checks*: Compare device results to reference and report test status


## Running the Program

To compile the design and C++ testbench:

```sh
make
```

To run the design:

```sh
make run
```

### Python Testbench

To compile the design and run the Python testbench:

```sh
make
```

To run the design:

```sh
make run_py
```
-----
[[Prev - Section 2](../section-2/)] [[Top](..)] [[Next - Section 4](../section-4/)]
