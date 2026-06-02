<!---//===- README.md ---------------------------------------*- Markdown -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
// 
//===----------------------------------------------------------------------===//-->

# <ins>Section 2e - Programming for multiple cores</ins>

* [Section 2 - Data Movement (Object FIFOs)](../../section-2/)
    * [Section 2a - Introduction](../section-2a/)
    * [Section 2b - Key Object FIFO Patterns](../section-2b/)
    * [Section 2c - Data Layout Transformations](../section-2c/)
    * [Section 2d - Runtime Data Movement](../section-2d/)
    * Section 2e - Programming for multiple cores
    * [Section 2f - Practical Examples](../section-2f/)
    * [Section 2g - Data Movement Without Object FIFOs](../section-2g/)

-----

This section walks through scaling a single-Worker IRON design out to multiple Workers. We will start with the code in [aie2.py](./aie2.py) which contains a simple design running on a single Worker, and progressively turn it into the code in [aie2_multi.py](./aie2_multi.py) which contains the same design distributed across three Workers.

Both files are wrapped in `@iron.jit`, so calling each design from `_run_and_verify` JIT-compiles and runs end-to-end on the attached NPU; `--emit-mlir` prints the lowered MLIR for inspection.

In the first part of our design we set up the data movement using Object FIFOs. The simple design has a total of four Object FIFOs, two of which are created by forwarding data for an implicit copy. The Object FIFOs move objects of datatype `<48xi32>`. `of_in` brings data from external memory and is linked, through a Mem tile, to `of_in0` which brings data from the Mem tile to the Worker. For the output side, `of_out0` brings data from the Worker to the Mem tile where it is linked to `of_out` to bring the data out to external memory. The corresponding code is shown below:
```python
data_size = 48

# Define tensor types
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]

# Input data movement
of_in = ObjectFifo(data_ty, name="in")
of_in1 = of_in.cons().forward(obj_type=data_ty, name="in1")

# Output data movement
of_out1 = ObjectFifo(data_ty, name="out1")
of_out = of_out1.cons().forward(obj_type=data_ty, name="out")
```
For our scale out design we will keep using a single Mem tile, but we will increase the number of Workers to three. Now each Worker will receive objects of datatype `<16xi32>`. Data brought into the AIE array via `of_in` will be split into three Object FIFOs for each Worker. Similarly data produced by each Worker will be joined and sent to external memory through `of_out`. Please [see distribute and join patterns](../section-2b/03_Implicit_Copy/README.md) for more details. These changes result in the following code:
```python
n_workers = 3
data_size = 48
tile_size = data_size // 3

# Define tensor types
data_ty = np.ndarray[(data_size,), np.dtype[np.int32]]
tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]

# Input data movement
of_offsets = [tile_size * worker for worker in range(n_workers)]

of_in = ObjectFifo(data_ty, name="in")
of_ins = (
    of_in
    .cons()
    .split(
        of_offsets,
        obj_types=[tile_ty] * n_workers,
        names=[f"in{worker}" for worker in range(n_workers)],
    )
)

# Output data movement
of_out = ObjectFifo(data_ty, name="out")
of_outs = (
    of_out.prod().join(
        of_offsets,
        obj_types=[tile_ty] * n_workers,
        names=[f"out{worker}" for worker in range(n_workers)],
    )
)
```
The Worker of this simple design acquires one object of each Object FIFO, adds `1` to each entry of the incoming data, copies it to the object of the outgoing Object FIFO, then releases both objects:
```python
# Task for the core to perform
def core_fn(of_in, of_out):
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    for i in range_(tile_size):
        elem_out[i] = elem_in[i] + 1
    of_in.release(1)
    of_out.release(1)


# Create a worker to perform the task
my_worker = Worker(core_fn, [of_in1.cons(), of_out1.prod()])
```
For our larger design we create more Workers and select the input and output Object FIFOs for each Worker from the lists we made in the previous part:
```python
# Create workers to perform the tasks
workers = []
for worker in range(n_workers):
    workers.append(
        Worker(
            core_fn,
            [
                of_ins[worker].cons(),
                of_outs[worker].prod(),
            ],
        )
    )
```
Finally, in our simple design we write a runtime sequence to bring data to/from external memory and start our Worker:
```python
# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, b_out, _):
    rt.start(my_worker)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)
```
The runtime sequence remains largely unchanged for the larger design except that it has to start all three Workers:
```python
# Runtime operations to move data to/from the AIE-array
rt = Runtime()
with rt.sequence(data_ty, data_ty, data_ty) as (a_in, b_out, _):
    rt.start(*workers)
    rt.fill(of_in.prod(), a_in)
    rt.drain(of_out.cons(), b_out, wait=True)
```

To build and run the designs:
```bash
make run                # JIT-compile + run aie2.py on the attached NPU
make run-multi          # JIT-compile + run aie2_multi.py on the attached NPU
make emit-mlir          # write the lowered MLIR for aie2.py to build/aie.mlir
make emit-mlir-multi    # write the lowered MLIR for aie2_multi.py to build/aie_multi.mlir
```

-----
[[Prev - Section 2d](../section-2d/)] [[Up](..)] [[Next - Section 2f](../section-2f/)]
