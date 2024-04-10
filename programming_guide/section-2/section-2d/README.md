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

The final part of this data movement section will focus on the process of taking code written for a single core and transforming it into a design with multiple cores relatively quickly. For this we will start with the code from ???.py which is ... on a single core, running on compute tile ???, and transform it into a design that uses ??? cores. This final result is available in ???.py. (TODO: complete with mini example)


```
ShimTile = tile(0, 0)
MemTile = tile(0, 1)
ComputeTile2 = tile(0, 2)
```

```
n_cores = 2

ShimTile = tile(0, 0)
MemTile = tile(0, 1)
cores = [tile(0, 2 + i) for i in range(n_cores)]
```

The Object FIFO primitive and its functions work well with this stype of coding as we can take the intialization of a single Object FIFO,
```
memRef_8_ty = T.memref(8, T.i32())
buffer_depth = 2

of_in1 = object_fifo("in1", MemTile, ComputeTile2, buffer_depth, memRef_8_ty)
```
then use a for loop to create many of them at once:
```
n_cores = 2

inA_fifo_names = [f"memA{i}" for i in range(n_cores)]   # list of object FIFO names
inA_fifos = {}                                          # map of names to object FIFOs

for i in range(n_cores):
    inA_fifos[inA_fifo_names[i]] = object_fifo(
        inA_fifo_names[i], MemTile, cores[i], buffer_depth, memRef_A_ty
    )
```

We can continue to apply the same method to transform the code using a single core,
```
@core(ComputeTile2)
def core_body():
    # Effective while(1)
    for _ in for_(8):
        elem_in = of_in1.acquire(ObjectFifoPort.Consume, 1)
        elem_out = of_out1.acquire(ObjectFifoPort.Produce, 1)
        for i in for_(8):
            v0 = memref.load(elem_in, [i])
            v1 = arith.addi(v0, arith.constant(1, T.i32()))
            memref.store(v1, elem_out, [i])
            yield_([])
        of_in1.release(ObjectFifoPort.Consume, 1)
        of_out1.release(ObjectFifoPort.Produce, 1)
        yield_([])
```
to a design with multiple cores:
```
for i in range(n_cores):
    # Compute tile i
    @core(cores[i], "add.o")
    def core_body():
        for _ in for_(0xFFFFFFFF):
            for _ in for_(tiles):
                elem_out = outC_fifos[outC_fifo_names[i]].acquire(
                    ObjectFifoPort.Produce, 1
                )
                elem_in_a = inA_fifos[inA_fifo_names[i]].acquire(
                    ObjectFifoPort.Consume, 1
                )
                elem_in_b = inB_fifos[inB_fifo_names[i]].acquire(
                    ObjectFifoPort.Consume, 1
                )

                call(
                    eltwise_add_bf16_vector,
                    [elem_in_a, elem_in_b, elem_out],
                )
                inA_fifos[inA_fifo_names[i]].release(
                    ObjectFifoPort.Consume, 1
                )
                inB_fifos[inB_fifo_names[i]].release(
                    ObjectFifoPort.Consume, 1
                )
                outC_fifos[outC_fifo_names[i]].release(
                    ObjectFifoPort.Produce, 1
                )
                yield_([])
            yield_([])
```