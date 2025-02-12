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

The `fill()` operation is used to fill an `in_fifo` ObjectFifoHandle of type producer with data from a `source` runtime buffer. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def fill(
        self,
        in_fifo: ObjectFifoHandle,
        source: RuntimeData,
        tap: TensorAccessPattern | None = None,
        task_group: RuntimeTaskGroup | None = None,
        wait: bool = False,
        placement: PlacementTile = AnyShimTile,
    )
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `placement` Shim tile can also be explicitly specified, otherwise the compiler will choose one based on the placement algorithm.

The `drain()` operation is used to fill an ObjectFifoHandle of type consumer of data and write that data to a runtime buffer. It is shown below and defined in [runtime.py](../../../python/iron/runtime/runtime.py):
```python
def drain(
    self,
    out_fifo: ObjectFifoHandle,
    dest: RuntimeData,
    tap: TensorAccessPattern | None = None,
    task_group: RuntimeTaskGroup | None = None,
    wait: bool = False,
    placement: PlacementTile = AnyShimTile,
)
```
When the `wait` input is set to `True` this operation will be waited upon, i.e., a token will be produced when the operation is finished that a controller is waiting on. A `placement` Shim tile can also be explicitly specified, otherwise the compiler will choose one based on the placement algorithm.

It is possible to reconfigure the DMAs in the AIE array at runtime to change the configuration of the data or to reuse some of the BDs, which can be very interesting as they are a limited resource. To facilitate this reconfiguration step, IRON introduces `RuntimeTaskGroups` which can be created using the `task_group()` function as defined in [runtime.py](../../../python/iron/runtime/runtime.py).

`RuntimeTasks` defined within the task group code region will be appended to the runtime sequence defined by that task group and executed as a single configuration of the runtime at time `t`. After the task group finishes at time `t+1`, the `RuntimeTasks` will be freed and the next task group will start. To mark the end of a task group code region, the `finish_task_group()` is used.

-----
[[Up](./README.md)]
