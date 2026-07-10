<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# IRON Python API

IRON is the high-level Python interface for programming AMD Ryzen™ AI NPUs.
It exposes tile placement, data movement, and host runtime as Python objects
that compile to an optimized `xclbin` + instruction stream via the MLIR-AIE
toolchain.

## Top-level namespace (`iron`)

::: iron
    options:
      members:
        - jit
        - get_current_device
        - arange
        - zeros
        - Program
        - Runtime
        - Worker
        - In
        - Out

## Core abstractions

### Worker

::: iron.worker
    options:
      show_root_heading: false

### ObjectFifo / Data movement

::: iron.dataflow.objectfifo
    options:
      show_root_heading: false

::: iron.dataflow.flow
    options:
      show_root_heading: false

::: iron.dataflow.endpoint
    options:
      show_root_heading: false

### Buffer

::: iron.buffer
    options:
      show_root_heading: false

### Lock

::: iron.lock
    options:
      show_root_heading: false

### Control flow

::: iron.controlflow
    options:
      show_root_heading: false

### Runtime

::: iron.runtime.runtime
    options:
      show_root_heading: false

::: iron.runtime.task
    options:
      show_root_heading: false

::: iron.runtime.taskgroup
    options:
      show_root_heading: false

::: iron.runtime.dmatask
    options:
      show_root_heading: false

### Program

::: iron.program
    options:
      show_root_heading: false

### Data types

::: iron.dtype
    options:
      show_root_heading: false
