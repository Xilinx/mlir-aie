<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Inlining external kernels into the core

IRON keeps control loops in Python and compute kernels in C++, so a compute core
issues a `func.call` into the kernel once per tile. In a tight loop that per-call
overhead accumulates (see [issue #3396](https://github.com/Xilinx/mlir-aie/issues/3396)).

`ExternalFunction(inline=True)` compiles the kernel to `alwaysinline` LLVM IR
(`.ll`) instead of an object file. `aiecc` `llvm-link`s that IR into the core's
LLVM module before `opt`/`llc` and the always-inliner folds the body in, so there
is **no surviving `func.call`** and **no separately object-linked kernel `.o`**.
The `.ll`/`.bc` link artifacts are routed to this IR-merge path and excluded from
object linking (linker script `INPUT()` / BCF `_include`), so each symbol is
merged exactly once. This is the Peano path only — the Chess front-end cannot
`llvm-link`.

Usage is a single keyword:

```python
from aie.iron import ExternalFunction

add_one = ExternalFunction(
    "add_one",
    source_string=...,
    arg_types=[...],
    inline=True,          # <-- fold the kernel body into the core
)
```

## Microbenchmark

`inline_kernel.py` runs a deliberately call-heavy design (a 16-element `add_one`
invoked once per tile over a large tensor) both object-linked and inlined, checks
the results are identical, and prints the host-visible latency of each:

```bash
python inline_kernel.py --num-elements 16384 --iters 50
```

It reports end-to-end host latency (launch + DMA + compute), not isolated on-core
cycles. Both variants move identical data, so the object−inline *delta* isolates
the on-core cost of the `func.call`s (DMA cancels). For cycle-accurate call
overhead, bracket the kernel loop with the AIE trace (`event0`/`event1`).

### Measured (Strix Halo, aie2p / npu2)

`add_one` over a 16-element tile, one call per tile, best of 3 runs (iters=200):

| calls / iter | object-link | inline    | speedup |
|-------------:|------------:|----------:|--------:|
|          256 |    0.244 ms |  0.209 ms |  1.17x  |
|         1024 |    0.340 ms |  0.261 ms |  1.30x  |
|         4096 |    0.970 ms |  0.452 ms |  2.15x  |
|        16384 |    2.894 ms |  1.488 ms |  1.95x  |

The object−inline delta grows with the number of calls — the signature of
per-call overhead being removed. Inspecting the linked core LLVM IR confirms the
mechanism: the object build keeps a `call @add_one` per (unrolled) tile, while the
inline build has **zero** surviving calls and no separate `add_one.o`.

## Constraints

- Peano front-end only (not `use_chess=True`).
- `inline=True` is incompatible with `symbol_prefix` (an inline kernel is emitted
  as LLVM IR and cannot be symbol-renamed); combining them raises a clear error.
