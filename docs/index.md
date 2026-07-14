---
hide:
  - toc
  - navigation
---

<!-- Copyright (C) 2024-2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# IRON / MLIR-AIE

**Close-to-metal Python programming for AMD Ryzen™ AI NPUs.**

IRON is an open-source toolkit that lets performance engineers write Python code
that compiles directly to the AI Engine array inside AMD Ryzen™ AI processors —
with full control over tile placement, data movement, and vectorized compute.

<div class="iron-cards" markdown>

<a class="iron-card" href="getting-started/" markdown>
<span class="iron-card-icon">🚀</span>
<span class="iron-card-title">Get Started</span>
<span class="iron-card-desc">Install IRON, build and run your first NPU design.</span>
</a>

<a class="iron-card" href="programming_guide/" markdown>
<span class="iron-card-icon">📖</span>
<span class="iron-card-title">Programming Guide</span>
<span class="iron-card-desc">Tiles, ObjectFifos, data movement, vectorization, and ML examples.</span>
</a>

<a class="iron-card" href="api/" markdown>
<span class="iron-card-icon">🐍</span>
<span class="iron-card-title">API &amp; Internals</span>
<span class="iron-card-desc">Full reference for iron, taplib, the kernel library, and the MLIR dialects.</span>
</a>

<a class="iron-card" href="programming_guide/mini_tutorial/" markdown>
<span class="iron-card-icon">⚡</span>
<span class="iron-card-title">Mini Tutorial</span>
<span class="iron-card-desc">Five short exercises — a working NPU design in minutes.</span>
</a>

</div>

---

## What is IRON?

The NPU inside Ryzen™ AI is a 2D array of **AI Engine tiles** — small, fast,
vector-capable cores connected by programmable stream switches and DMA engines.
IRON exposes that architecture directly in Python:

```python
import aie.iron as iron
from aie.iron import In, Out, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
import numpy as np

@iron.jit
def vector_add_one(a_in: In, b_out: Out):
    # Stream data from host → compute tile → host
    of_in  = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="in")
    of_out = ObjectFifo(np.ndarray[(1024,), np.dtype[np.int32]], name="out")

    def core_fn(of_in, of_out):
        ai = of_in.acquire(1)
        bo = of_out.acquire(1)
        for i in range_(1024):
            bo[i] = ai[i] + 1
        of_in.release(1)
        of_out.release(1)

    w = Worker(core_fn, [of_in.cons(), of_out.prod()])

    rt = Runtime()
    with rt.sequence(np.ndarray[(1024,), np.dtype[np.int32]],
                     np.ndarray[(1024,), np.dtype[np.int32]]) as (a, b):
        rt.start(w)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()

a = iron.arange(1024, dtype=np.int32, device="npu")
b = iron.zeros(1024,  dtype=np.int32, device="npu")
vector_add_one(a, b)   # JIT-compiles on first call, cached after
```

`@iron.jit` compiles your design to an `xclbin` + instruction stream using the
LLVM/MLIR-based [Peano](https://github.com/Xilinx/llvm-aie) compiler and runs it
on the attached NPU. Subsequent calls hit a cache.

---

## Architecture

<figure markdown>
  ![IRON software stack](assets/images/iron_stack.svg){ width=460 }
  <figcaption>From a Python design to running on the NPU — IRON lowers through the MLIR-AIE compiler and Peano, then XRT loads it onto the AI Engine tile array.</figcaption>
</figure>

