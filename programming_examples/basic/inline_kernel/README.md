<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Inlining external kernels into the core

IRON programs tend to keep control loops in Python and compute kernels in C++,
so a compute core might issues a `func.call` into the kernel once per tile or in
other tight loops, and that per-call overhead accumulates.

`ExternalFunction(inline=True)` compiles the kernel to `alwaysinline` LLVM IR
(`.ll`) instead of an object file. `aiecc` `llvm-link`s that IR into the core's
LLVM module before `opt`/`llc` and the always-inliner folds the body in, so there
is no `func.call` and no separately object-linked kernel `.o`.

Usage:

```python
from aie.iron import ExternalFunction

add_one = ExternalFunction(
    "add_one",
    source_string=...,
    arg_types=[...],
    inline=True,          # <-- fold the kernel body into the core
)
```

`inline=True` declares the kernel with `link_with_mode = "merge"`:

```mlir
func.func private @add_one(...) attributes {
  link_with = "add_one.ll",
  link_with_mode = "merge"    // llvm-link me into the core module
}
```

`aiecc` (`aie-assign-core-link-files` pass) sorts each core's artifacts into `link_files`
(object-linked) and `link_merge_files` (IR-merged), and only the former reach
the linker, so each symbol is included exactly once.

## Microbenchmark

`inline_kernel.py` runs a deliberately call-heavy design (a 16-element `add_one`
invoked once per tile over a large tensor) both object-linked and inlined over
the same input, checks the two outputs are identical, and reports the cost of
each:

```bash
python inline_kernel.py --num-elements 16384 --iters 50
```

Timing uses `aie.utils.benchmark.run_iters`. It reports **on-NPU time**,
captured around `kernel.wait()`, separately from end-to-end host latency;
the example quotes the NPU figure because excluding launch overhead is what makes the per-call delta legible. `warmup=1` absorbs the
JIT compile.

This is still not a cycle count. Both variants move identical data, so the DMA
cost cancels in the object−inline *difference* but is present in each absolute
number. For cycle-accurate call overhead, bracket the kernel loop with the AIE
trace (`event0`/`event1`).

### Example

`add_one` over a 16-element tile, one call per tile. Absolute times depend on the
machine, the toolchain and the run:

| calls / iter | inline speedup |
|-------------:|---------------:|
|          256 |          ~1.3x |
|         1024 |          ~1.5x |
|         4096 |          ~1.8x |
|        16384 |          ~2.1x |

Reproduce with (`calls/iter` is `num-elements / 16`):

```bash
for n in 4096 16384 65536 262144; do
    python inline_kernel.py --num-elements $n --iters 200
done
```

In this example the speedup rises with call count because of the per-call
overhead being removed. Expect run-to-run noise at the small sizes.

Inspecting the linked core LLVM IR confirms the mechanism: the object build
keeps a `call @add_one` per (unrolled) tile, while the inline build has zero
surviving calls and no separate `add_one.o`.

## Constraints

- Peano front-end only (not `use_chess=True`).
- `inline=True` is incompatible with `symbol_prefix` (an inline kernel is emitted
  as LLVM IR and cannot be symbol-renamed); combining them raises an error.
- With `inline=True`, an explicit `object_file_name` must end in `.ll` or `.bc`.
