---
name: aie-code-creator
description: Write efficient IRON Python designs and C++ AIE kernels for AMD XDNA NPUs (AIE2 / AIE2P architectures, e.g., Ryzen AI Phoenix, Hawk, Strix, Krackan Point). Use whenever the user asks about IRON, ObjectFifo, Worker/Program/Runtime, NPU programming, AIE kernels, MLIR-AIE, iron.jit, iron.kernels or iron.algorithms, CompileTime/In/Out design signatures, vectorizing for AIE, MMUL or mac_dims, bfloat16/int8 on NPU, NPU1/NPU2 devices, AIE_PREPARE_FOR_PIPELINING, or wants help writing/debugging a design that targets AMD's XDNA NPU — even if they don't name "IRON" explicitly.
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->


# AIE Code Creator — AMD XDNA NPU Programming

This skill teaches you how to write efficient designs for AMD XDNA NPUs (AIE2 in Phoenix/Hawk, AIE2P in Strix/Krackan) using the IRON Python API and write the C++ AIE kernels that run on the compute tiles.

An IRON design has two halves:

1. **Structural / data-movement code (Python)** — describes the AIE-array topology: which tiles run which tasks, what `ObjectFifo`s connect them, and what the host-side `Runtime` does to feed/drain data.
2. **Compute kernel code (C++)** — the actual functions that execute on each compute tile, written against the AIE API (`aie::vector`, `aie::mmul`, ...) and compiled into `.o` files referenced by the Python design.

The design is compiled to MLIR, then linked against the C++ kernel objects and a host runtime to produce an `xclbin` that runs on the NPU.

**You often don't have to write either half from scratch.** mlir-aie ships `aie.iron.kernels` (maintained, vectorized kernels for element-wise ops, reductions, activations, matmul, conv2d, vision) and `aie.iron.algorithms` (whole-design templates like `transform_parallel` and `reduce` that build the fifos, workers, and runtime for you). A multi-core bf16 element-wise design is ~15 lines of Python and zero C++. Start there and drop to hand-written code only where nothing fits — that's how you get a correct baseline fast, which is the thing worth optimizing later. See [`references/builtin_kernels.md`](references/builtin_kernels.md).

**Two libraries, one boundary.** The IRON API, `@iron.jit`, `aie.iron.kernels`, `aie.iron.algorithms`, and the `aie.iron` / `aie.helpers` packages all ship in the base **`Xilinx/mlir-aie`** toolchain (the `mlir_aie` wheel) — everything in this skill runs on that alone, *except* where noted. The **`amd/IRON`** repo is a separate downstream operator library (package `iron`, e.g. `iron.common`, `iron.operators`); its helpers — `MLIROperator`, `AIEContext`, `run_test`, `verify_buffer`, `AIERuntimeArgSpec` — are **not** in the wheel and require an `amd/IRON` checkout. When you use one of those, say so, and prefer wheel-only equivalents unless the user is working inside `amd/IRON`.

**Targets mlir-aie 1.4.0.** Version 1.4.0 reworked the `Runtime`/`Program` API into an eager-callback style (`Runtime(seq_fn, fn_args)`, `fill`/`drain` as `ObjectFifoHandle` methods, `Program(device, rt, workers=[...])`) — a breaking change from 1.3.x's `Runtime()`/`rt.sequence()`/`rt.fill()`/`rt.start()` style. This skill's wheel-only examples assume 1.4.0 throughout; older pinned versions are out of scope. The one deliberate exception is the `amd/IRON` `MLIROperator` pattern (see `complete_examples.md` Example 8), which still pins an older wheel and intentionally keeps the old Runtime API — don't "fix" that snippet to the new style.

Three other 1.3.x names are gone in 1.4.0 and show up in stale examples across the internet: **`aie.iron.placers` / `SequentialPlacer`** (placement is now the `--aie-place-tiles` compiler pass; `resolve_program()` takes no placer), **`Worker(placement=...)`** (the kwarg is `tile=`), and **`NpuTensor.cpu()`** (use `.numpy()`, or `.to("cpu")` to move residency). If you catch yourself writing any of them, you're pattern-matching on old code.

## Defaults

If the user does not specify, assume:

- **Kernel source**: a built-in from `aie.iron.kernels` if one covers the op and dtype; hand-written C++ only otherwise.
- **API level**: High-level — an `aie.iron.algorithms` template if the topology fits, else `Program` / `Worker` / `ObjectFifo`. Let the compiler place tiles.
- **Target**: current device (auto-detect via `iron.get_current_device()`); if you must pick, default to `NPU2` (AIE2P).
- **Data type**: `bfloat16` (best balance of accuracy and throughput on AIE).
- **Parallelism**: multi-core (data-parallel split across all available compute tiles in 1+ columns).

Drop to the lower-level API only when the user needs a hand-scripted DMA channel body, a custom flow not covered by `iron.Flow`/`iron.PacketFlow`, or is porting an existing MLIR-AIE example 1:1.

## Workflow when generating a design

**1. Identify the shape of the computation**: element-wise, reduction, matmul/conv, stencil/sliding-window, pipeline, broadcast?

**2. Check whether a built-in already covers it** ([`references/builtin_kernels.md`](references/builtin_kernels.md)):

- Is the compute in `aie.iron.kernels`? (`relu`, `add`, `mul`, `scale`, `softmax`, `gelu`, `silu`, `reduce_add/min/max`, `mm`, `mv`, `cascade_mm`, `conv2dk1/k3`, vision ops, …)
- Does the topology match an `aie.iron.algorithms` template? (`transform`, `transform_parallel`, `transform_binary`, `transform_parallel_binary`, `reduce`, `for_each`, the `row_at_a_time`/`sliding_3row` conv pipelines)

If both, the design is a few lines inside an `@iron.jit` function and **you are done** — skip to step 6. Check dtype and tile-size support before committing: several kernels are bf16-only or fixed-tile.

**3. If the topology doesn't fit a template**, pick the matching skeleton from [`references/patterns.md`](references/patterns.md) and build the `ObjectFifo`/`Worker`/`Runtime` graph explicitly — but still pass a `kernels.*` kernel into it if one exists.

**4. If no built-in kernel fits**, write the C++: pick a template from [`references/kernel_intrinsics.md`](references/kernel_intrinsics.md), then wire it in via `Kernel(...)` or `ExternalFunction(...)`. Argument types must match the `extern "C"` signature exactly.

**5. Choose data sizes** that respect [hardware limits and divisibility constraints](references/architecture.md). For MMUL, read the geometry off the kernel (`kernels.mm(...).mac_dims`) rather than hardcoding `(r, s, t)` — it differs by architecture and dtype.

**6. Verify against pitfalls** in [`references/pitfalls.md`](references/pitfalls.md) before declaring done.

When asked to produce a complete design, deliver the Python design file, a short build/run snippet, and — only if you had to hand-write one — the C++ kernel file. Say explicitly when you used a built-in kernel instead of writing one, so the user knows where the compute actually comes from.

## Reference files

Load the ones you need; don't pre-read all of them.

| File | When to read |
|------|--------------|
| [`references/builtin_kernels.md`](references/builtin_kernels.md) | **Read first.** `aie.iron.kernels` catalog, `aie.iron.algorithms` templates, `In`/`Out`/`CompileTime[T]` jit signatures, `.mac_dims` |
| [`references/architecture.md`](references/architecture.md) | Hardware limits, data types & vector widths, tile layout, NPU1 vs NPU2 |
| [`references/python_api.md`](references/python_api.md) | `Program`, `Worker`, `ObjectFifo`, `Runtime`, `Kernel`, `Buffer`, `iron.jit` signatures and examples |
| [`references/patterns.md`](references/patterns.md) | Copy-ready Python skeletons: single-core, multi-core data-parallel, broadcast, split/join, producer-consumer pipeline, reduction, RTP+barrier |
| [`references/kernel_intrinsics.md`](references/kernel_intrinsics.md) | C++ kernel cheatsheet: `aie::add/mul/mac`, `aie::load_v/store_v`, MMUL/accumulator, reductions, broadcast — with template skeletons |
| [`programming_guide/section-3/README.md`](../../programming_guide/section-3/README.md) / [`section-4b`](../../programming_guide/section-4/section-4b/README.md) | Building/running a design end-to-end, enabling trace |
| [`references/pitfalls.md`](references/pitfalls.md) | Anti-patterns: deadlocks, bad placement, alignment, restrict misuse, divisibility, RTP races |
| [`references/low_level_api.md`](references/low_level_api.md) | Lower-level API (`@device`, `@core`, explicit `tile(col,row)`) — only when the high-level API is insufficient |
| [`references/complete_examples.md`](references/complete_examples.md) | Nine full runnable designs: library form (no C++) → passthrough → inline `source_string` kernel → precompiled Kernel → TAP/task_group → split/join → ReLU jit → two-stage pipeline → MLIROperator pattern (last one is `amd/IRON`-only; see below) |
| [`assets/python_design_skeleton.py`](assets/python_design_skeleton.py) | Drop-in template for a JIT-compiled multi-core design |
| [`assets/kernel_skeleton.cc`](assets/kernel_skeleton.cc) | Drop-in template for a templated, vectorized C++ kernel |
| [`assets/architecture-diagram.txt`](assets/architecture-diagram.txt) | ASCII layout of NPU1 (4-col AIE2) and NPU2 (8-col AIE2P) tile arrays with placement sentinels |

## C++ kernel coding rules (always apply, unless told otherwise)

- **C++17**, no exceptions, no RTTI, freestanding-friendly.
- **Fixed-width integer types**: `int8_t`, `int16_t`, `int32_t`, `uint8_t`, ... — never bare `int`.
- **Templated** on data type and on size constants when reasonable; provide thin `extern "C"` wrappers with concrete types for MLIR linkage.
- **Vectorized** via the AIE API (`aie::vector`, `aie::mmul`, `aie::add/mul/mac`), **not** compiler-specific intrinsics like `v32bfloat16` or `broadcast_to_v16int32` unless the AIE API has no equivalent.
- **`__restrict` on every pointer** passed into the hot loop. Without it, the modulo scheduler will refuse to pipeline.
- **Loop annotations** from `aie_kernel_utils.h` — `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_MIN_ITERATION_COUNT(n)`, `AIE_LOOP_RANGE(min, max)`, `AIE_LOOP_UNROLL(n)`, `AIE_LOOP_UNROLL_FULL`. Under the default Peano/AIECC flow, `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_FLATTEN` are **no-ops** — only Chess honors them. `AIE_LOOP_MIN_ITERATION_COUNT(n)`, `AIE_LOOP_RANGE(min, max)`, and `AIE_LOOP_UNROLL_FULL` are real under **both** backends. Keep all of them in (they cost nothing under Chess), but never rely on `AIE_PREPARE_FOR_PIPELINING` alone. For a small, fixed-trip-count inner loop that branches or switches on the loop index (e.g. a 3×3 window), reach for `AIE_LOOP_UNROLL_FULL` rather than `AIE_LOOP_RANGE` — a live branch on the loop variable can block the modulo scheduler regardless of iteration-count hints (see `references/pitfalls.md`).
- **`event0()` / `event1()`** around the hot region for trace-based profiling.
- **`constexpr`** for vector widths and loop trip-count math.
- **Saturation/rounding** explicitly set for integer kernels that can overflow: `aie::set_saturation(aie::saturation_mode::saturate)`.

## Python design coding rules

- Prefer `aie.iron.kernels` / `aie.iron.algorithms` over rebuilding equivalent compute or topology by hand (see [`references/builtin_kernels.md`](references/builtin_kernels.md)).
- Declare `@iron.jit` tensor arguments with `In` / `Out` / `InOut`, and specialization knobs as keyword-only `CompileTime[T]` parameters. An unannotated non-tensor parameter with a default is rejected at decoration time — there's no runtime-scalar plumbing, so the default would be baked in and any per-call override silently ignored.
- Use `range_` from `aie.iron.controlflow`, **not** Python's `range`, for loops that should be emitted as AIE loops. Plain `range` unrolls at design-build time and will explode your code size or fail.
- Type tensors with `np.ndarray[(N,), np.dtype[T]]` — the shape and dtype must match what the kernel expects.
- ObjectFifo `depth` must be at least the producer-consumer working set; default to `2` (ping-pong); raise for deeper pipelines.
- Every `acquire(n)` must be paired with `release(n)`. Mismatches deadlock silently.
- The argument list passed to `Worker(fn, [...])` must match the order of `fn`'s parameters.
- Use multi-core (split/join or per-column workers) whenever the problem trivially data-parallelizes — a single Worker leaves 7 cores idle on NPU2.

## Where this skill stops

This skill covers getting a **correct, reasonable first design** written. Upstream mlir-aie
ships a phased skill family (`mlir-aie/skills/`) for what comes before and after — don't
re-derive their methodology here, point at it:

| Phase | Skill | Covers |
|-------|-------|--------|
| Prepare the model | `aie-model-baseline` | Quantization scheme, ONNX export, bit-exact numeric oracle |
| Validate pre-hardware | `aie-dataflow-presim` | Threaded ObjectFifo mock for deadlock/depth bugs; prove novel decompositions in numpy first |
| First hardware bring-up | `aie-hw-bringup` | Block-by-block bring-up against the oracle, methodical bisection |
| Optimize (micro) | `aie-kernel-opt` | Make one compiled kernel faster — measure first, then the lever catalog |
| Optimize (macro) | `aie-dataflow-opt` | NOOP-ablation ranking, placement/overlays, DMA bandwidth modeling |

Two habits from those skills are worth carrying into design time, because they're cheap now
and expensive later: **validate one block at a time against a reference** rather than wiring
the whole chain and hoping, and **write the smallest standalone design that exercises a
mechanism you haven't built before** instead of debugging it inside the full design.

## Quick triage cheatsheet

| User says | Reach for |
|-----------|-----------|
| "Element-wise op (add/mul/relu/scale) on a vector" | [`builtin_kernels.md`](references/builtin_kernels.md) `kernels.add/mul/relu/scale` + `transform_parallel` — only fall back to [`patterns.md`](references/patterns.md) §Element-wise + [`kernel_intrinsics.md`](references/kernel_intrinsics.md) §Element-wise if the dtype/tile isn't supported |
| "Matrix multiply / GEMM / conv" | [`builtin_kernels.md`](references/builtin_kernels.md) `kernels.mm`/`conv2dk*` (+ `.mac_dims`), then [`patterns.md`](references/patterns.md) §GEMM-style + [`kernel_intrinsics.md`](references/kernel_intrinsics.md) §MMUL |
| "Sum / max / argmax / softmax" | [`builtin_kernels.md`](references/builtin_kernels.md) `kernels.reduce_*`/`softmax` + `algorithms.reduce`, then [`patterns.md`](references/patterns.md) §Reduction |
| "Same data to many cores" | [`patterns.md`](references/patterns.md) §Broadcast |
| "Split a big tensor across N workers" | [`patterns.md`](references/patterns.md) §Distribute/Join |
| "Two-stage pipeline (kernel1 → kernel2)" | [`patterns.md`](references/patterns.md) §Producer-consumer pipeline |
| "Tunable parameter from host" | [`patterns.md`](references/patterns.md) §RTP + WorkerRuntimeBarrier |
| "Deadlock / hang" | [`pitfalls.md`](references/pitfalls.md) §Forgetting the trailing release after a sliding-window loop |
| "Wrong results / garbage output" | [`pitfalls.md`](references/pitfalls.md) §Vector size divisibility + §MMUL divisibility + §ObjectFifo type vs. kernel signature |
| "Output is all zeros" | [`pitfalls.md`](references/pitfalls.md) §Device name doesn't match the hardware |
| "My fix changed nothing" | [`pitfalls.md`](references/pitfalls.md) §Stale JIT/xclbin cache |
| "Slow / not pipelining" | [`pitfalls.md`](references/pitfalls.md) §Missing __restrict + §Relying on AIE_PREPARE_FOR_PIPELINING alone |
| "How do I run / test this?" | [`programming_guide/section-3/README.md`](../../programming_guide/section-3/README.md) |
