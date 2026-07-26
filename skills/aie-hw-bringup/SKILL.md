---
name: aie-hw-bringup
description: Guide to bringing up a new IRON design on real AIE/NPU hardware for the first time — sequential block-by-block (and, for large models, dispatch-by-dispatch) bring-up against a reference, immediate output comparison against the numeric oracle/ONNX Runtime, methodical bisection when something hangs or mismatches, memory-budget tile splits, and validating the host-orchestration handoff between dispatches as its own step. Use this whenever the user is getting a new design running on real NPU hardware for the first time, debugging a hang/deadlock/wrong-output on real hardware (as opposed to in simulation), comparing NPU output against a reference to find where a design diverges, hitting an L1/tile memory budget overflow, or debugging a multi-dispatch handoff — even if they haven't framed it as "bring-up."
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Hardware bring-up against a reference (the "minimal functional version" phase)

Third of four porting phases (see `aie-model-baseline` for the oracle this
phase compares against, `aie-dataflow-presim` for the simulation that
should already have caught topology/deadlock bugs before this phase
starts, `aie-kernel-opt`/`aie-dataflow-opt` for optimization afterward).

Simulation catches deadlocks and math bugs cheaply, but it can't catch
everything — real hardware has memory budgets, real DMA timing, and real
toolchain behavior that a mock doesn't model. This phase is about finding
*those* bugs efficiently once you're on real silicon, without re-deriving
the debugging process from scratch each time.

## 1. Bring up sequentially — never wire the whole design first

Get the smallest possible piece running and bit-exact on hardware first
(one block, one layer), then extend by one block/link at a time. At every
step, exactly one thing is new since the last known-good state, which is
what makes the next section possible.

Resist the temptation to wire the full chain because "the simulator
already validated it" — the simulator validates the topology and the math;
it doesn't validate real DMA timing, real memory layout, or toolchain
codegen, so hardware can still fail in ways simulation couldn't predict.

If the model spans multiple dispatches (see `aie-model-baseline`), bring
up and validate each dispatch on its own first, exactly like a block or
layer — then bring up the host-orchestration path that stitches dispatches
together as its **own**, separate step. A correct dispatch A and a correct
dispatch B do not imply a correct handoff between them: the host code that
writes A's output, reloads weights, and invokes B is new surface area that
doesn't exist in a single-dispatch design, and it fails in its own ways —
a race between when the host reads A's output and when B starts, a layout
or scale mismatch between what A wrote and what B expects to read, a
weight-reload that didn't happen because a cache thought nothing changed.
Treat it as a bring-up target in its own right, not an assumed-correct
implementation detail.

## 2. Compare against the reference immediately, on the same input

At every step, run the same canary input through both the NPU design and
the phase-1 oracle (or ONNX Runtime / the original framework), and diff
the output immediately — don't accumulate several new blocks before
checking. A single fixed canary input is enough for fast iteration; save
broader validation (many inputs, accuracy metrics) for once the design is
stable.

## 3. Bisect methodically — one hypothesis, one change, at a time

When something hangs or the output mismatches, treat every candidate cause
as an independent hypothesis and test it in isolation rather than changing
several things and hoping: disable streaming, lower an ObjectFifo depth,
split a kernel across two tiles, remove one dependency — one change per
test, narrowing to the minimal trigger. If a sibling block/layer already
hit a similar-looking failure, try its known fix first before reaching for
something novel — a new hypothesis is only worth it once the boring ones
are ruled out.

**If comparing two versions of a design that behave differently**, a
structural diff of the generated MLIR (FIFOs, locks, links, call
arguments) between the working and broken version is often faster than
staring at either one in isolation — the diff itself points at what
changed.

Concrete, previously-hit gotchas worth checking early in a bisection:
- **Devicename must match the physical hardware.** An npu1 xclbin run on a
  Strix (npu2) board can return all-zero output silently rather than
  erroring — check `xrt-smi examine` against the target string before
  chasing a "wrong output" bug that's actually a device mismatch.
- **Stale build artifacts masquerade as a code bug.** Incremental builds
  track file mtimes, not semantic changes; a `.prj/` directory or a stale
  `.exe` can silently keep running yesterday's design. When a fix doesn't
  change behavior, confirm the artifact actually rebuilt before concluding
  the fix didn't work.
- **NPU JIT/xclbin caching can serve a stale binary across runs.** Clear
  the cache directory between configurations you're trying to compare, and
  verify with a fresh cache location, not an assumption that "it must have
  rebuilt."
- **Device driver state can persist across runs in ways that look like a
  hardware fault.** Autosuspend timing and memory fragmentation after
  repeated rebuild/reload cycles have both produced failures that looked
  like a design bug but were resolved by a driver-level reset — worth
  ruling out before assuming the design itself regressed.
- **For a multi-dispatch design, check the boundary before the dispatches
  themselves.** If dispatch A and dispatch B each validate correctly in
  isolation but the chained result is wrong, suspect the handoff first —
  a scale/layout mismatch between what A wrote and what B expects, or a
  stale weight buffer that wasn't reloaded — before re-debugging either
  dispatch's internals.

## 4. Tile memory budget overflow: split, don't squeeze

When a tile's L1 (or a DMA descriptor's budget) overflows because a kernel
needs to hold more simultaneous state than fits — multiple activations
alive at once for a wide skip/concat, say — the fix that scales is
splitting the work across two tiles and cascading the partial result, not
squeezing the existing kernel by shrinking precision or unrolling less.
Precision/unroll cuts trade away exactly the correctness or performance
this phase is trying to establish; a tile split is a dataflow decision
that doesn't cost either.

## Checklist before moving to optimization (`aie-kernel-opt` / `aie-dataflow-opt`)

- [ ] Every block/layer brought up and validated bit-exact on real
      hardware, incrementally, not all at once
- [ ] Full chain validated bit-exact against the oracle/reference on at
      least one real canary input
- [ ] If the model spans multiple dispatches: each dispatch validated
      independently, and the host-orchestration handoff between them
      validated as its own step, not assumed correct because both sides
      passed in isolation
- [ ] Any memory-budget overflow was resolved via a tile split, not a
      precision/unroll compromise
- [ ] Design rules learned during bring-up (minimum FIFO depth for a
      pattern, a tile split that was necessary) are written down for reuse
      on the next variant, not left to be rediscovered
