---
name: aie-dataflow-presim
description: Guide to validating an AIE/IRON dataflow design in software before it ever touches hardware — a threaded ObjectFifo/Worker mock for deadlock/FIFO-depth detection, bit-exact validation against a numeric oracle, tiny isolated probes to de-risk novel mechanisms, and capacity/bandwidth modeling to catch a compute-bound-vs-DRAM-bound mistake before it's baked into placement. Use this whenever the user is designing a new IRON dataflow (ObjectFifo topology, kernel wiring, a novel streaming/attention/pipeline mechanism) and hasn't run it on real NPU hardware yet, is debugging a hang/deadlock and wants to isolate it without hardware, or is asking how to validate a design "before/without the NPU" — even if they don't use the word "simulator."
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Pre-hardware dataflow simulation (the "brainstorm without the NPU" phase)

Second of four porting phases (see `aie-model-baseline` for phase 1, the
numeric oracle this phase validates against; `aie-hw-bringup` for phase 3;
`aie-kernel-opt`/`aie-dataflow-opt` for optimization once it's correct).

The point of this phase: a deadlock, a wrong FIFO depth, or a math error in
a new design is cheap to find in a Python simulation that runs in seconds
on a laptop, and expensive to find on real hardware, where the failure mode
is often just "it hangs" or "the output is wrong" with no further
information volunteered. Do the cheap thing first.

## 1. Build a software mock of the ObjectFifo/Worker API, and simulate the exact topology

A useful mock doesn't need to model timing accurately — it needs to model
**the same synchronization semantics** your real design uses: bounded
queues per ObjectFifo with the depth you've declared, acquire/release
paired the way your kernels pair them, and a timeout on every blocking
operation so a deadlock fails fast (in seconds) instead of hanging forever.
A thread per tile/worker plus Python's `threading.Condition` per FIFO is
enough to build this once, model-agnostically, and reuse it for every
future design.

Plug in real numpy implementations of your kernels' math (not stubs) so
the same run that catches a deadlock also catches a computation bug.

**This has repeatedly caught real bugs for the cost of a test run**: wrong
FIFO depth causing a silent deadlock on the second iteration of a
weight-reload loop, a missing output parameter, a sign error in a shift —
all found in simulation, none of them requiring a single hardware cycle to
discover.

## 2. Validate bit-exact against the phase-1 oracle, incrementally

Don't wire the whole design and simulate it once — build and validate one
block/layer in isolation first, matching it bit-exact against the
`aie-model-baseline` oracle, before chaining two blocks together and
validating *that*. Growing the chain one link at a time means a mismatch
always has a small, recent diff to blame; validating the full chain cold
means a mismatch could be anywhere.

Turn each validated block into a standing regression test. A dozen fast
simulation tests that run in under ten seconds combined is cheap insurance
against silently reintroducing a bug while working on something else.

## 3. De-risk a genuinely novel mechanism with a tiny isolated probe first

If part of the design does something you haven't built before — a new
attention decomposition, a new streaming/broadcast pattern, a new way of
appending to a growing buffer — don't discover whether it works by wiring
it into the full design. Write the smallest possible standalone IRON
program that exercises just that mechanism, confirm it does what you think
it does, and only then integrate it. A probe that fails is a cheap,
isolated failure; the same bug discovered inside the full design is buried
under everything else that's also new.

## 4. Prove novel algorithmic decompositions are mathematically equivalent before realizing them in IRON

If a mechanism *changes the math*, not just the data movement (chunked/
streaming attention instead of full attention is the canonical example),
validate the decomposition against the reference **numerically, in plain
numpy, independent of IRON entirely** — e.g. confirm chunked-with-running-max
softmax produces cosine similarity ~1.0 against full softmax on realistic
inputs — before spending any time building the dataflow for it. If the
math doesn't check out, no amount of correct IRON plumbing will fix it;
if it does check out, you've separated "is the algorithm right" from
"is the dataflow right" into two independently debuggable questions.

## 5. Model capacity/regime (compute-bound vs. DRAM-bound) here, not after placement is fixed

If the workload is bandwidth-sensitive, build a rough per-op cost model
(MACs, bytes moved, cycles) against the target's known ceilings (peak
INT8 TOPS, DMA GB/s per column, on-chip memory budget) *before* deciding
tile placement. This tells you which regime you're in — compute-bound or
DRAM-bound — which is the single biggest input to how you should place
things in `aie-dataflow-opt` later. Getting this wrong doesn't show up as
a correctness bug in simulation; it shows up much later as a placement
that can't hit its performance target no matter how it's kernel-tuned,
because the constraint it was actually bound by was never modeled.

If the workload has more than one regime (e.g. a compute-bound prefill
pass and a bandwidth-bound decode pass in an autoregressive model), model
each separately — a single average cost model will mask both.

## Checklist before moving to phase 3 (`aie-hw-bringup`)

- [ ] Every block/layer simulated bit-exact against the phase-1 oracle,
      individually before chaining
- [ ] A standing regression suite of the simulated blocks exists and runs
      in seconds
- [ ] Any novel mechanism was probed in isolation, and any novel algorithmic
      decomposition was validated numerically against the reference
- [ ] If the workload is bandwidth-sensitive: a capacity/regime model
      exists identifying compute-bound vs. DRAM-bound per phase of the
      workload, before placement is decided
