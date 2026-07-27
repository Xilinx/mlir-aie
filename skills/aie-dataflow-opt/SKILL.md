---
name: aie-dataflow-opt
description: Guide to optimizing an AIE/IRON design's dataflow — dispatch partitioning, tile placement, overlays, weight/activation streaming strategy, and DMA bandwidth/compression — once it's already functionally correct. Distinct from aie-kernel-opt (which optimizes a single compiled kernel in place): this is about which op runs on which tile or dispatch, how data moves between them, and where the real bottleneck is before touching any kernel at all. Use this whenever the user is optimizing throughput/latency of a full IRON design (not a single kernel), deciding tile placement, overlay layout, or dispatch boundaries for a multi-dispatch model, choosing between static/streamed weights, modeling DMA bandwidth or compression ratios, or asking why a design is slow when profiling shows no single kernel is unusually expensive — even if they haven't used the word "dataflow."
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Dataflow optimization (the "make the whole design fast" phase)

Fourth of four porting phases, alongside `aie-kernel-opt` (see
`aie-model-baseline` for the oracle, `aie-dataflow-presim` for pre-hardware
validation, `aie-hw-bringup` for getting to a correct baseline on real
hardware — this phase assumes that baseline already exists and is correct).

**This skill is macro; `aie-kernel-opt` is micro.** `aie-kernel-opt`
answers "how do I make this one compiled kernel faster." This skill
answers "which op should run on which tile, how should data move between
them, and is a kernel even the right thing to be optimizing" — questions
that have to be answered *before* kernel-level tuning pays off, because no
amount of kernel polish fixes a design that's bottlenecked on placement or
bandwidth.

## 1. Rank the actual bottleneck before optimizing anything — NOOP ablation, not intuition

Before touching a single kernel or placement decision, replace each op
class with a no-op DMA pass (data moves, no compute happens) and re-measure
the full chain. The delta each NOOP causes is that op class's real
contribution to end-to-end time — rank by that, not by which kernel looks
expensive in isolation or which one you assume is slow. Static analysis
and paper cycle-counts are unreliable for this; a measured ablation isn't.

A kernel contributing 4ms is not worth the same optimization effort as one
contributing 99ms, even if the 4ms one looks the messier of the two.

## 2. If the model spans multiple dispatches, treat the boundary as the top-level placement decision

A model too large to fit one dispatch (see `aie-model-baseline`) was
already split there, but only for feasibility — that cut may not be the
one that performs best, so revisit it here rather than treating it as
fixed. Where to cut, and how many dispatches to use, is a placement
decision that dominates everything inside a single dispatch: it has to be
settled before overlay/tile placement within a dispatch is worth deciding,
because an overlay designed for a boundary that later moves is wasted
work.

Every dispatch boundary carries a real, measurable cost — writing
intermediate activations out, a host round-trip, reloading weights for the
next dispatch unless they're already resident. Model this the same way you
model any other bandwidth cost (below): more, smaller dispatches trade
resource headroom for more round-trip overhead; fewer, larger dispatches
trade overhead for tighter per-dispatch budgets. Measure the actual
round-trip cost against the resource pressure it relieves rather than
assuming either direction is obviously right.

## 3. Design separate placements for genuinely separate regimes *within* a dispatch — don't compromise into one

If the workload has more than one distinct operating regime — a
compute-bound pass and a bandwidth-bound pass (e.g. prefill vs. decode in
an autoregressive model; batch=1 vs. batch=N), or an accuracy-focused mode
and a preview mode — a single placement tuned for both regimes will
under-serve both. Compile and place separately per regime (a distinct tile
layout / column allocation per regime, sometimes called separate
"overlays"), and switch between them by workload phase rather than finding
a single compromise layout.

The signal that you're in this situation: the same op class binds on a
different resource depending on which regime you're in (e.g. compute-bound
in one, DMA-bound in the other) — if that's true, one placement literally
cannot be optimal for both.

## 4. Model bandwidth and compression before betting the design on an assumption

- **Don't assume equal DMA bandwidth across columns/channels** — measure
  it directly with a bandwidth-sweep harness before sizing per-column
  transfers. If columns are asymmetric, size buffer-descriptor byte
  allocations proportional to measured bandwidth rather than splitting
  evenly; an equal split runs at `N × min(column_bw)`, a proportional
  split recovers closer to `sum(column_bw)`.
- **Measure compression ratios (weight sparsity, KV-cache compression)
  against real weights/data, not a theoretical best case.** A structured
  sparsity scheme's real compression ratio on actual trained weights can
  differ substantially from its nominal ratio; measure before the ratio
  becomes a load-bearing part of a capacity plan.
- **Audit what the DMA genuinely can do before designing a transform onto
  it.** Buffer descriptors move addresses/strides/lengths and (on the
  hardware that supports it) apply compression — they do not perform
  arithmetic or dtype casts. If your dataflow needs a type conversion
  between two hops, that's a compute-tile cost you must budget explicitly,
  not something you can push onto the DMA for free.
- **A recurrent/cache-carrying workload accumulates its own bandwidth
  pole.** If the design carries state across iterations (a KV cache, a
  running accumulator that grows with context), model that state's DMA
  traffic explicitly as context grows — at some scale it can dominate
  weight traffic even if it looked negligible at the sizes you first
  tested.

## 5. Treat weight/activation streaming strategy as a placement decision, not an implementation detail

Whether weights are staged statically (fit once, reused every call) or
streamed continuously affects tile occupancy and DMA scheduling, not just
which API call you use — decide it alongside placement, informed by
whether weights fit in on-chip memory at all and how the phase-1
capacity model characterized the regime. A design that fits weights
statically and self-loops a single buffer avoids per-call host-device
handoff latency entirely; a design that doesn't fit needs an explicit
streaming plan (and the two are different data-movement footprints, not
a switch you flip after the fact).

## Checklist

- [ ] NOOP ablation has ranked the real per-op-class contribution to
      end-to-end time — optimization effort is going to the top of that
      ranking, not to whichever kernel seemed slow
- [ ] If the model spans multiple dispatches, the boundary was evaluated
      for performance (round-trip cost vs. per-dispatch resource
      pressure), not just carried over unchanged from the phase-1
      feasibility cut
- [ ] Distinct workload regimes (if any) have distinct placements *within*
      a dispatch, not one compromise layout
- [ ] DMA bandwidth (per column/channel) and any compression ratio are
      measured against real hardware/data, not assumed
- [ ] Any required dtype/arithmetic transform between DMA hops is budgeted
      onto a compute tile explicitly, not assumed free
- [ ] Weight/activation streaming strategy (static vs. streamed) was
      decided as part of placement, informed by the capacity model from
      `aie-dataflow-presim`
