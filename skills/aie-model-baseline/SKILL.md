---
name: aie-model-baseline
description: Guide to preparing an ML model for AIE/NPU deployment before any device or dataflow work begins — choosing/locking a quantization scheme, exporting ONNX, extracting a deployment manifest (per-op scales/zero-points/shifts/layout), and building a bit-exact numeric oracle. Use this whenever the user is quantizing a model for NPU/AIE deployment, exporting to ONNX, picking an INT8/XINT8 scheme, building a reference/oracle to validate a kernel or dataflow design against, or asking how to start porting a model to run on AIE — even if they haven't mentioned "oracle" or "manifest" by name.
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Model baseline & numeric oracle (the "before you write any IRON" phase)

This is the first of four phases in porting a model to AIE/NPU:

1. **Model baseline & numeric oracle** (this skill) — get a validated software
   reference and a quantization spec before touching hardware or dataflow.
2. **Pre-hardware dataflow simulation** (`aie-dataflow-presim`) — validate the
   *design* (FIFOs, kernels, novel mechanisms) against the oracle in software.
3. **Hardware bring-up against reference** (`aie-hw-bringup`) — get a minimal
   version running on real silicon, comparing against the oracle at every step.
4. **Kernel optimization** (`aie-kernel-opt`) and **dataflow optimization**
   (`aie-dataflow-opt`) — once it's correct, make it fast.

Skipping this phase doesn't save time — it moves the cost downstream, where
it's an order of magnitude more expensive to pay: a scale/shift/rounding bug
found here is a five-minute numpy fix; the same bug found during hardware
bring-up looks like a mysterious accuracy cliff or a hang, and costs days to
isolate.

## 1. Lock the quantization scheme before designing anything downstream

Pick precision and quantization granularity (per-tensor / per-channel /
per-token) *before* any kernel or dataflow design starts — the scheme
determines what the kernels have to do (a fixed power-of-2 shift vs. a
runtime-variable one; a per-row `reduce_max` before requant vs. not), so
changing it later means redesigning kernels, not just retuning them.

- If your quantizer claims power-of-2 scales, **verify it at design time**,
  not after kernel-writing has started — walk the quantized graph and check
  every scale is actually `2^k`. If it isn't, either fix the quantizer
  config or accept the cost now (kernels will need a variable-shift path,
  which is slower and a real design decision, not a bug to fix later).
- **The right granularity is model-class dependent, and the wrong guess is
  invisible in a per-op unit test.** A spatial CNN can often tolerate
  per-tensor activation quantization; an LLM frequently cannot — outlier
  channels in transformer activations produce a catastrophic accuracy cliff
  (a measured case: +1300% perplexity) under per-tensor quantization that
  disappears under per-token/per-row scaling. This only shows up when you
  validate the **full model on real data**, never in an isolated op test
  with random inputs. If you're quantizing a transformer, budget for a
  per-row `reduce_max` before every requant from the start — it's cheap in
  hardware but expensive to retrofit into kernels and dataflow after the
  fact.
- Decide now whether activation/weight/KV-cache (if the model is
  recurrent/autoregressive) each need their own granularity — they're
  independent decisions with independent hardware costs.

## 2. Budget the whole model against a single dispatch before designing anything downstream

Both reference ports this skill family is grounded in (a small CNN
classifier, a 1B-parameter LLM) fit entirely within a single NPU dispatch
— one compiled design, one set of tiles, no host round-trip mid-model.
That's not automatic for a larger model, and getting it wrong looks like
everything else in this phase working perfectly right up until dataflow
design starts, at which point nothing fits.

Before committing to a design, do a rough budget: total weight bytes at
your locked quantization scheme, a rough peak activation footprint, and
instruction/program-memory needs, against the NPU configuration's actual
ceilings (columns available × per-tile L1, memtile/L2 capacity, program
memory per tile, how many distinct kernel bodies can coexist). If it
doesn't clearly fit with headroom, decide now — not once dataflow design
is underway — where the model splits into multiple dispatches (separate
compiled designs, each invoked separately, with intermediate results
round-tripping through host/DRAM between them).

Split at a natural boundary the model already has (a layer/block boundary
that was going to produce an intermediate tensor anyway) rather than
mid-block — a mid-block split adds a host round-trip exactly where none
was needed and can force more intermediate state to stay live than the
boundary naturally requires.

Once dispatch boundaries are decided, extend the manifest and oracle from
this phase to make each dispatch's boundary tensors first-class: the
oracle needs to expose the exact intermediate activations at each cut
point, because `aie-dataflow-presim`/`aie-hw-bringup` will validate each
dispatch independently against those tensors — the same way per-block
validation works for a single-dispatch design, except these boundaries
are real host-mediated hops, not just internal tile boundaries. See
`aie-dataflow-opt` for how the boundary choice itself becomes a
performance decision later, not just a feasibility one.

This is the least battle-tested section in this skill family — both
source projects were small enough to skip it entirely. Verify the
specific ceilings you use against current toolchain/hardware documentation
rather than treating any number here as fixed.

## 3. Export ONNX, then extract a deployment manifest — don't rely on the graph itself as the source of truth

Walking the quantized ONNX graph by hand at kernel-design time, over and
over, is slow and error-prone. Instead, do it once: extract every op's
quantization parameters (scales, zero points, right-shifts, bias
pre-shift/promotion scheme) and its data layout into a single manifest
(e.g. a JSON file) up front. This becomes:

- The **source of truth** that kernel generation and IRON design consume,
  instead of everyone re-deriving math from the graph independently.
- A **regression artifact** — if a re-quantization or re-export changes a
  scale, the manifest diff shows exactly what moved.

Record weight *layout* (e.g. `OIYX` vs. a packed/tiled format) as a
separate field from quantization — they're independent transforms, and
keeping them separate lets you test layout round-trips without needing any
quantization math, and vice versa.

## 4. Build a bit-exact numeric oracle, and validate the manifest against it before writing any device code

Write a pure-software reference (numpy or equivalent) that implements the
*exact* arithmetic your kernels will use — the same rounding mode, the same
accumulator width/overflow behavior, the same shift-then-clamp order — not
just a floating-point approximation. Then validate:

- **Manifest extraction against this oracle, bit-exactly**, before any IRON
  or kernel work starts. If your numpy INT8 conv/matmul doesn't match your
  extracted scales/shifts bit-for-bit against ONNX Runtime on the same
  input, the bug is in your understanding of the quantization scheme —
  find it here, where it costs a numpy print statement, not on hardware
  where it costs a trace capture and a guessing game.
- **The full model on real data, not just isolated ops with random
  inputs.** Random-input unit tests on individual ops are necessary but not
  sufficient — they systematically miss data-dependent failure modes (the
  outlier-channel quantization break above is invisible to a per-op random
  test and only appears running real text/images through the whole model).

This oracle is not throwaway scaffolding — it's the reference every later
phase compares against: the software dataflow simulator (phase 2) validates
against it, and hardware bring-up (phase 3) validates against it again on
real silicon. Time spent making it trustworthy now pays off three times
over.

## Checklist before moving to phase 2 (`aie-dataflow-presim`)

- [ ] Quantization scheme chosen and its granularity validated on the full
      model with real data (not just random-input op tests)
- [ ] Power-of-2 scale assumption checked, if your kernel design depends on
      it
- [ ] Model's total weight/activation/instruction footprint budgeted
      against the NPU's single-dispatch ceiling; if it doesn't fit,
      dispatch boundaries are decided now (ideally at natural layer/block
      boundaries) and the manifest/oracle expose each boundary's
      intermediate tensors
- [ ] Deployment manifest extracted (scales/zero-points/shifts/layout per
      op) as a standalone artifact, not re-derived from the graph ad hoc
- [ ] Numeric oracle implemented with the *exact* target arithmetic
      (rounding mode, accumulator width, shift order) and validated
      bit-exactly against the manifest + ONNX Runtime
- [ ] Oracle validated against the full model on real data, not only
      per-op random-input tests
