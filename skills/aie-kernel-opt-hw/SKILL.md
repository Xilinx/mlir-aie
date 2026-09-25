---
name: aie-kernel-opt-hw
description: Measure and land AIE kernel speedups on the NPU. Takes a candidate report from aie-kernel-opt-static, or a raw compiled kernel (C++ built by Peano for AIE2P or AIE2, in aie_kernels/ or the user's own .cc), and settles it on hardware. Use when the user has an NPU and wants to know whether a kernel change is really faster, wants traced cycles per call, sees a trace interval count that doesn't match the calls, wants to claim bit-exactness, sees a kernel win that doesn't show in wall clock, or is running a multi-kernel or multi-agent optimization campaign. Drives the in-repo test_kernels_e2e gate, and test_kernels_perf with kd.cycles_per_call and --baseline-sources, measuring base and candidate back to back; covers reading cycle rows, mutation-proven gates, raw output diffs, one commit per kernel with its numbers, and feeding each outcome back to the static table. Not for tile placement or DMA bandwidth (aie-dataflow-opt), or for writing a first kernel (aie-code-creator).
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AIE kernel optimization: hardware A/B

This skill decides, on the NPU, whether a kernel change is faster, keeps it
correct, and commits it with its number. It assumes two things already hold:

- the kernel is correct on hardware (`aie-hw-bringup`)
- the kernel is worth optimizing (`aie-dataflow-opt` ranks kernels by
  ablation)

**Evidence rule.** Every claimed speedup comes from two arms measured back
to back with the same harness, as traced cycles per call of the kernel
itself, split from any initializer's. Every claim of exactness comes from a raw output diff.
A static number is a prediction to check, never a result.

## Setup

```bash
source /opt/xilinx/xrt/setup.sh            # XRT first, or pyxrt is missing
source <venv>/bin/activate
source utils/env_setup.sh install          # from the repository root (docs/Building.md)
REPO=$PWD; W=$(mktemp -d); K=<factory>; CASE=<case name from kernel_cases.py>
CPUS=<fixed host CPU list>; NPU_LOCK=<lock file shared by everyone on this NPU>
```

Check that the device matches the target (`xrt-smi examine`): npu2 is
`aie2p`, npu1 is `aie2`. If other agents share the NPU, read
`references/campaign.md` first.

## Steps

`references/measurement.md` has the exact command and reading rules for
each step.

1. **Get a candidate report.** If you were handed one from
   `aie-kernel-opt-static`, check that it names the entry symbol, the class,
   the static metrics, the cases and the markers line. For a raw kernel, run
   the static screen per `aie-kernel-opt-static` first, and come back with
   its report. Don't copy its steps here.

2. **Gate, and prove the gate.**
   ```bash
   pytest test/python/npu/test_kernels_e2e.py -m extensive -k "$CASE" --seeds 3
   ```
   - Run it on the candidate, including the report's remainder case.
   - Mutate the kernel, watch the gate fail, then revert. Report any mutation
     that survives, with the reason (§Gate).

3. **A/B, back to back.** Build the base arm `$BASE` as
   `aie-kernel-opt-static` describes (its `static-checks.md` §Base arm), then
   measure both arms in one run:
   ```bash
   flock "$NPU_LOCK" taskset -c $CPUS \
     pytest test/python/npu/test_kernels_perf.py -m perf -k "[$CASE]" --no-compile \
     --baseline-sources $BASE --perf-out $W/perf.json --perf-meta $W/meta.json
   ```
   - The terminal summary prints `cycles base -> cur`, `npu_us min base ->
     cur` and the raw output words that differ, per case. `--perf-meta`
     has the same under `baseline.cases.<case>`.
   - Add a case of a kernel you didn't touch
     (`-k "[$CASE] or [<unchanged case>]"`). It must reproduce to the cycle
     across the arms.
   - No `<case>/cycles` row, or an interval error: apply the decision rules
     below.
   - Build both tanh configurations for activations (default native, and
     `-DACTIVATIONS_TANH_LUT=1`).

4. **Verdict** per `references/hw-levers.md` §Confirming or rejecting:
   - **confirmed**: ranges separate, and every unchanged kernel reproduces
     to the cycle
   - **no change**
   - **rejected**
   - **void**: an unchanged kernel moved, so the arms leaked; fix them and
     rerun

   Compare the result with the report's per-call prediction. If they
   disagree, find out which is wrong first.

5. **Exactness.** To say "bit-identical", every case must read `same`
   (`differing_words` 0) in step 3's run. Mutation-prove it: a broken
   candidate must show words that differ (§Raw diff). If the change reorders
   accumulation, say it isn't bit-identical and give the max |Δ|.

6. **Record.**
   - Write one commit per kernel and change, with base → after cycles per
     case, n, the arm sources and the gate line.
   - List each rejected variant with the number that killed it.
   - Add the outcome as an S row to `aie-kernel-opt-static`
     `references/levers.md` in the same PR, hit or miss
     (`hw-levers.md` §Feeding outcomes back).
   - If nothing moves, report NO-CHANGE with the measured bound. That is a
     valid result.

## Decision rules for hardware rows

| You see | It means | Do |
|---|---|---|
| No `<case>/cycles` row | The contract declares `Trace.none` or `Trace.partial` (`set_rounding`, `fused_mm`, the default `mha` build) | Use wall clock `min` and say so. To time it, make one marker pair bracket the whole call and declare `Trace.whole_call()` (`measurement.md` §Reading the cycles row) |
| `expected E trace intervals, got M; a kernel on the core emits markers its contract's trace does not declare` | Markers the contracts don't declare: an inner-loop bracket, or an initializer with undeclared markers | `pytest test/python/test_kernel_trace_markers.py` names the build |
| `truncated` in the cycles range, or `none of the kernel's` | The trace buffer filled, or the markers are missing | Compare `n=` with `calls`, flag n < calls/2; run the marker audit |
| A stream you traced yourself, K instrumented kernels per call | The performance check splits by position; a hand trace doesn't | Population j is `intervals[j::K]`. Never quote min or median over the mixed stream |
| A row moved but its file didn't | Something in its include closure changed | `git diff <rev> -- <include closure>`; credit the right kernel (`hw-levers.md` §Trace-row attribution) |
| `core_fraction = cycles·calls/1.76e9/npu_s` < 0.5 | Dispatch-bound at this shape (60-90 µs floor) | Judge on cycles. If production is also dispatch-bound, go to `aie-dataflow-opt` |
| Wall-clock delta | Host noise band −19.1..+13.3 µs plus ~3% | Use `min` from `range`, pinned. The delta must clear both. Never use the mean (X07) |
| An unchanged kernel moved between arms | The arms differ in more than your change | Check `diff -rq` against `$BASE` and the `kernels` digest in `extra`; the verdict is void |

## Non-negotiables

- One change per A/B, and both arms in one run (`--baseline-sources`).
- Correctness before cycles: a failing gate, or a performance case that fails its
  contract in either arm, voids the numbers.
- A tolerance pass is not bit-exactness (X11).
- Keep every `extern "C"` name, signature and buffer layout. Leave
  `*_scalar` variants alone.
- Write provenance next to every number: the row's `extra` (commit, Peano,
  `kernels` digest), `git status --short` over the include closure, a
  timestamp, and what `$BASE` was archived from.

## References

- `references/measurement.md`: the gate, the performance A/B, reading the cycles
  row, clock and dispatch, wall clock, raw diff, ablation, provenance.
- `references/hw-levers.md`: verdict rules, trace-row attribution, port
  balance as a diagnosis, feeding rows back to the static table, and
  anti-patterns measured on hardware (X01-X13).
- `references/campaign.md`: several kernels or several agents on one tree
  and one NPU.
- `aie-kernel-opt-static`: the static screen, levers L01-L20, the signal →
  HW outcome table, and Peano traps.

For background only (no step needs it): `programming_guide/section-4/section-4d/README.md` §Markers come first,
§Two clocks, §Cycles you can trust, §A worked example: add, §Keeping it
correct, §Changes that measured worse.
