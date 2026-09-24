---
name: aie-kernel-opt-hw
description: Measure and land AIE kernel speedups on the NPU. Takes a candidate report from aie-kernel-opt-static, or a raw compiled kernel (C++ built by Peano for AIE2P or AIE2, in aie_kernels/ or the user's own .cc), and settles it on hardware. Use when the user has an NPU and wants to know whether a kernel change is really faster, wants traced cycles per call, sees a trace interval count that doesn't match the calls, wants to claim bit-exactness, sees a kernel win that doesn't show in wall clock, or is running a multi-kernel or multi-agent optimization campaign. Drives the in-repo test_kernels_e2e gate, test_kernels_bench with kd.cycles_per_call, and MLIR_AIE_KERNEL_SOURCES arms measured back to back; covers population splits, mutation-proven gates, raw output diffs, one commit per kernel with its numbers, and feeding each outcome back to the static table. Not for tile placement or DMA bandwidth (aie-dataflow-opt), or for writing a first kernel (aie-code-creator).
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
to back with the same harness, as traced cycles per call with the
populations split. Every claim of exactness comes from a raw output diff.
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

3. **A/B, back to back.** Build the base arm as `aie-kernel-opt-static`
   describes (its `static-checks.md` §Arms). Then, in one session, for each
   arm:
   ```bash
   rm -rf $W/cache-<arm>
   NPU_CACHE_HOME=$W/cache-<arm> MLIR_AIE_KERNEL_SOURCES=<arm> flock "$NPU_LOCK" taskset -c $CPUS \
     pytest test/python/npu/test_kernels_bench.py -m benchmark -k "[$CASE]" --no-compile \
     --bench-out $W/<arm>-bench.json
   ```
   - No `<case>/cycles` row, or an interval-count error: apply the decision
     rules below.
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

5. **Exactness.** To say "bit-identical", diff the raw output words of both
   arms and mutation-prove the diff (§Raw diff). If the change reorders
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
| No `<case>/cycles` row | The contract lacks `trace_cycles=True` (only passthrough has it) | Check the markers line, then use the INTERIM probe (`measurement.md` §Probe) |
| `expected N ... got M`, M = K·N | K instrumented kernels per call (usually `zero` first) | Split by order: `intervals[j::K]`. Never quote min or median over the mixed stream |
| M > N, not a multiple | An inner-region bracket, or an initializer at another rate | Dump the stream in order and find the period (`measurement.md` §Reading intervals) |
| M < N | The trace buffer is full (bench uses 16384 B) | Use the probe with a larger trace size |
| M = 0, or no `event0()` in the selected source | No cycle number exists for this kernel | Use wall clock `min` and say so. A clean row can be `zero` |
| A row moved but its file didn't | Something in its include closure changed | `git diff <rev> -- <include closure>`; credit the right kernel (`hw-levers.md` §Trace-row attribution) |
| `core_fraction = cycles·calls/1.76e9/npu_s` < 0.5 | Dispatch-bound at this shape (60-90 µs floor) | Judge on cycles. If production is also dispatch-bound, go to `aie-dataflow-opt` |
| Wall-clock delta | Host noise band −19.1..+13.3 µs plus ~3% | Use `min` from `range`, pinned. The delta must clear both. Never use the mean (X07) |
| A run finished suspiciously fast | The JIT cache served a stale build | Wipe `NPU_CACHE_HOME` and rerun |

## Non-negotiables

- One change per A/B, and both arms in one session.
- Wipe `NPU_CACHE_HOME` per arm before every run you will report.
- Correctness before cycles: `ok=False` or a failing gate voids the numbers.
- A tolerance pass is not bit-exactness (X11).
- Keep every `extern "C"` name, signature and buffer layout. Leave
  `*_scalar` variants alone.
- Write provenance next to every number: commit, `git status --short` over
  the include closure, timestamp, Peano version, arm source directory.

## References

- `references/measurement.md`: the gate, bench commands, reading intervals,
  clock and dispatch, wall clock, the INTERIM probe, raw diff, ablation,
  provenance.
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
