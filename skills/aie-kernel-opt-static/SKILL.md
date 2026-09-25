---
name: aie-kernel-opt-static
description: Find and screen speedup candidates for one compiled AIE kernel without a device. For C++ kernels built by Peano (llvm-aie) for AIE2P or AIE2 in bf16, float, int8 or int16, from elementwise and normalization kernels to matmul, GEMV, attention and conv. Use when the user wants to know why a loop has a high II, won't pipeline, spills, overflows its stack or calls __mulsf3, __divsi3 or another libcall; wants to vectorize or restructure a kernel in aie_kernels/ or their own .cc; or has no NPU at hand. Drives the in-repo aie.utils.compile.remarks report and its base-arm diff, the trace-marker audit, and test_kernel_contracts.py. Output is a candidate report per change (diff, static metrics before → after, a confidence class from a table of static signals vs measured HW outcomes, including the misses). It never claims a speedup; aie-kernel-opt-hw measures candidates on the NPU. Not for tile placement or DMA bandwidth (aie-dataflow-opt), or for writing a first kernel (aie-code-creator).
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AIE kernel optimization: static screen

This skill finds changes to **one compiled kernel** that the compiler's own
output says should be faster, and ranks them by how often that kind of
static signal has turned into a hardware win. It needs Peano and the
repository. It doesn't need an NPU.

**HARD RULE: never claim a speedup.** A static win is a *candidate*. Don't
write "faster", "N% speedup" or "Nx" about the user's kernel, in the report,
a commit message, a PR, or a code comment, even when the II halves. Say
"candidate, HW unconfirmed" and hand the report to `aie-kernel-opt-hw`. The
record says why: an II33 → 18 change measured 594 → 594 on hardware (it
edited an uncalled symbol), and an II37 → 31 change measured slower
(`references/levers.md` S14, S15).

The precedent numbers in `references/levers.md` are hardware measurements of
*other* kernels. Cite them as the record behind a confidence class, never as
the expected result for this one.

## Setup

```bash
source /opt/xilinx/xrt/setup.sh            # if installed; env_setup.sh expects it first
source <venv>/bin/activate
source utils/env_setup.sh install          # from the repository root (docs/Building.md)
REPO=$PWD; W=$(mktemp -d); K=<factory>
```

Use `--target aie2p` for npu2 (Strix and later) and `--target aie2` for npu1
in every remarks command.

| Internal name (toolchain, `detail/` dir) | Marketing name | Chips |
|---|---|---|
| `aie` | AIE | Versal VCK190 |
| `aie2` (npu1) | AIE-ML (XDNA) | Phoenix |
| `aie2p` (npu2) | XDNA2 | Strix Point, Strix Halo, Krackan |
| `aie2ps` | AIE-MLv2 | Telluride |

The aie_api headers are the ground truth for what an intrinsic lowers to and
which vector widths and `mmul` shapes a target has:
`third_party/aie_api/include/aie_api/` (`aie.hpp` is the umbrella), with the
per-architecture lowerings in `detail/aie2/`, `detail/aie2p/` and so on.

## Steps

Run them in order. `references/static-checks.md` has the exact command and
reading rules for each step.

1. **Resolve what runs.** Follow the factory in `python/iron/kernels/` to the
   source, the production `-D` flags and the `extern "C"` symbol, then to the
   function that symbol calls; remarks prints it as `[OK] name: symbol from
   source`. Edit only that function. Record the contract's `trace=`
   (§Resolve).

2. **Contracts.** `pytest test/python/test_kernel_contracts.py -k "$K" -q`,
   host only. It must pass before and after (§Contract check).

3. **Base arm and profile.** Build a base arm at the starting revision and
   run remarks with `--keep` (§Base arm, §Static report). Read the
   `libcalls` and `stack_bytes` rows (the stack against the contract's
   budget), `pm_bytes`, and `[sp, #` traffic in the kept objects
   (§Libcalls, stack and objects).

4. **Diagnose** with the table below. Don't pick a lever without a
   diagnosis. If the loop already sits at its bound, stop: report NO-CHANGE
   with the bound (`levers.md` §Bounds).

5. **Apply one change** and re-run remarks with `--baseline-sources` on the
   base arm; it prints each row that moved. The lever's Check must move
   (`levers.md`). If no static metric moved, revert it.
   - For an unroll, run the unroll screen first (§Unroll screen).
   - If the change creates a path no case reaches, add a remainder `Case`
     (§Remainder case).
   - Re-run step 2.

6. **Classify and predict.** Give the candidate exactly one class from
   `levers.md` §Confidence classes: `strong`, `likely`, `experiment` or
   `reject`. Write the per-call prediction `bundles + 5 + (trips-1) × II`
   (§Predict). Compare II per unit of work (per row, value or tile), not
   per loop: a raw II that rises can still be a win (S17). Scale a small II
   change by the trip count before dismissing it (S09).

7. **Report** each candidate with the template below. Report rejected
   variants too, with the compiler's number.

## Diagnosis → lever

| Remarks / object shows | Lever | Class if the Check moves |
|---|---|---|
| The `libcalls` row names a `traps.md` P01 helper called in a loop | L02; L11 for `__divsi3` | strong |
| `vector<float>` multiply or min/max in a loop | L03 skip ×1, three-limb split, bf16 clamp after rounding; L04 32 lanes (AIE2P) | likely |
| Array of accumulators or vectors indexed by a loop counter; stack traffic | L01 `UNROLL_FULL` | strong |
| Loop you care about isn't innermost or single-block (unpipelined parent) | L10 fold or unroll into it | strong if it newly pipelines |
| Loop branches on its counter | L13 `UNROLL_FULL`, not `RANGE` | likely |
| Address arithmetic in the body; no `__restrict` | L06 walking restrict cursors | likely |
| Pipelined; `byte_count` flat from ×1 to ×4 | L07 `UNROLL(4)` | strong |
| AIE2: unrolled body at MII above `SwpMaxMii` 27, postpipeliner only | `__restrict` plus one chain under `AIE_LOOP_NO_UNROLL` (L07's AIE2 note; S24, S25, S58, S63) | strong (the loop newly pipelines) |
| AIE2: runtime trip count, loop not overlapped | `AIE_LOOP_MIN_ITERATION_COUNT(n)` plus a plain loop for shorter rows (L11; S29, S36, S40, S44) | likely |
| AIE2: LUT reads ordered against the stores, loop runs in series | Rotate the loop one vector ahead, or `lut_map_bf16` (S33, S67) | likely |
| crRnd saved, set and restored inside the loop | Hoist the rounding-mode change out of the loop (S31) | likely |
| AIE2: bf16 `sliding_mul` (no native one) | `mac_elem_16_2` / `shuffle_down_fill` (S57) | likely |
| AIE2: `aie::transpose` in the loop | Two-register `::shuffle` stages (S81, S82) | likely |
| Stepping 16 lanes on 8/16-bit, or on bf16 on AIE2P | L08 64 lanes / `AIE_BF16_LANES` | likely |
| One long mac chain, or broadcasts spilled per block | L09 split chains | likely |
| Pipelined, `ns` = 3, one long latency chain with no dominant step | L17 `--aie-pipeliner-max-stagecount=4/5` in the factory's `compile_flags` | likely (scale by trips) |
| Several `mmul<8,8,8>` accumulators for Y += S·V; high II, large frame | L18 two 8x8 tiles on one 64-lane accumulator | likely |
| A `reduce_add` or `reduce_max` (horizontal tree) per row, or a per-row helper call, unpipelined or a long chain the row loop never overlaps | L20 several rows through one transposed tree | likely (II per row, S17, S20) |
| bfp16 stream state spilled in the loop: `sfl/sfh` to `[sp]` around `vst.push`, or FIFO state around `vldb.pop` (more streams than registers) | L19 one stream per operand | strong (in-loop `[sp` drops, S19) |
| Unpipelined outer loop around a pipelined mac loop | L10 `UNROLL(2)` on it | strong (`unpipelined_loops` drops) |
| `(tanh+1)/2`, or separate constant multiplies | L05 fold into one mac | likely |
| `lda.s8`/`st.s8` byte loop | L12 `uint64_t` or vector stores | strong |
| Scalar int8 requantize tail | L14 int32 bias + `to_vector<int8>` | likely |
| Scalar gather building the mmul A operand | L15 producer writes A order | likely |
| Body is only a rearrangement | L16, hand off to `aie-dataflow-opt` | n/a |
| II equals the resource or latency bound (`mm_bfp_mixed` II6; bf16 `mm` MII34 vs II35, above `SwpMaxMii` 27, `traps.md` P13) | NO-CHANGE with the bound | n/a |

A candidate that gets its II drop by hiding work from LLVM (an opaque
pointer bump, `volatile`, a LICM blocker) is `experiment` at best: that is
the S15 miss. A variant that matches `levers.md` §Compiler reports (X20-X46)
is `reject`.

## Candidate report

One block per change:

```
Candidate: <kernel> / <entry symbol> : <lever ID or "no precedent"> : <one line>
Class: strong | likely | experiment | reject   (precedent: S.., L..)
Status: candidate, HW unconfirmed. Hand to aie-kernel-opt-hw.
Diff: <unified diff of the source, one change only>
Static, base → candidate (per loop fn/bb and per build):
  II, ns, pipelined, zol, byte_count, unpipelined_loops, pm_bytes,
  libcalls, stack_bytes vs stack_budget, [sp, # count
Predicted per call (not measured): <bundles + 5 + (trips-1) x II, per case>
Cases: <case names; new remainder case if any>. Markers: whole_call | none | partial (<reason>)
Contracts: test_kernel_contracts.py -k <K> pass
Rejected variants: <change>: <compiler's number>
```

For NO-CHANGE, give the loop, its II, the bound and where the bound comes
from.

## Non-negotiables

- Never claim a speedup (HARD RULE above). No commit message or PR text may
  state one on static evidence.
- One change per candidate. A candidate that moves no static metric is
  reverted, not reported as a win.
- `AIE_PREPARE_FOR_PIPELINING` expands to nothing under Peano, so it's never
  a lever. `AIE_PREPARE_FOR_POSTPIPELINING` turns pipelining **off**
  (`traps.md` P03, P04).
- Keep every `extern "C"` name, signature and buffer layout. Leave
  `*_scalar` variants alone.
- Check the `stack_bytes` row against the budget after any change that grows
  the frame. Overflow is silent on hardware (`traps.md` P08).

## References

- `references/levers.md`: the static signal → HW outcome table (S01-S83,
  hits and misses; S23 onward on AIE2, npu1), confidence classes, levers L01-L20 with
  when/do/check/HW precedent, compiler-reported rejects X20-X46, bounds.
- `references/static-checks.md`: exact commands for every step, and how to
  read the remarks output, rows, meta and kept objects.
- `references/traps.md`: Peano, AIE2 and AIE2P traps P01-P14.
- `aie-kernel-opt-hw`: measures candidates on the NPU and adds each outcome
  back to the S table.

For background only (no step needs it):
- `programming_guide/section-4/section-4c/README.md`: how the compiler schedules a loop, reading the remarks and the
  disassembly, trip counts, the loop-pragma reference.
- `programming_guide/section-4/section-4d/README.md` §Reading the static report, §Levers that measured faster, §Peano
  and AIE2P traps, §When to stop.
- `programming_guide/kernels_library.md`, "Testing, benchmarking and static
  checks".
