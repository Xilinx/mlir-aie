<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Hardware-only judgement: verdicts, attribution, anti-patterns

The levers themselves, and the static signal each one needs, live in
`aie-kernel-opt-static` `references/levers.md`. This file holds what only a
device can settle:

- whether a candidate held up
- which kernel a number belongs to
- the anti-patterns that looked fine statically and lost on hardware

## Confirming or rejecting a candidate

Judge on traced cycles per call, arms back to back, populations split
(`measurement.md` §Hardware cycles).

| You see | Verdict |
|---|---|
| Both arms' ranges don't overlap and the candidate is lower; every unchanged kernel reproduces to the cycle | **confirmed**: commit with base → after per case |
| Ranges overlap, or the delta sits inside the spread of an unchanged kernel | **no change**: revert, and record the static signal as a miss |
| The candidate is higher | **rejected**: revert; record it as an X row here and a miss in the S table |
| An unchanged kernel moved | **void**: the arms leaked (cache, sibling include, stale install). Fix the arms, then rerun both |
| Only wall clock is available | Use `min`, pinned; the delta must clear −19.1..+13.3 µs plus ~3% (`measurement.md` §Wall clock) |

Traced cycles on an unchanged kernel repeat to the cycle, or within a few
cycles when the kernel has data-dependent paths. `q4nx_dequant` read
2245-2265 before and 2115-2116 after its stage-cap change. That tight range
is why a 1-cycle II move over 128 trips (-130 cycles) was a real result.

Check the measurement against the static report's prediction
(`bundles + 5 + (trips-1) × II`):

- `zero` predicted ~75 / ~130 and measured 78 / 134.
- `mm_bfp_mixed` predicted 1348 / 1312 and measured 1349 / 1313.
- If the prediction and the measurement disagree, find out which is wrong
  before you report either.

## Trace-row attribution

A cycle row names the design, not the kernel. Before you credit a delta:

- Confirm `event0()` is in the entry point the factory selects. With no
  marker there, the row is whatever else on the core is instrumented,
  usually `zero`. A `zero.cc` delta was once credited to matmul and prefill
  three times.
- Split by order when intervals = K × calls. One GEMV row read median 142
  when the kernel was 291 (`measurement.md` §Reading intervals).
- A row that moved while its file didn't change belongs to something it
  includes. Check `git diff <rev> -- <include closure>`: `mha.cc` includes
  `mm.cc`, `softmax.cc` and `zero.cc`.

## Port balance: diagnosis, not a lever

There is no measured hardware win from rebalancing load/store or vector
ports. Treat it as a diagnosis that explains a bound, and any change it
suggests as an `experiment`.

- `mm_bfp_mixed`: moving both A conversions onto the multiplier (`vmul` by
  ones, port b) to free port a left the object at 81 bundles per tile, no
  change. Its k loop is bound by the `vmac.f` accumulator recurrence at II6,
  not by a port (`aie-kernel-opt-static` `levers.md` §Bounds).
- Prefill `fv`'s paired loop (L18) is port-A bound, with about 36 `vlda` per
  iteration at II37. The one attempt to take loads out of that loop (X13) got
  a better static II and ran slower.

Before you call a loop port-bound, count the slot users per iteration in
`llvm-objdump` and compare with the II. If the II sits above the count, the
bound is something else (a recurrence, or MII over `SwpMaxMii`).

## Feeding outcomes back to the static table

Every confirmed, unchanged or rejected candidate becomes one row in
`aie-kernel-opt-static` `references/levers.md` §Signal → HW outcome, in the
same PR as the kernel change:

```
| S<n> | <static signal, before → after> | <kernel>, <change> (<L-ID>) | <base → after cycles per call> | hit | miss | hit only if <condition> |
```

- A miss is as valuable as a hit. S14 (uncalled symbol) and S15 (a better II
  that ran slower) are the reason the static skill never claims a speedup.
- If a lever's class changes because of the new row, update the ranked
  table's Class column too.
- A new anti-pattern gets an X row below, and the static S row points to it.
- `mha` round-6 row: placeholder, to be added when harvested.

## Bounds found by ablation

Replace one region with a cheap wrong computation and trace
(`measurement.md` §Ablation). The bf16 GEMV went 1154 → 634, which priced its
horizontal reduction at 520 of 1154 cycles, about 866 of which are
independent of K.

An ablation price is where the time goes, not a bound. After it, six
kernel-local tweaks lost (X02-X04), and the GEMV looked done. Restructuring
the reduction itself, four rows through one transposed tree (static L20),
then took it 1154 → 788 (-31.7%), bit-identical. Report NO-CHANGE only
against a bound the hardware can't beat (an accumulator recurrence, a slot
count), not against a price.

Some effects are only visible on hardware. At 4x2048 the same L20 change cut
the mac loop II9 → II7 but moved the call only 420 → 408: rows 4 KB apart
likely contend for the same memory banks, which remarks can't see (static
S18).

## Anti-patterns measured on hardware

| ID | Change | Kernel | HW result | Why |
|---|---|---|---|---|
| X01 | Optimizing a symbol no factory calls | gelu in-place | 594 → 594 despite II33 → 18 | wrong symbol (static S14) |
| X02 | `vec_size` 16 / 32 | bf16 `mv` | 1641 / 1321 vs 1154 | fewer lanes, same reduction |
| X03 | `UNROLL_FULL` on the chunk loop | bf16 `mv` | 1154 → 1062 at 4 chunks, 1010 → 1122 at 2, no build at ≥ 8 | shape-dependent, not shippable |
| X04 | 8 rows per group / `fold_to_16` / hoisting `b` | bf16 `mv` | stack 0x100 → 0xE00 / vst 4 → 12 / reloads | register pressure |
| X05 | Exact float `log2(e)` scale | flash prefill | error unchanged (2.62 → 2.75 steps), about 110 more instructions | the scale multiplies `s - m`, where the error doesn't grow |
| X06 | 2-limb bf16 split | fused epilogue | 12 / 32768 and 11 / 32768 words changed | 16 < 24 significand bits |
| X07 | Averaging host wall clock | any | phantom 7.5% between byte-identical builds | host noise |
| X08 | `mmul<8,8,8>` with a scalar A gather | 1x1 int8 conv | 3.34 → 3.58 ms (internal model) | the gather doubles |
| X09 | `if`/ternary inside a mac loop | int8 conv | 7% slower (internal model) | breaks the uniform issue |
| X10 | Unroll 2 on int16 `zero_scalar` | `zero` | wrong output | miscompile (static `traps.md` P07) |
| X11 | Trusting a tolerance for "exact" | `fused_mm` | 27 / 27 pass with a dropped term | `measurement.md` §Raw diff |
| X12 | `UNROLL_FULL` on `fused_mm`'s i loop (alone, or with z) | `fused_mm` | k_step 216 → 287; gelu/silu/sigmoid fail aiecc's `stack_size = 4096` check; with z, accumulators spill | register pressure (llvm-aie#1066); use z `UNROLL(2)` (L10) |
| X13 | Opaque `add_2d` pointer bump to keep S loads in the loop | prefill `fv` | 1374-1576 vs 1265 at 512 (static II37 → 31, yet slower); volatile+noinline operand build 1408; other tilings 1469-1852 | a better static II lost on HW (static S15) |

"Internal model" rows are wall-clock or end-to-end numbers from a quantized
model outside this repository. The number is real, but you can't reproduce it
from this repository.
