<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Traps (Peano, AIE2 and AIE2P)

Code that compiles cleanly and then does nothing, crashes the compiler, or
produces wrong data. Every item here was hit in real kernel work. Compiler
behavior depends on the Peano version:

- The pin is in `utils/peano-requirements.txt` (now `22.0.0.2026092401`).
- Check what you actually run with `$PEANO_INSTALL_DIR/bin/clang --version`.
- Items marked "older pin" haven't been re-tested on the current one.
  Re-check before building a workaround into new code.

## P01 Scalar float, 64-bit and divide helpers are libcalls (AIE2 and AIE2P)

The remarks `libcalls` row (and its `calls the runtime library:` line)
names each one:

| Source | Helper |
|---|---|
| `float * float` | `__mulsf3` |
| `float / float` | `__divsf3` |
| `(float)int32`, `(float)uint32` | `__floatsisf`, `__floatunsisf` |
| float `<`, `>` | `__ltsf2`, `__gtsf2` |
| any `double` op | `__*df*` |
| 64-bit multiply, including a constant divide by a non-power-of-two | `__muldi3` |
| signed int divide (including by 2^k) | `__divsi3` |

The native alternatives are float add/sub, `aie::inv`, `aie::invsqrt`,
`aie::to_float`, `aie::max`/`min` and bf16↔float casts. A libcall in a loop
also blocks pipelining. The fix is lever L02 (`levers.md`).

`aie::to_float` and `to_fixed` are native, but they aren't free inside a
vector chain: they go through SRS/UPS with mode-register writes (L04).

## P02 No f32 vector multiplier (AIE2 and AIE2P)

`aie::mul`/`aie::mac` on `vector<float,N>` is emulated with bf16 products,
at 32 lanes on AIE2P (II77 to II143 per 16 lanes on AIE2, S61): 224 B of code where one bf16 mac is 4 B. There is no mixed
f32×bf16 `aie::mul`. The fixes are L03 and L04.

## P03 Pragmas that expand to nothing under Peano

`aie_kernel_utils.h:55-66` makes these empty under Peano. They are real only
under Chess:

- `AIE_PREPARE_FOR_PIPELINING`, `AIE_NO_PREPARE_FOR_PIPELINING`
- `AIE_MODULO_SCHEDULING_BUDGET_RATIO`, `AIE_KEEP_SW_LOOP`,
  `AIE_PEEL_PIPELINED_LOOP`
- `AIE_KEEP_FREE_FOR_PIPELINING`, `AIE_ALLOCATE`, `AIE_NO_HW_LOOP`,
  `AIE_LOOP_FLATTEN`

`mv_bf16` was byte-identical in 6 configurations with and without
`AIE_PREPARE_FOR_PIPELINING`. Leaving one in existing code is harmless, but
adding one is never a lever, and none of them explains why a loop did or
didn't pipeline.

## P04 `AIE_PREPARE_FOR_POSTPIPELINING` turns pipelining off

Under Peano it expands to `clang loop pipeline(disable)`
(`aie_kernel_utils.h:65`). swiglu went to II49 with it (X25).
`VERSIONED_LOOP` forwards extra pragmas, so check its call sites too.

The pragmas that act under Peano:

| Macro | Effect |
|---|---|
| `AIE_LOOP_UNROLL(n)`, `AIE_LOOP_UNROLL_FULL`, `AIE_LOOP_NO_UNROLL` | unroll count / full / off |
| `AIE_LOOP_MIN_ITERATION_COUNT(n)`, `AIE_LOOP_MAX_ITERATION_COUNT(n)` | trip-count hints. MIN can cost the zero-overhead loop (X37), so re-check `non_zol_loops`. On AIE2, MIN(n) on a runtime-count loop let it overlap (`axpy` 269 → 87, S29; `levers.md` L11), with a plain loop kept for shorter rows. `MIN(2)` on a loop that always runs ≥ 2 trips let the postpipeliner overlap it: `partial_softmax` 1861 → 1720 (`levers.md` L20) |
| `AIE_LOOP_RANGE(lo, hi)` | a trip-count hint only. It does **not** unroll (L13) |
| `AIE_TRY_INITIATION_INTERVAL(n)` | asks the pipeliner for an II |
| `AIE_LOOP_HINT(k, v)`, `AIE_LOOP_GPR_REALLOC` | backend loop hints |

Put each macro immediately before the `for`. Confirm it took effect: remarks'
`pass_failed` lists pragmas the compiler dropped.

## P05 `chess_storage(...)` is a no-op under Peano

It silently drops both alignment and bank placement. The same LUT was
32 B-aligned under Chess and 4 B-aligned under Peano. Use `alignas(32)` or
`alignas(aie::vector_decl_align)`.

## P06 Compiler crashes with a workaround

| Trigger | Symptom | Workaround |
|---|---|---|
| Subtracting two lane-extracted multiply results | `unable to legalize instruction: G_FSUB` | Keep it in an accumulator with `aie::msc` |
| `acc.to_vector<int8>(s)` feeding `aie::unpack` directly | SIGSEGV in instruction selection | Store the narrowed vector and reload it before unpacking |
| Any inline `asm(...)`, even `nop` | IRTranslator crash | Put assembly in a separate `.s` file |
| An f32 `vtanh` result | spurious recursion in aiecc's stack measurement | None recorded. Re-test on the current pin |
| 3 or 4 8x8 tiles per 64-lane accumulator group (prefill fv, current pin) | AsmPrinter crash | Stay at 2 tiles per accumulator (`levers.md` L18) |

The other rows were seen on older pins; re-test before relying on a
workaround.

## P07 Miscompiles

- An int16 `zero_scalar` loop at unroll 2 stored the loop offset instead of 0
  (8c560ac998f; device suite 870 pass / 3 fail → 871 / 2). Disabling the
  unroll fixed it (X10).
- Weight planes derived as `w + t*stride` all collapsed to plane 0
  (6d877c1bc62). Passing the planes as separate arguments fixed it.

Probe scalar store loops and derived pointer planes with poisoned outputs and
one-hot inputs (`aie-kernel-opt-hw` `measurement.md` §Gate).

**S3 (stale):** a byte-store loop at `-O2` lost the last store per unrolled
iteration on older Peano. It was fixed in `22.0.0.2026082201`, which the pin is
later than. Byte loops are still slow (L12).

## P08 Stack overflow is silent (AIE2 and AIE2P)

An overflow doesn't trap. It corrupts the neighbouring buffer, so suspect the
stack first when errors are small, scattered and row-local. In one internal
conv, 421 of 262144 values mismatched with max |Δ| 8, and the mismatches
cleared at a 2048 B stack.

- IRON's Worker default is 1024 B.
- aiecc measures each core's stack and errors when the declaration is short
  (`this core needs N bytes`). Declare what it measured.
- Before any device build, the remarks `stack_bytes` row gives the deepest
  frame path from the entry symbol and warns above the contract's budget.
  It leaves out the frames of any routine in the `libcalls` row.
- Anything that grows the frame (more unroll, more accumulators, trace markers
  at 64 B) needs a new declaration. For example flash needs 2304 / 3392 B.
- A change that cuts the stack can make the *old* build fail to fit. Snapshot
  the Python-side stack sizes along with the base arm.

## P09 `to_vector<int32>` lane order differs from `to_vector<int8>`

`mmul::to_vector<int32>` is a raw accumulator dump. `to_vector<int8>(shift)`
applies the row-major permutation. Indexing one as the other got 61454 of
65536 values wrong.

## P10 No data-driven gather

The `load_4x*` intrinsics bit-spread their addresses across lanes.
`parallel_lookup` with `lut<4, int8, int8>` fetches 32 lanes from only 16
unique inputs.

## P11 tanh takes an f32 argument for a reason

Feeding tanh a bf16-rounded argument multiplied silu's error by 1.35. This
affects accuracy, not speed.

There are two tanh builds on AIE2P, and each has its own binding constraint:

- **Native `vtanh`** is the default. It is latency-bound, and it returns x
  for |x| ≤ 0.5.
- **The LUT tanh** (`-DACTIVATIONS_TANH_LUT=1`) is table-load-bound.

Gate unrolls on `ACTIVATIONS_NATIVE_TANH`, and measure both builds (X22).

## P12 A two-limb bf16 split changes bits

2 × 8 < 24 significand bits. The two-limb split changed 12 / 32768 words
(gelu) and 11 / 32768 (silu). Use three limbs, or split only where the other
operand is exact (L03).

## P13 The software pipeliner gives up above MII 27

Peano's modulo scheduler stops at `SwpMaxMii` (27). A loop whose MII is
higher gets only the postpipeliner, and the remarks meta `schedule_notes`
shows it. bf16 `mm`'s k loop sits there (MII34, II35). Raising
`-pipeliner-max-mii` gave "Unable to find schedule". Treat such a loop as at
its bound (`levers.md` §Bounds) unless a change lowers its MII below 28.

A loop over the cap can still be the better design when it does more work
per trip. `partial_softmax`'s 8-row max fold has MII 38 and runs at 47
bundles per 8 rows under the postpipeliner, and the kernel went 4.8x faster
on hardware (`levers.md` S20). Compare per row, not per loop.

## P14 `pop()` then `pop_seek(odd)` reads the wrong blocks (AIE2P, bfp16)

On a bfp16 input stream, a `pop_seek` that directly follows a plain `pop`
lands correctly only for an even block stride. `vldb.pop.576` loads 512 b
and consumes 576 b, so what is left in the `lf` FIFO depends on history.
`mm_bfp`'s paired loop failed on hardware for K=24 and K=40 (3 and 5 blocks)
with all three data patterns, and passed for every even count. A
`pop_seek` after every `pop` passes at 3, 5 and 8 blocks. No aie_api doc
states `pop_seek`'s contract.

The compiler reports nothing and every even shape passes, so add a case with
an odd block count (`mm_bfp/32x24x48x4/bfp16ebs8/odd-k`) before the gate.

## `aie::exp2` is an interpolant

On AIE2P, `aie::exp2` computes `2^floor(x) · (1 + frac(x))`, which overshoots
true `exp2` by up to 6.15%. Keep `np.exp2` in references (`aie-kernel-opt-hw` `measurement.md`
§Gate). AIE2 has no `aie::exp2`; the cubic in `aie_kernels/common/exp2_bf16.h`
stands in (S80).

## Disproven beliefs

- **S9:** "mmul `a.grow` fusion bug." The mechanism was disproven. If a
  small-LSB mismatch appears around an int8 mmul, bisect it from scratch.
- "Missing `__restrict` costs 5-20x." No measurement backs the range. The
  measured L06 wins are 1.1x to 7.5x. On AIE2, restrict alone gave 7.5x
  (`add_weighted`, S58), 3.4x (`threshold`, S62) and 1.7x (`add`, S24).
