<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Measurement, attribution and correctness on hardware

This file gives the exact commands behind each step of SKILL.md, how to read
what they print, and the rules that stop a clean-looking number from
describing the wrong thing. Every rule here was made after a real number
turned out wrong.

Resolving the symbol, the marker audit, the base arm, the remarks report, the
unroll screen and the per-call prediction belong to `aie-kernel-opt-static`
(`references/static-checks.md` there). This file starts from its report.

Run everything from the repository root (`REPO=$PWD`) with the environment
from SKILL.md §Setup:

- `$K` is the kernel factory, and `$CASE` a case name from
  `test/python/npu/kernel_cases.py`.
- `$W` is a scratch directory of your own. `$BASE` is the base arm, a
  directory holding `aie_kernels/` and `aie_runtime_lib/`.
- `$CPUS` is a fixed host CPU list for `taskset`.
- `$NPU_LOCK` is the lock file shared by everyone on the device
  (`campaign.md`).

## Tool map

| Need | Tool |
|---|---|
| Correctness at every data case | `pytest test/python/npu/test_kernels_e2e.py -m extensive` |
| Cycles per call, wall clock, compile time, ELF bytes | `pytest test/python/npu/test_kernels_bench.py -m benchmark` (`kd.cycles_per_call` behind it) |
| Base against candidate: cycles, `npu_us` min, raw output words | the same bench with `--baseline-sources $BASE --bench-meta <file>` |
| Whether the entry symbol's markers bracket one whole call | `pytest test/python/test_kernel_trace_markers.py` (host only) |

## Gate (step 2)

```bash
pytest test/python/npu/test_kernels_e2e.py -m extensive -k "$CASE" --seeds 3
```

- `-k` is a substring match: `-k add` also runs `mul_add`. Check what it
  selects with `--collect-only -q` before a run you'll quote.
- The default (smoke) run covers only `smoke=True` cases. Use `-m extensive`
  to cover every case, every data case (`random` plus the contract's edge
  cases) and every seed. Test IDs are `<case>/<data_case>/s<seed>`.
- Include the remainder case the static report named, if it named one.
- Hardware-only failures need their own shape. `mm_bfp`'s paired `pop`/
  `pop_seek` passed every even block count and failed K=24 and K=40 on the
  device (static `traps.md` P14). A change to stream seeks, strides or
  blocking needs an odd-count or non-square case in the gate.
- **Mutation-prove the gate.** Break the kernel on purpose (drop a term, skip
  the tail, swap two loads), run the gate, and watch it fail. Then revert.
  - If a mutation survives, report it with the reason. Don't swap in a
    louder mutation to make the gate look good.
  - In one sigmoid epilogue, a dropped-term mutant changed 0 words because
    tanh absorbed it.
  - A mixed 64x32x32 matmul case failed 5 of 6 mutations that the square
    shapes passed.
  - A mutation run with `--baseline-sources` (§Raw diff) also shows whether
    any case reaches the path you broke.
- **Derive every tolerance** from the arithmetic, and size it over the
  longest case: one kernel measured 0.07 error at 4 calls and 0.25 at 256.
  - A bf16 round trip gets 1 ulp plus a floor for subnormal flush.
  - A bfp output gets a bound scaled to its output range.
- **Judge against the mathematics.** On AIE2P, `aie::exp2` is a linear
  interpolant that overshoots true `exp2` by up to 6.15%.
  - Keep `np.exp2` in the reference, with a derived ~7% envelope
    (`test/python/npu/test_mha_e2e.py:118`).
  - Native `vtanh` returns x for |x| ≤ 0.5. Judge the native build at its
    real error and the LUT build tightly.
- **The reference must not replay the kernel.** Write a single-pass reference
  in fp32, not the kernel's recurrence.
- **Gate every entry point alone**, and the composition once.
- Outputs are poisoned (`kd.upload(..., poison=True)`), so an unwritten
  output fails. Keep it that way. One-hot inputs found a weight-plane
  miscompile that no reading of the code did.

## Hardware cycles (step 3)

Build `$BASE` as in `aie-kernel-opt-static` `static-checks.md` §Base arm.
Then measure both arms in one run:

```bash
flock "$NPU_LOCK" taskset -c $CPUS \
  pytest test/python/npu/test_kernels_bench.py -m benchmark -k "[$CASE]" --no-compile \
  --baseline-sources $BASE --bench-out $W/bench.json --bench-meta $W/meta.json
```

- Each case is measured from this tree, then from `$BASE` right after it,
  with the same inputs. Both must pass the contract.
- The terminal summary prints one line per case: `cycles base -> cur`,
  `npu_us min base -> cur`, and `same` or `N differ` for the raw output
  words. `--bench-meta` holds the same under `baseline.cases.<case>` as
  `{cycles: [base, cur], npu_us_min: [base, cur], differing_words: N}`.
- The candidate is this checkout. To screen a candidate kept in its own copy
  (`$W/cand-<name>`), set `MLIR_AIE_KERNEL_SOURCES=$W/cand-<name>` on the
  command. The factories and cases still come from the installed Python, so
  both arms share their Python-side parameters (`campaign.md` F11).
- The JIT cache key includes `MLIR_AIE_KERNEL_SOURCES` and the core stack
  size, so the arms can't share a stale build and no cache wipe is needed.
- Bench test IDs are `test_kernel_benchmark[<case>]`, so `-k "[$CASE]"`
  (brackets included) selects exactly one case. Without the brackets,
  `-k "add/1024x16/bfloat16"` also ran `mul_add`. `-k` rejects `=`, so for a
  case like `dwconv1d/.../kernel_size=9/...` select `[dwconv1d/1040x16` and
  check the selection with `--collect-only -q`.
- `--pmode` only checks the device power mode (default: any). The nightly
  runs at `performance`. Record the mode with the numbers.
- `--no-compile` skips the cold-rebuild timing. Drop it when you want
  `compile_s` and the ELF byte rows.
- `--bench-out` is a list of `{name: "<case>/<metric>", unit, value, range?,
  extra}` rows for this tree only. It's written only if the whole session
  passed. `--bench-meta` is written either way.
  - `<case>/cycles` is the **min** of the kernel's intervals. Its `range`
    gives `median X max Y n=N`, then `init[i] min M` per traced initializer,
    then `truncated` if the trace buffer filled.
  - `npu_us` and `e2e_us` hold the median, with `range` giving
    `min X max Y n=N`. Quote the min (§Wall clock).
  - `extra` is the provenance: commit, Peano version, `kernel_sources`
    (when set) and `kernels <digest>`, a content hash of the kernel tree
    that ran.
- **Check the prediction.** Compare the measured cycles with the static
  report's "predicted per call".
  - A match validates both the static model and the measurement.
  - A miss means one of them is wrong. Find out which before you report
    either.

### Reading the cycles row

Every contract declares `trace=`: `Trace.whole_call()`,
`Trace.none(reason)` or `Trace.partial(reason)`. Only a `whole_call` kernel
is timed. The harness runs the setup kernel once, then per call the traced
initializers in contract order and the kernel last, and
`kd.split_intervals` labels the stream by that position. `kd.CallCycles`
holds `kernel`, `initializers`, `setup`, `truncated` and `untimed`.

| You see | Meaning | Action |
|---|---|---|
| A `<case>/cycles` row | The kernel's own intervals, split from any initializer's | Still confirm the static report's `[OK] name: symbol from source` line names *your* entry point |
| No `<case>/cycles` row | The contract declares `none` or `partial`. Library-wide that's `set_rounding`, `fused_mm` and the default `mha` build (a masked call returns before its markers) | Fall back to wall clock and say so. To time it, move one marker pair onto the entry so it brackets every call, declare `Trace.whole_call()`, and pass the marker audit |
| `expected E trace intervals, got M; a kernel on the core emits markers its contract's trace does not declare` | Something traced but undeclared: a marker around an inner loop, or an initializer that has markers but doesn't declare them | Run `pytest test/python/test_kernel_trace_markers.py`. It names the build and the marker it rejects |
| `truncated` in the range | The trace buffer filled. The row is the min over the calls that fit | The bench sizes the buffer from `kd.traced_intervals`, so this is rare. Compare `n=` with the case's `calls`, and flag n < calls/2 as undersampled |
| `the trace holds M intervals and none of the kernel's` | The buffer is too small, or the markers are missing | Run the marker audit |
| A `partial` initializer raises with its reason | Its intervals can't be told apart from the kernel's | Fix the initializer's markers first |

Why the split matters, on a GEMV with an instrumented `zero` (the raw stream
before the split existed):

| | raw stream | min | median | real kernel |
|---|---|---:|---:|---:|
| before | `2 ×16`, `283-295 ×16` | 2 | 142 | 291 |
| after | `2 ×16`, `127 ×16` | 2 | 64 | 127 |

With a stream you traced yourself, split it the same way: with K
instrumented kernels per call, population j is `intervals[j::K]`. Never
quote min or median over the mixed stream.

Several other checks apply to any row:

- Corroborate an initializer's `init[i] min` by the bytes it zeroes. 4096
  float (16384 B) took about twice the cycles of 4096 bf16.
- A matmul row that "improved" while its file didn't change is the
  initializer, or something else in its include closure. A 33% "matmul"
  gain in one sweep was entirely `zero`. Check `git diff <rev> -- <include
  closure>`.
- **Unchanged kernels must reproduce to the cycle across arms.** Select one
  alongside yours (`-k "[$CASE] or [<unchanged case>]"`). That's the proof
  that the arms differ only in your change.
- The markers cost stack: 64 B on one kernel. Re-check the `stack_bytes` row
  after adding them (`aie-kernel-opt-static` `traps.md` P08).

### Clock and dispatch

- Convert cycles to time at the **measured** 1.76 GHz (Strix-class AIE2P; it
  measured core_fraction 1.004 on a long compute-bound kernel).
- `core_fraction = cycles × calls ÷ 1.76e9 ÷ npu_seconds`, from the cycles
  row and the `npu_us` min. Below 0.5, the row is dispatch-bound at this
  shape.
  - Judge the kernel on cycles.
  - If the production shape is also dispatch-bound, kernel work won't pay
    off, so hand off to `aie-dataflow-opt`.
- A 16-call dispatch costs about 60-90 µs whatever the core does. One kernel
  that went 15x faster on the core moved wall clock by 30%.
- Example: `add/1024x16/bfloat16` went 390 → 150 cycles per call, while
  `npu_us` read 85.16 → 90.53 (median). Its core_fraction is about 0.015, so
  the wall clock says nothing about the kernel at this shape.

### Wall clock (only when the kernel can't be traced)

- Quote **`min`** (the `range` column, or `npu_us_min` in the baseline
  table), with the host pinned (`taskset`), never the mean. Two
  byte-identical builds showed a phantom 7.5% on the mean (X07).
- The noise band across 18 byte-identical kernels was −19.1 to +13.3 µs,
  plus about 3%. A delta must clear both.
- **Rep slope.** Measure two `Case`s that differ only in `calls`, and
  difference the times: `(t20 − t4) / 16` is the per-call cost with dispatch
  cancelled. A bfp16 shuffle read −56% on wall clock and 10.4x per call by
  slope. If the second case doesn't exist, add it to `kernel_cases.py`
  (and to `benchmark_series.txt`).
- Before you believe a delta, confirm the artifacts differ: compare the
  `core_elf_bytes` rows (run without `--no-compile`), or `cmp` the core
  ELFs.

## Raw diff (bit-exactness)

A tolerance pass isn't proof of exactness: `fused_mm` passed 27 of 27 cases
with a dropped term (X11). To claim "bit-identical", read `differing_words`
from the `--baseline-sources` run in step 3. Both arms get the same inputs,
and the count compares stored bits, not values within a tolerance. `same`
on every case is the claim.

- **Mutation-prove the diff.** Run once more with a deliberately broken
  candidate, and expect a nonzero count. The real patch changed 0 of 32768
  words, while the two mutants changed 12 and 11. A diff that catches
  nothing measures nothing.
- Reordering accumulation isn't bit-identical. Say so, and report the max
  |Δ| against the base arm's hardware output.

## Ablation (attribution)

- **Across kernels:** gate each kernel call behind a flag, keeping the
  acquires and releases. Re-time, and rank the kernels by their delta. That's
  `aie-dataflow-opt`'s first step. Do it before this skill.
- **Inside a kernel:** replace one region with a cheap wrong computation,
  and trace. The bf16 GEMV went 1154 → 634, which priced its reduction at
  520 cycles. Six kernel-local tweaks then lost (X02-X04); restructuring the
  reduction (static L20) won 1154 → 788. An ablation price tells you where
  the time is, not that it's fixed.

## Provenance

Every bench row's `extra` carries the commit, the Peano version,
`kernel_sources` and the `kernels` digest. The digest changes with any edit
to the kernel tree that ran, committed or not. Next to every number you
quote, also record:

- `git status --short` over the kernel's include closure, not only its own
  file
- a timestamp
- `$BASE` and what it was archived from

A shared-tree all-clear once came from a teammate's uncommitted fix. When two
numbers for one kernel conflict, compare their provenance (the `kernels`
digests first) before their values.
