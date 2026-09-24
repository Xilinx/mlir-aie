<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Measurement, attribution and correctness on hardware

This file gives the exact commands behind each step of SKILL.md, how to read
what they print, and the rules that stop a clean-looking number from
describing the wrong thing. Every rule here was made after a real number
turned out wrong.

Resolving the symbol, the marker check, arms, the remarks report, the unroll
screen and the per-call prediction belong to `aie-kernel-opt-static`
(`references/static-checks.md` there). This file starts from its report.

Run everything from the repository root (`REPO=$PWD`) with the environment
from SKILL.md §Setup:

- `$K` is the kernel factory, and `$CASE` a case name from
  `test/python/npu/kernel_cases.py`.
- `$W` is a scratch directory of your own. `$BASE` and `$CAND` are the two
  arms.
- `$CPUS` is a fixed host CPU list for `taskset`.
- `$NPU_LOCK` is the lock file shared by everyone on the device
  (`campaign.md`).

`TODO(d-tools:Gn)` marks a command that stands in for a missing in-repo
capability, listed in `GAPS.md` under that ID. When the in-repo option lands,
the marked command is replaced by it.

## Tool map

| Need | Tool |
|---|---|
| Correctness at every data case | `pytest test/python/npu/test_kernels_e2e.py -m extensive` |
| Cycles per call, wall clock, compile time, ELF bytes | `pytest test/python/npu/test_kernels_bench.py -m benchmark` (`kd.cycles_per_call` behind it) |
| Cycles for a case without `trace_cycles`, a population split, raw outputs | the INTERIM probe (§Probe) |

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

Build the arms as in `aie-kernel-opt-static` `static-checks.md` §Arms
(TODO(d-tools:G6) arm pairing in the bench). Then:

```bash
for ARM in base:$BASE cand:$CAND; do
  rm -rf $W/cache-${ARM%%:*}                     # TODO(d-tools:G11) cache key misses Python-side params
  NPU_CACHE_HOME=$W/cache-${ARM%%:*} MLIR_AIE_KERNEL_SOURCES=${ARM#*:} \
    flock "$NPU_LOCK" taskset -c $CPUS \
    pytest test/python/npu/test_kernels_bench.py -m benchmark -k "[$CASE]" \
    --no-compile --bench-out $W/${ARM%%:*}-bench.json --bench-meta $W/${ARM%%:*}-meta-bench.json
done
```

- Bench test IDs are `test_kernel_benchmark[<case>]`, so `-k "[$CASE]"`
  (brackets included) selects exactly one case. Without the brackets,
  `-k "add/1024x16/bfloat16"` also ran `mul_add`.
- `--pmode` only checks the device power mode (default: any). The nightly
  runs at `performance`. Run both arms at the same mode and record it.
- Run the two arms back to back in one session, and use a fresh
  `NPU_CACHE_HOME` for each. A 16 B stack size "passed" 12 tests in 0.48 s
  from cache and failed all 12 after a wipe. A run that finishes
  suspiciously fast compiled nothing.
- The JSON output is a list of `{name: "<case>/<metric>", unit, value,
  range?, extra}` rows. It's written only if the whole session passed.
  - `<case>/cycles` is the median of `cycles_per_call`.
  - `npu_us` and `e2e_us` hold the median, with `range` giving
    `min X max Y n=N`.
- `extra` stamps the checkout's `HEAD`, not the arm (TODO(d-tools:G7)).
  Write the arm's source directory next to each number yourself
  (§Provenance).
- **No `<case>/cycles` row** means the kernel's contract lacks
  `trace_cycles=True`, which today is set only for passthrough
  (TODO(d-tools:G1)).
  - Check the static report's "markers" line, then use the probe.
  - Flip `trace_cycles` in the contract only if one marker pair brackets
    exactly one whole call. That's a change to the kernel's Python factory;
    commit it with the kernel.
- `RuntimeError: ... expected N whole-call trace intervals, got M` means the
  count is wrong. Read M as in §Reading intervals (TODO(d-tools:G3)).
- **Check the prediction.** Compare the measured cycles with the static
  report's "predicted per call".
  - A match validates both the static model and the measurement.
  - A miss means one of them is wrong. Find out which before you report
    either.

### Reading intervals

| M vs `calls` N | Meaning | Action |
|---|---|---|
| M = N | One instrumented region per call | Still confirm `event0()` is in *your* entry point. A clean single-valued row can be `zero` |
| M = 2N (or K·N) | K instrumented kernels per call, usually `zero` then the kernel | Split by order: population j is `intervals[j::K]` (TODO(d-tools:G2)). Never quote min or median over the whole stream |
| M > N, not a multiple | A marker brackets an inner tile or chunk, or an initializer runs at another rate (once per output tile) | Dump the stream in order and find the period. Not a per-call number until split |
| M < N | The trace buffer filled (bench `TRACE_SIZE` is 16384 B). `add/1024x256` gave "expected 256 whole-call trace intervals, got 91", and the session wrote no JSON | Use the probe with a larger trace size, or fewer calls (TODO(d-tools:G5)). Flag M < N/2 as undersampled |
| M = 0 | No `event0()` in the selected source | No cycle number exists. Fall back to wall clock, and say so |

Why the split matters, on a GEMV with an instrumented `zero`:

| | raw stream | min | median | real kernel |
|---|---|---:|---:|---:|
| before | `2 ×16`, `283-295 ×16` | 2 | 142 | 291 |
| after | `2 ×16`, `127 ×16` | 2 | 64 | 127 |

Several other checks apply to any row:

- Corroborate an initializer population by the bytes it zeroes. 4096 float
  (16384 B) took about twice the cycles of 4096 bf16.
- A matmul row that "improved" while its file didn't change is the
  initializer. A 33% "matmul" gain in one sweep was entirely `zero`. Check
  `git diff <rev> -- <include closure>`.
- **Unchanged kernels must reproduce to the cycle across arms.** That's the
  proof that the arms compiled different code and nothing leaked through a
  cache.
- The markers cost stack: 64 B on one kernel. Re-check the stack after adding
  them (`aie-kernel-opt-static` `traps.md` P08).

### Clock and dispatch

- Convert cycles to time at the **measured** 1.76 GHz (Strix-class AIE2P; it
  measured core_fraction 1.004 on a long compute-bound kernel).
- `core_fraction = cycles × calls ÷ 1.76e9 ÷ npu_seconds`. Below 0.5, the row
  is dispatch-bound at this shape (TODO(d-tools:G9)).
  - Judge the kernel on cycles.
  - If the production shape is also dispatch-bound, kernel work won't pay
    off, so hand off to `aie-dataflow-opt`.
- A 16-call dispatch costs about 60-90 µs whatever the core does. One kernel
  that went 15x faster on the core moved wall clock by 30%.
- Example: `add/1024x16/bfloat16` went 390 → 150 cycles per call, while
  `npu_us` read 85.16 → 90.53 (median). Its core_fraction is about 0.015, so
  the wall clock says nothing about the kernel at this shape.

### Wall clock (only when the kernel can't be traced)

- Quote **`min`** from the `range` column, with the host pinned (`taskset`),
  never the mean. Two byte-identical builds showed a phantom 7.5% on the
  mean (X07).
- The noise band across 18 byte-identical kernels was −19.1 to +13.3 µs,
  plus about 3%. A delta must clear both.
- **Rep slope.** Build at two call counts, run each, and difference the
  times: `(t20 − t4) / 16` is the per-call cost with dispatch cancelled. A
  bfp16 shuffle read −56% on wall clock and 10.4x per call by slope.
  - Bench has no `--calls` option (TODO(d-tools:G5), TODO(d-tools:G10)).
  - Use the probe with `calls=` overridden, or two cases.
- Before you believe a delta, confirm the artifacts differ: compare the
  `core_elf_bytes` rows, or `cmp` the core ELFs.

## Probe

**INTERIM.** This probe stands in for TODO(d-tools:G1), G2, G3, G5 and G12.
Delete this section, and switch every step that uses it, once d-tools lands
the in-repo equivalents in `test_kernels_bench.py`, `kd.cycles_per_call` and
the e2e/bench raw-output option.

It works on any case, whether or not its contract sets `trace_cycles`. It:

- judges the case
- saves the raw output words and all trace intervals
- prints populations split by order

It uses only in-repo APIs, the same calls `test_kernels_e2e.py` and
`kd.cycles_per_call` make. Save it as `$W/probe.py` and run it from
`test/python/npu`.

```python
import sys, tempfile
from pathlib import Path
import numpy as np
from aie.iron import kernels
from aie.iron.algorithms import kernel_design as kd
from aie.utils.trace import TraceConfig
from aie.utils.trace.utils import get_cycles_summary
from cases import inputs_for
from kernel_cases import CASES

name, out_path = sys.argv[1], sys.argv[2]
trace_size = int(sys.argv[3]) if len(sys.argv) > 3 else 262144
case = next(c for c in CASES if c.name == name)
fn = case.fn()
inputs = inputs_for(case, "random", np.random.default_rng(0))
design = kd.design(getattr(kernels, case.factory), **case.harness_opts(),
                   params=fn.param_values(inputs), **case.kwargs)
out_n, out_dt = kd.output_size(fn, calls=case.calls), fn.output_dtype()

ins, out = kd.upload(inputs, out_n, out_dt, fn=fn, poison=True)
outs = out if isinstance(out, tuple) else (out,)
design(*ins, *outs)
got = [o.numpy().copy() for o in outs]
verdict = fn.judge(got if len(got) > 1 else got[0],
                   fn.expected(inputs, scalars=case.scalars), calls=case.calls)

work = Path(tempfile.mkdtemp(prefix="probe-"))
cfg = TraceConfig(trace_size=trace_size, trace_file=str(work / "trace.txt"))
ins, out = kd.upload(inputs, out_n, out_dt, fn=fn)
design(*ins, *(out if isinstance(out, tuple) else (out,)), trace_config=cfg)
cfg.trace_to_json(cfg.physical_mlir_path, str(work / "trace.json"))
iv = [int(d) for p in get_cycles_summary(str(work / "trace.json")) for d in p[1:]]
np.savez(out_path, *[g.view(f"u{g.itemsize}") for g in got], intervals=np.array(iv))

print(f"{name}: ok={bool(verdict)} {verdict.detail} calls={case.calls} intervals={len(iv)}")
k, r = divmod(len(iv), case.calls)
if r or not k:
    print("  not a whole multiple of calls: truncated trace, inner-region markers,"
          " an initializer at another rate, or no markers")
for j in range(k):
    p = sorted(iv[j::k])
    print(f"  pop {j}: min {p[0]} med {p[len(p) // 2]} max {p[-1]} n={len(p)}")
```

```bash
cd test/python/npu
for ARM in base:$BASE cand:$CAND; do
  rm -rf $W/cache-${ARM%%:*}
  NPU_CACHE_HOME=$W/cache-${ARM%%:*} MLIR_AIE_KERNEL_SOURCES=${ARM#*:} \
    flock "$NPU_LOCK" taskset -c $CPUS python $W/probe.py "$CASE" $W/${ARM%%:*}.npz
done
```

- The kernel is the population whose values scale with its work. The
  initializer is the one that scales with bytes zeroed.
- `ok=False` means the numbers are void. Fix correctness first.
- Pass a third argument to raise the trace size when intervals < calls.

## Raw diff (bit-exactness)

A tolerance pass isn't proof of exactness: `fused_mm` passed 27 of 27 cases
with a dropped term (X11). To claim "bit-identical", compare the raw output
words of the two arms on identical inputs. The INTERIM probe saves them with
a fixed seed (TODO(d-tools:G12)).

```python
import numpy as np, sys
a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
for k in a.files:
    if k != "intervals":
        print(k, int((a[k] != b[k]).sum()), "of", a[k].size, "words differ")
```

- **Mutation-prove the diff.** Run a deliberately broken third arm, and
  expect a nonzero count. The real patch changed 0 of 32768 words, while the
  two mutants changed 12 and 11. A diff that catches nothing measures
  nothing.
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

Next to every number, record (TODO(d-tools:G7)):

- the commit, and `git status --short` over the kernel's include closure, not
  only its own file
- a timestamp
- the Peano version (`extra` has it)
- which source directory each arm compiled from

A shared-tree all-clear once came from a teammate's uncommitted fix. When two
numbers for one kernel conflict, compare their provenance before their
values.
