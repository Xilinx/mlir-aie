<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# Static checks: exact commands and how to read them

This file backs each step of SKILL.md with the exact in-repo commands, and
the rules that stop a clean-looking static number from describing the wrong
thing. None of it needs a device.

Run everything from the repository root (`REPO=$PWD`) with the environment
from SKILL.md §Setup. `$K` is the kernel factory (for example `rms_norm`) and
`$W` is a scratch directory of your own.

## Tool map

| Need | Tool |
|---|---|
| Loop II, stage count (`ns`), pipelined or not, ZOL, bundles and bytes per loop, dropped pragmas, `pm_bytes`, libcalls, stack depth, entry symbol and source per build | `python -m aie.utils.compile.remarks` |
| Before → after of every one of those rows | the same, with `--baseline-sources $BASE` |
| Entry symbol's `event0()`/`event1()` bracket one whole call, as its contract's `trace=` declares | `pytest test/python/test_kernel_trace_markers.py` (host only, lit) |
| Contract agrees with its factory; the design lowers to MLIR; declared stack covers known minimums | `pytest test/python/test_kernel_contracts.py` (host only) |
| Spill traffic, a helper's call site, bundles in a `noinline` body | `llvm-objdump` (`$PEANO_INSTALL_DIR/bin`) on the objects remarks keeps with `--keep` |

## Resolve the called symbol (step 1)

```bash
grep -n "def $K\b" -A40 python/iron/kernels/*.py    # source file, -D flags, extern "C" symbol per dtype, trace=
grep -n "Case(\"$K" test/python/npu/kernel_cases.py   # the cases, their calls= and kwargs
python -m aie.utils.compile.remarks --target aie2p --only "^$K" --out $W/rows.json --meta $W/meta.json
```

- Remarks prints `[OK] <build>: <symbol> from <source>` for every build.
  That line is the called symbol and the file it comes from. Follow the
  symbol to the function it calls. A `.cc` often holds siblings the factory
  never selects (in-place and out-of-place, vector and `_scalar`). An II33
  → 18 win on an uncalled gelu variant measured 594 → 594 on hardware
  (`levers.md` S14).
- Record the production `-D` flags. Remarks compiles each factory at its
  defaults and at each `.dtypes` entry. If a case sets a different flag
  through its kwargs, the remarks rows describe the default build, not the
  case's. Say so in the report.
- **Marker check.** Read the contract's `trace=`:
  - `Trace.whole_call()`: one `event0()`/`event1()` pair brackets every call
    of the entry symbol, and the bench will chart its cycles.
  - `Trace.none(reason)` or `Trace.partial(reason)`: no cycle number will
    exist for it. Put the reason in the report.

  `pytest test/python/test_kernel_trace_markers.py` checks every build's
  declaration against its compiled IR, following callees across
  `#include`. Run it after any change that moves, adds or removes a
  marker, or adds an early return. Put "markers: whole_call / none /
  partial" in the report so the HW skill knows before it runs.

## Contract check (step 2)

```bash
pytest test/python/test_kernel_contracts.py -k "$K" -q
```

It needs no device. It checks that the roles match `arg_types()`, that the
reference takes what the contract hands it, and that a design lowers to MLIR.
A new `Case` you add (for example a remainder case) is picked up here too.
Also add the case name to `test/python/npu/benchmark_series.txt`, or
`test_benchmark_series_names.py` fails.

## Base arm (step 3)

An arm is a directory holding `aie_kernels/` and `aie_runtime_lib/`. Remarks
and the JIT compile from `MLIR_AIE_KERNEL_SOURCES=<arm>` when it is set, and
both tools take a base arm as `--baseline-sources <arm>`.

```bash
BASE=$W/base; mkdir -p $BASE
# Nobody else editing siblings: the whole tree at the base revision.
git archive HEAD aie_kernels aie_runtime_lib | tar -x -C $BASE
# Shared tree, or your file includes a sibling (mha.cc includes mm.cc and softmax.cc; many include zero.cc):
cp -r aie_kernels aie_runtime_lib $BASE/ && git show <rev>:<file> > $BASE/<file>   # per file you changed
diff -rq aie_kernels $BASE/aie_kernels   # must list only your files
```

The candidate arm is the live checkout, or a copy per candidate
(`$W/cand-<name>`, selected with `MLIR_AIE_KERNEL_SOURCES`) when you screen
several. The factories and their `-D` flags come from the installed Python
either way, so both arms share them.

## Static report (steps 3 and 5)

```bash
python -m aie.utils.compile.remarks --target aie2p --only "^$K" \
  --out $W/rows.json --meta $W/meta.json --keep $W/objs --baseline-sources $BASE
```

- Use `--target aie2` for npu1. The command exits 3 if any build fails to
  compile. A compile failure is a finding: report it.
- Build names are `<factory>` or `<factory>/<k>=<v>` per dtype.
- Per build it prints `[OK] name: symbol from source`, then any `dropped
  pragma:`, `calls the runtime library: <symbols>`, and a `stack:` warning
  when the deepest frame path is over the contract's `stack_bytes` (else
  the device default).
- With `--baseline-sources` it compiles every selected build again from
  `$BASE` and ends with `baseline <dir> -> this tree: N rows differ`, one
  `name: before -> after` line per row. That list is the static A/B.
- Rows: `<name>/unpipelined_loops`, `non_zol_loops`, `missing_bank_loads`,
  `pass_failed_warnings`, `pm_bytes`, `libcalls` (a count; the names are in
  the range), `stack_bytes`, and per loop `loop/<fn>/<bb>/II` and
  `loop/<fn>/<bb>/not_zol`. The loop counts and `pm_bytes` cover only the
  functions the entry symbol reaches, which are what the core link keeps.
- Meta, per build: `source`, `symbol`, `object`, `libcalls`, `stack_bytes`,
  `stack_budget`, `pm_bytes_by_function`, and `loops`, keyed `fn/bb`, each
  with `ii`, `ns` (stage count), `pipelined`, `pipeliner`, `missed_reason`,
  `zol`, `bundle_count`, `byte_count`, `file` and `line`; also `pass_failed`
  (pragmas the compiler dropped) and `schedule_notes` (`MII`, `SwpMaxMii`,
  "Unable to find schedule").
- Meta describes this tree only. For per-loop `ns` and `byte_count` on the
  base arm too, run remarks a second time with
  `MLIR_AIE_KERNEL_SOURCES=$BASE` and its own `--meta`, and diff the two
  per loop.
- A change that moves no static metric is not a change; revert it.
- `schedule_notes` with MII above `SwpMaxMii` (27) means only the
  postpipeliner ran (`traps.md` P13). bf16 `mm` sits there at MII 34 / II35
  (`levers.md` §Bounds).

### Libcalls, stack and objects

- **Libcalls.** A non-zero `libcalls` row names runtime-library routines
  (`__divsf3`, `__mulsf3`, ...). A `traps.md` P01 helper called from a hot
  loop is L02. Confirm the call site with `llvm-objdump -d` on the kept
  object (a `jl` to the helper inside the loop).
- **Stack.** The `stack_bytes` row is the deepest frame path from the entry
  symbol. It excludes the runtime routines' own frames when `libcalls` is
  non-zero (`__divsf3` added 64 B that an earlier measurement missed).
  Compare it with `stack_budget` in meta. A frame over the budget makes the
  candidate `reject` until the contract's `stack_bytes` is raised in the
  same change (`traps.md` P08).
- **Spills.** `--keep $W/objs` keeps one `build<i>/` per build (the base
  arm's under `$W/objs/baseline/`), and meta's `object` names each `.o`.
  `[sp, #...]` inside a loop body is a spill:

  ```bash
  llvm-objdump -d --no-show-raw-insn <object> | grep -c '\[sp, #'
  ```

- **Bytes per symbol.** `pm_bytes` and `pm_bytes_by_function` count only
  the shipped functions. An object's whole `.text` counts every sibling and
  helper; one helper was counted three times that way. Making a helper
  `static` halved the object and saved 0 B in the ELF (`levers.md` S16).
- **A `noinline` straight-line body** (such as the exp emulation) is its own
  function in `pm_bytes_by_function` and reports its *caller's* loop II.
  Count the bundles in the callee with `llvm-objdump` and divide by the
  elements per call.

### Unroll screen (before L07)

Copy the kernel tree into two scratch arms, `x1` and `x4`, that differ only
in the pragma, and run remarks on `x4` with `--baseline-sources` set to `x1`
(`MLIR_AIE_KERNEL_SOURCES=$W/x4 ... --baseline-sources $W/x1`). For the
loop `byte_count`, keep a `--meta` from each arm.

- Loop `byte_count` flat, or nearly flat, at ×4: the body is latency-bound.
  Candidate class `strong` (S01).
- `byte_count` roughly ×4, or II ×4: resource-bound. `reject` (tanh stayed
  at II4, X23).
- A new `[sp, #`, a higher `stack_bytes`, or a higher `pm_bytes` for
  nothing: `reject`.

Screen 2 and 8 too when 4 is borderline. The winner depends on the body:
cast was best at 8, most activations at 4.

### Remainder case

An unroll or a new blocking can create a path no existing case reaches: an
unroll remainder, a non-square shape. Add a `Case` that reaches it, for
example 160 elements after `UNROLL(4)` on a 32-lane body, or a mixed
64x32x32 matmul (it failed 5 of 6 mutations that the square shapes passed).
Name it in the report, so the HW skill gates it. Its gate mutation (break
the tail, watch that case fail) is the proof that the case reaches the tail.

## Predict (step 6)

For the loop you changed, write down the per-call prediction from the object:

```
cycles ≈ bundles_outside_loop + 5 + (trips - 1) × II      (per loop, summed)
```

Take the trip count from the loop-count setup in the disassembly. Precedent:
`zero` predicted ~75 / ~130 and measured 78 / 134; `mm_bfp_mixed` matched to
within one cycle (S05). `q4nx_dequant`'s 1-cycle II drop over 128 trips
predicted ~2117 and measured 2115 (S09).

- Put the prediction in the report as "predicted, not measured".
- Never convert an II ratio into a speedup. The prefill `fv` II ratio said
  -70% and hardware gave -35% (S10).
