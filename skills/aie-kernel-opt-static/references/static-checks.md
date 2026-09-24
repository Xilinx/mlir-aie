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

`TODO(d-tools:Gn)` marks a command that stands in for a missing in-repo
capability, listed in `GAPS.md` under that ID. When the in-repo option lands,
the marked command is replaced by it.

## Tool map

| Need | Tool |
|---|---|
| Loop II, stage count (`ns`), pipelined or not, ZOL, bundles and bytes per loop, dropped pragmas, `pm_bytes` | `python -m aie.utils.compile.remarks` |
| Contract agrees with its factory; the design lowers to MLIR; declared stack covers known minimums | `pytest test/python/test_kernel_contracts.py` (host only) |
| Libcalls, spills, frame size, `.text` per symbol | `llvm-nm`, `llvm-objdump`, `llvm-readelf`, `llvm-size` on the objects remarks leaves behind (`$PEANO_INSTALL_DIR/bin`) |

## Resolve the called symbol (step 1)

```bash
grep -n "def $K\b" -A40 python/iron/kernels/*.py    # source file, -D flags, extern "C" symbol per dtype
grep -n "Case(\"$K" test/python/npu/kernel_cases.py   # the cases, their calls= and kwargs
grep -n 'event0()' <selected source and its includes> # TODO(d-tools:G4) marker audit in test_kernel_contracts.py
```

- Follow the factory to its `extern "C"` symbol, then to the function that
  symbol calls. A `.cc` often holds siblings the factory never selects
  (in-place and out-of-place, vector and `_scalar`). An II33 → 18 win on an
  uncalled gelu variant measured 594 → 594 on hardware (`levers.md` S14).
- Record the production `-D` flags. Remarks compiles each factory at its
  defaults and at each `.dtypes` entry, so a flag a case sets through kwargs
  may be missing from the remarks build. Check the flags in the build
  directory (TODO(d-tools:G8d) per-case builds).
- **Marker check.** Note whether the selected entry point has `event0()`. If
  it doesn't, a traced cycle number for it would belong to another kernel on
  the core, usually the `zero` initializer. Put "markers: yes/no" in the
  report so the HW skill knows before it runs.

## Contract check (step 2)

```bash
pytest test/python/test_kernel_contracts.py -k "$K" -q
```

It needs no device. It checks that the roles match `arg_types()`, that the
reference takes what the contract hands it, and that a design lowers to MLIR.
A new `Case` you add (for example a remainder case) is picked up here too.
Also add the case name to `test/python/npu/benchmark_series.txt`, or
`test_benchmark_series_names.py` fails.

## Arms (step 3)

An arm is a directory holding `aie_kernels/` and `aie_runtime_lib/`. The
remarks tool (and the JIT, for the HW skill) honor
`MLIR_AIE_KERNEL_SOURCES=<arm>`. TODO(d-tools:G6) arm pairing in the tools.

```bash
BASE=$W/base; mkdir -p $BASE
# Nobody else editing siblings: the whole tree at the base revision.
git archive <rev> aie_kernels aie_runtime_lib | tar -x -C $BASE
# Shared tree, or your file includes a sibling (mha.cc includes mm.cc and softmax.cc; many include zero.cc):
cp -r aie_kernels aie_runtime_lib $BASE/ && git show <rev>:<file> > $BASE/<file>   # per file you changed
diff -rq aie_kernels $BASE/aie_kernels   # must list only your files
```

The candidate arm is the live checkout (`$REPO`), or a copy per candidate
(`$W/cand-<name>`) when you screen several.

## Static report (steps 3 and 5)

```bash
for ARM in base:$BASE cand:$REPO; do
  MLIR_AIE_KERNEL_SOURCES=${ARM#*:} python -m aie.utils.compile.remarks --target aie2p \
    --only "^$K" --out $W/${ARM%%:*}-rows.json --meta $W/${ARM%%:*}-meta.json
done
```

- Use `--target aie2` for npu1. The command exits 3 if any build fails to
  compile. A compile failure is a finding: report it.
- Build names are `<factory>` or `<factory>/<k>=<v>` per dtype.
- Rows: `<name>/unpipelined_loops`, `non_zol_loops`, `missing_bank_loads`,
  `pass_failed_warnings`, `pm_bytes`, and `loop/<fn>/<bb>/II`.
- Meta: `loops`, keyed `fn/bb`, each with `ii`, `ns` (stage count),
  `pipelined`, `pipeliner`, `missed_reason`, `zol`, `bundle_count`,
  `byte_count`, `file` and `line`; also `pass_failed` (pragmas the compiler
  dropped) and `schedule_notes` (`MII`, `SwpMaxMii`, "Unable to find
  schedule").
- **Diff the two meta files per loop**: `ii`, `ns`, `pipelined`, `zol`,
  `byte_count`. A change that moves no static metric is not a change; revert
  it.
- `schedule_notes` with MII above `SwpMaxMii` (27) means only the
  postpipeliner ran (`traps.md` P13). bf16 `mm` sits there at MII 34 / II35 (`levers.md`
  §Bounds).

### Objects

The workdir is `aie-static-*` under the system temp directory, with one
`build<i>/` per build. The tool doesn't print it, so take the newest
directory (TODO(d-tools:G8e)):

```bash
O=$(ls -td ${TMPDIR:-/tmp}/aie-static-* | head -1); find $O -name '*.o'
llvm-nm -u <obj> | grep ' __'                           # TODO(d-tools:G8a) libcalls; traps.md P01 lists the costly ones
llvm-objdump -d --no-show-raw-insn <obj> | grep -c '\[sp, #'   # spill and stack traffic
llvm-readelf --stack-sizes <obj>                        # TODO(d-tools:G8g) frame per function
llvm-nm -S --size-sort <obj>                            # TODO(d-tools:G8b) .text per symbol
```

- A P01 helper called from a hot loop is L02. Confirm the call site with
  `llvm-objdump -d` (a `jl` to the helper inside the loop).
- `[sp, #...]` inside a loop body is a spill.
- **Stack.** Compare the entry point's frame plus its callees against the
  contract: `python -c "from aie.iron import kernels; print(kernels.$K().contract.stack_bytes)"`.
  compiler-rt helpers emit no `.stack_sizes` (`__divsf3` added 64 B that the
  measurement missed). A frame over the budget makes the candidate `reject`
  until the contract is raised in the same change (`traps.md` P08).
- **`.text` per symbol, not per object.** An object's `.text` counts every
  sibling and helper; one helper was counted three times. Making a helper
  `static` halved the object and saved 0 B in the ELF (`levers.md` S16).
- **A `noinline` straight-line body** (such as the exp emulation) reports its
  *caller's* loop II. Count the bundles in the callee with `llvm-objdump` and
  divide by the elements per call (TODO(d-tools:G8c)).

### Unroll screen (before L07)

Copy the kernel into two scratch arms, `x1` and `x4`, that differ only in
the pragma, and run remarks on both.

- Loop `byte_count` flat, or nearly flat, at ×4: the body is latency-bound.
  Candidate class `strong` (S01).
- `byte_count` roughly ×4, or II ×4: resource-bound. `reject` (tanh stayed
  at II4, X23).
- A new `[sp, #`, or a higher `pm_bytes` for nothing: `reject`.

Screen 2 and 8 too when 4 is borderline. The winner depends on the body:
cast was best at 8, most activations at 4.

### Remainder case

An unroll or a new blocking can create a path no existing case reaches: an
unroll remainder, a non-square shape. Add a `Case` that reaches it, for
example 160 elements after `UNROLL(4)` on a 32-lane body, or a mixed
64x32x32 matmul (it failed 5 of 6 mutations that the square shapes passed).
Name it in the report, so the HW skill gates it. TODO(d-tools:G13) proof
that a case reaches the tail.

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
