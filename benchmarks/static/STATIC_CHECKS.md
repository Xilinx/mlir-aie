<!-- Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# Static kernel checks

CPU-only: every registry kernel is compiled with Peano exactly as the JIT
compiles it, plus optimization-record flags, and the records become
per-kernel series. No NPU is involved, so a Peano bump that changes a
loop's schedule shows up here before anyone looks at device numbers.

Everything below was read off real builds with llvm-aie
22.0.0.2026090201 (the pin in `utils/peano-requirements.txt`) on both
`aie2` and `aie2p`. The record shapes are Peano's, not LLVM's documented
ones, and they matter: the loop-scheduling pass reports as `pipeliner`
(a `postpipeliner` filter records nothing), its give-up record is
`Missed/canPipelineLoop` with no loop name, and the three passes name
the same loop differently.

## Record shapes

| Pass | Kind / Name | Args | Tracked as |
| --- | --- | --- | --- |
| `pipeliner` | `Passed` / `schedule` | `II`, `NS`, `Loop`, `Pipeliner`, `PrologueBundles`, `EpilogueBundles` | `loop/<fn>/<bb>/II`; hover text carries NS, prologue/epilogue, ZOL, the engine and `at <file>:<line>` |
| `pipeliner` | `Missed` / `canPipelineLoop` | `String` ("Failed to pipeline loop"); loop located by `DebugLoc` only | `unpipelined_loops` (keyed `L<line>`) |
| `pipeliner` | `Analysis` / `schedule` | `MII`, `SwpMaxMii`, "Unable to find schedule" | `schedule_notes` in the meta file |
| `aie-hardware-loops` | `Analysis` / `analysis` | `BasicBlock`, `Zero-Overhead-Loop` | `non_zol_loops` |
| `aie-asm-printer` | `Analysis` / `analysis` | `BasicBlock`, `BundleCount`, `ByteCount` | `pm_bytes` (summed per function) |
| `aie-multi-slot-pseudo` | `Missed` / `missing-memory-bank` | `Instruction` | `missing_bank_loads` |
| stderr | `-Wpass-failed` | a `#pragma clang loop` / `AIE_*` macro the compiler dropped | `pass_failed_warnings`, with the text kept |

The pipeliner names loops by machine basic block (`bb.1.for.body.i`);
the other two passes use the IR block (`for.body.i`). `remarks._block`
strips the prefix so II, ZOL and bundle counts land on one loop.

Flags (`remarks.REMARK_FLAGS`): `-fsave-optimization-record
-foptimization-record-passes='pipeliner|aie-hardware-loops|aie-asm-printer|aie-multi-slot-pseudo'`,
plus `-Rpass` / `-Rpass-missed` / `-Rpass-analysis` for the same passes
so the log carries readable copies.

`unpipelined_loops` counts every loop the pipeliner declined, including
outer loops it never intends to pipeline, so the absolute number means
little and the change means everything: a bump that newly declines an
inner loop raises it by one.

## How the kernels are compiled

`run.py:compile_command` builds the command with
`aie.utils.compile.utils.cxx_core_compile_command`, the function the
JIT's `compile_cxx_core_function` runs, so the target triple, warning
set, defines and section flags cannot drift from what ships in a design.
Only the record flags and four alignment warnings (`-Wcast-align
-Walign-mismatch -Wunaligned-access -Wframe-larger-than=1024`) are
added. Source, include directories and flags come from the read-only
`ExternalFunction` accessors; inline sources (the aie2 LUT activations)
are written to a file first, as the JIT does. Chess-built kernels are
rejected: the remarks are Peano's.

The target's device (`NPU1Col1` for aie2, `NPU2Col1` for aie2p) is set
before any factory runs, since factories pick their source and
`mac_dims` from the current device.

The workflow installs `mlir_aie` from the published wheel rather than
building the PR. Two steps make a PR run speak to the PR: the checkout's
`aie.iron.kernel`, `aie.iron.kernels`, harness, bfp, verify, benchmark
and compile-recipe files are copied over the wheel's (only those, since
mixing whole packages across wheel versions produces circular imports),
and `--source-root $GITHUB_WORKSPACE` rewrites every wheel path the
factories hand out (a `source_file`, an include directory, the aie2 LUT
factories' inline `#include`) to the checkout.

## What a run looks like

Every registry case with a distinct factory + kwargs compiles: 68
kernel builds on aie2p and 55 on aie2 (the transformer blocks,
`convert_copy`, `exp2f_vec`, `conv2dk14`, `dwconv1d` and the bfp16ebs8
matmuls are aie2p-only), in a few minutes on one core. That yields 561
rows on aie2p (221 per-loop II) and 432 on aie2 (152 II). A sample from
aie2p:

| loop | II | via |
| --- | --- | --- |
| `eltwise_add_bf16_vector / for.body.i` | 12 | postpipeliner |
| `eltwise_add_bf16_scalar / for.body.i` | 88 | MachinePipeliner |
| `reduce_add_vector / for.body.i` | 2 | postpipeliner |
| `reduce_add_scalar / for.body.i` | 10 | MachinePipeliner |
| `matmul_bf16_f32 / for.body34.i.i` | 35 | postpipeliner (MII 34 > SwpMaxMii 27 noted) |
| `zero_f32 / for.body.i` | 3 | postpipeliner |

`missing_bank_loads` is non-zero for about half the kernels on both
targets. `pass_failed_warnings` is zero for every kernel but one: the
uint8 vector path of `aie_kernels/aie2/conv2dk3.cc` puts
`AIE_LOOP_UNROLL_FULL` on `for (j = 0; j < kernel_width; j++)` at three
sites, and `kernel_width` is a runtime argument, so the compiler reports
"loop not unrolled" and drops the pragma (`AIE_LOOP_RANGE(3, 3)` beside
it does not make the bound a constant). That is a kernel-source fix -- a
constant 3, which the kernel assumes anyway -- not a checker setting.

## Regression rules

All series are integers and deterministic. `II`, `unpipelined_loops`,
`non_zol_loops`, `missing_bank_loads` and `pass_failed_warnings` alert
on any increase (101 %). `pm_bytes` is recorded as its own group
(`--out-pm`, `bench/static-pm/`) at 103 %: a Peano bump routinely moves
program memory by a few bytes, and an alert on that teaches reviewers to
ignore the comment.

Nothing here gates. A dropped pragma is a kernel-source bug, so its text
is kept (`StaticReport.pass_failed`) and, under Actions, `run.py` emits
it as a `::warning file=<checkout path>,line=..` workflow command, which
GitHub renders on the pull request's Files tab; a kernel that fails to
compile becomes one `::error` with the first `error:` line. A warning on
a wheel header, outside the checkout, is attached to no file.

## Runs

`staticKernelChecks.yml`: ubuntu-latest, both targets. Nightly runs
record the baseline at `bench/static/<target>/`. Pull requests run when
they change what gets compiled: `utils/peano-requirements.txt`,
`aie_kernels/`, `aie_runtime_lib/`, `python/iron/kernels/`,
`benchmarks/` or the workflow itself. `utils/clone-llvm.sh` is not a
trigger: the LLVM pin builds mlir-aie, and this checker only runs
Peano's clang. A kernel that fails to compile invalidates the run (exit
3, no JSON, nothing recorded). Fork PRs run too; only the alert comment
needs a same-repo token, and the job summary carries the table either
way.

## Not implemented

- Alignment: assert every `aie.buffer` address in
  `input_with_addresses.mlir` is a multiple of the largest vector width
  the kernel uses (64 B on AIE2/AIE2P). Zero cost; belongs in
  `harness.measure_compile`.
- On-device UBSan: build the correctness suite once with
  `-fsanitize=alignment,undefined,bounds -fsanitize-trap=all`. Needs a
  device to observe the trap.
- `aiecc` saves no optimization records itself; recompiling here is the
  only source of II.
