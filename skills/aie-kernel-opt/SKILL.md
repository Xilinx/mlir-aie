---
name: aie-kernel-opt
description: Make one compiled AIE kernel faster, and prove it. For C++ kernels built by Peano (llvm-aie) for AIE2P (npu2) or AIE2 (npu1) in bf16, float, int8 or int16, from elementwise and normalization kernels to matmul, GEMV, attention and conv, in aie_kernels/ or the user's own .cc. Use when a loop has a high II, won't pipeline, spills, overflows its stack or calls __mulsf3, __divsi3 or another libcall; when the user wants to vectorize or restructure a kernel, or port one architecture's tuned code to the other; or when they want to know whether a kernel change is really faster or bit-identical on the NPU. Drives the in-repo aie.utils.compile.remarks report and its base-arm diff, the trace-marker audit, the test_kernels_e2e gate and the test_kernels_perf back-to-back A/B. Carries the levers that measured faster on hardware and the traps that compile cleanly and run wrong. Not for tile placement or DMA bandwidth (aie-dataflow-opt), or for writing a first kernel (aie-code-creator).
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AIE kernel optimization

This skill makes **one compiled kernel** faster. It assumes the kernel is
already correct on hardware (`aie-hw-bringup`) and worth the work
(`aie-dataflow-opt` ranks kernels by ablation).

**Evidence rule.** The static report tells you what to try and predicts the
result. Only two arms measured back to back on the NPU, as traced cycles per
call of the kernel itself, show a speedup. Only a raw output-word diff shows
bit-exactness. Without a device, report a static change as "candidate, HW
unconfirmed", never as faster: an II33 → 18 change measured 594 → 594 (it
edited a symbol no factory calls), an II37 → 31 change measured slower, and
rolling int8 `mm`'s K loop, a uniform static win, measured +3..+28% on half
the shapes (51d0a8e000a).

## Setup

```bash
source /opt/xilinx/xrt/setup.sh            # XRT first, or pyxrt is missing
source <venv>/bin/activate
source utils/env_setup.sh install          # from the repository root (docs/Building.md)
export MLIR_AIE_KERNEL_SOURCES=$PWD        # else "this tree" is build/include's stale copy
W=$(mktemp -d); K=<factory>; CASE=<case name from kernel_cases.py>; CPUS=<fixed host CPU list>
```

Remarks takes `--target aie2p` for npu2 (Strix, Krackan) or `--target aie2`
for npu1 (Phoenix); `xrt-smi examine` shows which device you have. What an
intrinsic lowers to is in `third_party/aie_api/include/aie_api/detail/<arch>/`.

## Workflow

### 1. Resolve what runs

```bash
grep -n "def $K\b" -A40 python/iron/kernels/*.py      # source, -D flags, extern "C" symbol, trace=
grep -n "Case(\"$K" test/python/npu/kernel_cases.py     # cases, calls=, kwargs
python -m aie.utils.compile.remarks --target aie2p --cases test/python/npu/kernel_cases.py \
  --only "^$K" --out $W/rows.json --meta $W/meta.json
```

- Remarks prints `[OK] <build>: <symbol> from <source>` per build. Follow
  that symbol to the function it calls and edit only that. A `.cc` often
  holds siblings no factory selects (in-place, `_scalar`, another dtype).
- `--cases` compiles exactly the builds the cases run, named by case
  (4c8c4d91ff4); it first prints `cases: N builds for M cases ...`, and a
  case missing from M is a finding. Without it remarks builds only factory
  defaults and `.dtypes`, which missed every stride-2 and sized `-D` build
  of the conv kernels. For a shape no case has, use
  `--build FACTORY:KEY=VALUE,...` (repeatable; 3dfa82d81af).
- Read the contract's `trace=`. Only `Trace.whole_call()` gets a cycles row.
  `Trace.none(reason)` and `Trace.partial(reason)` leave you with wall clock.
- **Grep the vector path's guard.** Every condition on a runtime scalar
  (not width or alignment) needs a case on each side, at the values the
  consumer design passes. MobileNet passed `skip_scale` 0, the guard was
  `> 0`, no case used 0, and three layers ran scalar at 579k-927k cycles a
  call, 16.2 of the network's 16.3 ms. Relaxing the guard took the network
  to 0.77 ms (9e649eda9d7).
- **Time the case at the design's scalars.** A case at other widths can
  take another path. `xy_pool` read 820 at `outC` 120, but MobileNet
  passes 960 padded to 1280, and a scalar pad loop ran at 3066 a call
  until it stored vectors (875, 8fafbef845b). bn2's dw row (56 wide) ran
  the generic path because the 8-pixel path stopped at 32 pixels: 4302 →
  840 (eb89013e5d8). A gap that is the same on every row is per-call work,
  not a DMA stall.

### 2. Gate, and prove the gate

```bash
pytest test/python/test_kernel_contracts.py -k "$K" -q           # host only
pytest test/python/npu/test_kernels_e2e.py -m extensive -k "$CASE" --seeds 3 \
  --report-error $W/err.json > $W/gate.log 2>&1
```

- `-m extensive` runs every case, every data case (random plus the
  contract's edge cases) and every seed. `-k` is a substring match (`-k add`
  also runs `mul_add`); check the selection with `--collect-only -q`.
  Keep the whole log: an unreproducible failure needs its words.
- `--report-error` records per case `max_ulp`, `not_correctly_rounded`,
  the worst index and the reference precision judged against
  (8f749a06d52); a "correctly rounded" claim must quote a `float64` entry.
  For a unary bf16 kernel run the whole domain:
  `aie.utils.accuracy.all_bf16(nan=False)` against `round_to(ref64, bfloat16)`.
- **Mutation-prove it.** Break the kernel on purpose (drop a term, skip the
  tail, swap two loads), watch the gate fail, revert. A surviving mutant
  is a missing case or dead work: decide which. Report it with the reason;
  don't swap in a louder one. Mutate data, not addresses, on aligned paths
  (a 64 B-aligned `load_v` ignores a 32 B offset), and count only random
  data runs (zeros pass most mutants).
- **Derive every tolerance** from the arithmetic, over the longest case
  (one kernel's error was 0.07 at 4 calls and 0.25 at 256).
- **The reference must not replay the kernel.** Write a single-pass fp32
  (or float64) reference, and judge against the mathematics (see the exp2
  and tanh traps below). Define the semantics at every scalar the kernel
  accepts; a scalar path that is undefined there (a shift by -1) is not a
  reference.
- **One case per parameter and per path.** A check case for each factory
  parameter away from its default (`conv2dk14` second tile group: 404/1024
  wrong, fae15d5b998), each fallback dtype (`cascade_mm` bf16 fallback: max
  |Δ| 1.091 → 1.7e-6, aa3d8476a03), and a size at which an unrolled or
  versioned loop actually runs, not only its fallback.
- **Misaligned views.** A design that packs several weights into one
  buffer hands them out at offsets like 16 mod 64; kernel tests bind each
  argument alone and aligned. Test it with `Case(...,
  arg_byte_offsets=((idx, 16),))` (c7d229b9316), and run it on the base
  kernels first so it fails. Declare `alignments` in the contract for
  what the fast path loads (d8e877d26fc).
- Keep outputs poisoned (`kd.upload(..., poison=True)`; the core's output
  tile is poisoned too, 36bd3b637a7) and guarded (`kd.design(...,
  guard=True)`), so an unwritten output or a write past a tile's end fails.
  Gate every entry point alone, and the composition once.

### 3. Build the base arm

An arm is a directory holding `aie_kernels/` and `aie_runtime_lib/`.
Remarks and the JIT compile from `MLIR_AIE_KERNEL_SOURCES=<arm>` when it is
set, and both take `--baseline-sources <arm>` for the comparison.

```bash
BASE=$W/base; mkdir -p $BASE
git archive HEAD aie_kernels aie_runtime_lib | tar -x -C $BASE
# Shared tree, or your file includes a sibling (mha.cc includes mm, softmax, zero):
cp -r aie_kernels aie_runtime_lib $BASE/ && git show <rev>:<file> > $BASE/<file>
diff -rq aie_kernels $BASE/aie_kernels     # must list only your files
```

The factories, cases and `-D` flags come from the installed Python in both
arms, so snapshot Python-side parameters (stack sizes, geometry tables) you
change along with the arm. Archive right before the A/B: on a shared
branch an older archive lacks others' later commits and credits their gains
to you.

### 4. Read the static report

```bash
python -m aie.utils.compile.remarks --target aie2p --cases test/python/npu/kernel_cases.py \
  --only "^$CASE" --out $W/rows.json --meta $W/meta.json --keep $W/objs --baseline-sources $BASE
```

- Per build: `[OK]`, then any `dropped pragma:`, `calls the runtime
  library: <symbols>`, and a `stack:` warning over the contract's budget.
  It exits 3 if a build fails to compile; that is a finding, and the
  failure prints the first `error:`/`LLVM ERROR`/assertion lines
  (9872411c243). The builds that compiled still get rows and a diff
  (d7a37082e05). With `--baseline-sources` it ends with `N rows differ` and
  one `name: before -> after` line per row: that is the static A/B.
  - Both summaries name the `aie_kernels/` they compiled and warn when
    `MLIR_AIE_KERNEL_SOURCES` is unset or both arms are one tree
    (0e867c0d9c4): the comparison is void.
  - Loops LLVM only renumbered are paired and reported as `M loops renamed
    with the same rows, not listed` (2f11f40eefb).
  - `0 rows differ` after an edit you expected to move something: check
    that the hunk landed (`diff -r` against the arm) before concluding.
- Rows: `unpipelined_loops`, `non_zol_loops`, `missing_bank_loads`,
  `pass_failed_warnings`, `pm_bytes`, `libcalls`, `kernel_stack_bytes`, and
  per loop `loop/<fn>/<bb>/II` and `not_zol`. They cover only functions the
  entry symbol reaches.
- A loop with a constant trip count also gets `II_x_trips` (a0340f8676b).
  Compare it, not II, when a change unrolls or widens a body: the llama
  transpose went II 26 → 27 but II_x_trips 3328 → 864, and HW 3344 → 880.
  Runtime-count loops have no trips; multiply nested loops yourself.
- Meta per loop: `ii`, `ns` (stage count), `pipelined`, `zol`,
  `bundle_count`, `byte_count`, `file`, `line`; per build `pass_failed` and
  `schedule_notes` (`MII`, `SwpMaxMii`). Meta covers this tree only; for
  the base arm's loops run again with `MLIR_AIE_KERNEL_SOURCES=$BASE`.
  `ns` 1 means the iterations don't overlap: the II is one iteration's
  critical path, and it moves unpredictably with unrelated edits.
- Trust the loop row, not a pipeliner remark: a copy loop "found II 3"
  and was emitted without a ZOL at II 7-8.
- Why a loop missed its II: `remarks.compile_command(ef, target, dir)`
  gives the exact clang++ line. Add `-mllvm -debug-only=pipeliner` for
  `MII = N (rec=R, res=S)` and Found/Rejected per loop,
  `-debug-only=postpipeliner` for ResMII/RecMII, or `-mllvm -debug`
  grepped for `^PLI` for each rejected II's reason.
- Spills: `[sp, #` inside a loop body in the kept object.
  ```bash
  $PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn <object> | grep -c '\[sp, #'
  ```
- Size: use `pm_bytes` / `pm_bytes_by_function`, not an object's `.text`,
  which counts siblings the link drops.
- Stack: `kernel_stack_bytes` is the kernel's own call path (398e3d9b77e);
  the core adds main's frame (+128 B on AIE2P, +96 on AIE2 in the kernel
  harness). Size a contract's `stack_bytes` from aiecc's `this core needs
  N`, over every size the cases use.
- A `noinline` body reports its caller's II; count its bundles yourself.
- **Other architecture unchanged.** Run remarks with the other `--target`
  and `--keep` in both arms; `0 rows differ` covers only the builds it
  compiled. Compare per section (`llvm-objdump -s -j .text.<symbol>`),
  since a moved source line renumbers `.Ltmp` labels in `-d` output.

### 5. Diagnose, then change one thing

| Report or object shows | Lever |
|---|---|
| The other architecture has a tuned branch; this one runs portable code | L21 |
| `libcalls` names a helper called in a loop | L02; L11 for `__divsi3` |
| `vector<float>` multiply, min or max in a loop | L03; L04 on AIE2P |
| Array of accumulators or vectors indexed by a loop counter; `[sp` traffic | L01 |
| `kernel_stack_bytes` > 0 on a leaf kernel with no arrays | L25 |
| The loop you care about isn't innermost or single-block; unpipelined parent | L10 |
| Loop body branches on its counter | L13 |
| Address arithmetic in the body; no `__restrict`; `ns` 1 with load → op → store | L06 |
| Runtime trip count, not pipelined or `ns` 1 | L22 |
| Pipelined, many empty bundles; `byte_count` flat from ×1 to ×4 | L07 |
| `ns` 1, one long chain per vector (activation, exp, epilogue) | L07 |
| Stepping fewer lanes than the accumulator holds; one wide load per four V-ops | L08 |
| int8 `mmul<4,8,8>` on AIE2P | L23 |
| One long mac chain, or broadcasts spilled per block; one A load per output block | L09 |
| First mac waits on loading C | L28 |
| MAC loop pipelined, epilogue after it isn't | L27 |
| Pipelined, `ns` 3, one long latency chain with no dominant step | L17 |
| Several `mmul<8,8,8>` accumulators for Y += S·V; high II, big frame | L18 |
| A `reduce_add`/`reduce_max` per row that the row loop never overlaps | L20 |
| bfp16 stream state (`sfl/sfh`, FIFO) spilled around `vst.push`/`vldb.pop` | L19 |
| `(tanh+1)/2`, or separate constant multiplies | L05 |
| Table gather and arithmetic in one loop | L26 |
| RecMII holds an unaligned-store loop with a small body (AIE2P) | L24 |
| Runtime shape the factory knows at build time; dead chunk-count bodies | L29 |
| `lda.s8`/`st.s8` byte loop | L12 |
| Scalar int8 requantize tail | L14 |
| Scalar gather building the mmul A operand | L15 |
| Body is only a rearrangement | L16 |
| II already at a resource or latency bound | stop (§When to stop) |

- One change per candidate. The lever's Check must move in the
  `--baseline-sources` diff; if no static metric moved, revert.
- Before an unroll, screen it: two scratch arms that differ only in the
  pragma, remarks on `x4` with `--baseline-sources $W/x1`. Loop
  `byte_count` flat at ×4 means latency-bound: go. Roughly ×4, a new
  `[sp, #`, or more stack: stop. Try 2 and 8 when 4 is borderline.
- If the change creates a path no case reaches (an unroll remainder, a
  non-square shape, an odd block count, an odd row length), add a `Case` to
  `test/python/npu/kernel_cases.py` and its name to
  `test/python/npu/perf_series.txt` (`test_perf_series_names.py` checks).
  `check(...)` cases are gated but never timed.
- A change that hides work from LLVM (an opaque pointer bump, `volatile`, a
  LICM blocker) to get a lower II has measured slower (prefill `fv`: static
  II37 → 31, HW 1374-1576 against 1265).
- Write the per-call prediction before measuring:
  `bundles_outside_loop + 5 + (trips - 1) × II`, per loop, trips from the
  loop-count setup in the object. Compare II per unit of work (per row,
  value or tile), not per loop: a raw II that rises can still win. Never
  turn an II ratio into a speedup (prefill `fv`'s said -70%; HW gave -35%).
  A loop that isn't a pipelined ZOL defeats the model: int8 `mm`'s
  predicted a wash and measured -19% (d3b4b66fd7a).
- Re-run `test_kernel_contracts.py` and the gate.

### 6. Measure both arms back to back

```bash
taskset -c $CPUS pytest test/python/npu/test_kernels_perf.py -m perf -k "$K" \
  --baseline-sources $BASE --perf-out $W/perf.json --perf-meta $W/meta.json
```

- Each case runs from this tree, then from `$BASE`, with the same inputs.
  The summary prints `cycles base -> cur` and each arm's `min..max n=` for
  cycles and `npu_us`, `same` or `N differ` raw output words, and each
  arm's error vs the reference (3b507540694, 8f749a06d52). `--perf-meta`
  holds the same under `baseline.cases.<case>` (`cycles_range`,
  `npu_us_range`, `accuracy` as [base, cur]).
- **Time every case of the kernel**, not the one you targeted. Every case
  is one command, and a change often wins one shape and loses another: 8
  pixels per mmul won six bn 1x1 shapes and lost relu 28/40/120 2236 →
  2853 (147bbd32f8d); an unrelated loop in the same function moved II 103
  → 105.
- `-k "[$CASE]"`, brackets included, selects exactly one case. `-k`
  rejects `=`; select a prefix and check with `--collect-only -q`. `-k`
  subsets still write their rows.
- The baseline is timed even if it fails this tree's contract
  (a82f73e3e27) and rebuilt with the stack it needs (16d203e06d2); a
  failing candidate still gets its words diffed (74550f29195). Each
  `check(...)` case gets an untimed `test_kernel_same_as_baseline[<case>]`
  words line (56a68449765); select it by node id.
- To screen a candidate kept in its own copy, set
  `MLIR_AIE_KERNEL_SOURCES=$W/cand-<name>`; the JIT cache key includes it
  and the design generator's own helpers (d4f088355b5). After editing IRON
  or `aie.iron.kernels`, use a fresh `NPU_CACHE_HOME`. To A/B compile flags
  without editing a shared factory, wrap `ExternalFunction.__init__` in a
  pytest `-p` plugin.

Verdict:

| You see | Verdict |
|---|---|
| Candidate's max below base's min; the unchanged kernel reproduces to the cycle | confirmed |
| Ranges overlap, or the delta sits inside an unchanged kernel's spread | no change: revert |
| Candidate higher | rejected: revert |
| The unchanged kernel moved | void: the arms differ in more than your change; fix and rerun |

Check the result against the prediction (`zero` predicted ~75 / ~130,
measured 78 / 134); if they disagree, find out which is wrong first.

### 7. Exactness and record

- "Bit-identical" means every case reads `same` in step 6's run. A
  tolerance pass isn't proof: `fused_mm` passed 27 of 27 cases with a
  dropped term, and a mutation that changed 45,159 words passed rtol 0.04.
  Mutation-prove the diff too, with a mutant inside the tolerance
  (`set_rounding(ceil)`, one LSB, a dropped low limb): the words line must
  show it. A mutant that only touches values random data never produces
  (-NaN lanes) reads `same`. If the change reorders accumulation, say so and
  give the max |Δ|. Integer paths with an exact contract are proved by
  the gate itself.
- **Build every design that links the kernel** and read each core's program
  memory and stack. A core holds every kernel and size variant it calls in
  16384 B: bn 1x1 relu + skip grew 800 + 848 → 8944 + 10736 B and
  overflowed a MobileNet core the kernel tests never see (a9cfcbc2130,
  1d9069f69d7). Designs that build their own Workers use the 1024 B default
  stack, not the contract's.
- One commit per kernel and change, with base → after cycles per case, n,
  the arm sources and the gate line. List each rejected variant with the
  number that killed it. NO-CHANGE with a measured bound is a valid result.
  A rejected lever is worth retrying after a structural change: the
  whole-granule store lost before a row-width specialization and won after
  it (1c32c246a33).
- Record provenance with every number: the row's `extra` (commit, Peano,
  `kernels` digest) and what `$BASE` came from. `git status` at measure time
  is not provenance on a shared tree; the JIT cache is:
  `$NPU_CACHE_HOME/objects/<key>/` holds a copy of every `.cc` it compiled
  (the `.o.d` depfiles name them), so diff those against `git show
  <rev>:<path>`. When two numbers conflict, compare digests first.
- Several agents on one NPU: wrap every hardware command in `flock` on one
  shared lock file, and build base arms by copying the live tree.
- Keep every `extern "C"` name, signature and buffer layout. Leave
  `*_scalar` variants alone.

## Reading hardware rows

`<case>/cycles` is the **min** of the kernel's own intervals, split from its
initializers' by position (`kd.cycles_per_call`). Its `range` gives
`median X max Y n=N` and `init[i] min M` per initializer. The harness
traces markers only and flushes after the last call, so every call arrives
or the test fails with `N of M calls traced` (53f84828939); intervals over
524,288 cycles are unwrapped (62739edc49f).

| You see | Do |
|---|---|
| No `<case>/cycles` row | The contract is `none`/`partial` (in the library, only `set_rounding`), or the case is a `check`. Use wall clock and say so, or move one marker pair to bracket the whole call and declare `Trace.whole_call()` |
| `expected E trace intervals and up to F flush pairs` | A marker the contract doesn't declare. `pytest test/python/test_kernel_trace_markers.py` names the build |
| `N of M calls traced` | A stale design or a marker pair too short to flush; use a fresh `NPU_CACHE_HOME`, then run the marker audit |
| A row moved but its file didn't | Something in its include closure changed: `git diff <rev> -- <closure>`. A 33% "matmul" gain was all `zero.cc` |
| A stream you traced yourself, K kernels per call | Population j is `intervals[j::K]`. Never quote min or median over the mixed stream. Your own `pipeline(...)` flushes with `Stage.trace_flush` |

- A cycle row names the design, not the kernel. Before crediting a delta,
  confirm the `[OK]` line names the entry you changed.
- A fast kernel's max carries the NPU's stall tail (`add` 46, max 68):
  compare mins.
- `core_fraction = cycles × calls ÷ 1.76e9 ÷ npu_seconds` (1.76 GHz
  measured on AIE2P). Below 0.5 the case is dispatch-bound: a 16-call
  dispatch costs 60-90 µs whatever the core does. Judge the kernel on
  cycles; if production is dispatch-bound too, hand off to
  `aie-dataflow-opt`.
- Wall clock, only when the kernel can't be traced: quote the pinned
  **min** (`npu_us_min`), never the mean (it showed 7.5% between
  byte-identical builds). For the per-call cost, difference two cases that
  differ only in `calls`: `(t20 − t4) / 16`.
- Ablation prices a region: replace it with a cheap wrong computation and
  trace. The bf16 GEMV went 1154 → 634, pricing its reduction at 520. A
  price is where the time goes, not a bound: six local tweaks lost, then
  restructuring the reduction (L20) won 1154 → 788.

**Whole-design latency** (to confirm a kernel win in the design that uses
it, c0991c9950f):

- The NPU ramps for 6-13 launches after load (MobileNet 5826, 1604, 947, …
  732 µs); one warmup and five iterations read ~1.6x high. Warm up ≥ 20.
- Time back to back on one hardware context. A context switch reloads the
  whole PDI (+2.85 ms for 3.95 MB), and after ≥ 50 ms idle even an empty
  transaction costs ~340 µs against 55 µs.
- Quote the min and record `uptime`: at load average 9-15 MobileNet's
  median spread over 725-850 µs while its min held 695-712.
- For a base arm of the whole design, `git archive` the base's Python too,
  put it first on `PYTHONPATH`, and check one known base number before
  trusting the rest (MobileNet 176.4 ms against 176.2).
- Find the layer that paces the design before optimizing one: make the
  kernel return at entry for that layer's shape and time the design. The
  noop is live only if the output changed (MobileNet's residuals can keep
  it passing, so check that `max_difference` moved). bn2's dw noop took
  MobileNet 604 → 561-569 µs; the fix then reached 557-588 (eb89013e5d8).
  Noops of bn6's and bn12's dw moved nothing.
- The min spread 533-571 µs over 38 runs of one build, so a Δmin under
  ~15 µs from two ABAB pairs is noise: four 3x3 layers read −9..−13 µs,
  then +1..+3 in three more pairs.
- Compile each arm once outside the NPU lock (`--xclbin-path`,
  `--insts-path`, the arm's `NPU_CACHE_HOME`); a cached timed run holds it
  ~4 s.

## Levers that measured faster

Each gives the change, what must move in the static report where it isn't
obvious, and one hardware result (npu2 cycles per call unless noted;
"model" means wall clock in a model outside this repository).

**L01 Fully unroll loops that index register arrays.** A short loop
indexes `acc[i]` or calls `insert`/`extract` with its counter; `[sp` around
the accumulators. `AIE_LOOP_UNROLL_FULL` makes each index constant. Check:
stack traffic gone, the enclosing loop gets an II. `conv2dk1_i8` 7623 → 504.
A full unroll over a size parameter grows the frame with it: chunk the
loop (`dwconv1d_cl` 8608 → ≤ 1280 B at C=960, cycles unchanged,
8343faea7b9). A loop that runs exactly once lets LICM hoist its loads into
the parent and spill them (1216 B at width 56, 8bd78ee2771).

**L02 No libcalls in hot loops.** The `libcalls` row names a helper (see
the traps). Use `aie::max`/`min` for compares, `aie::inv` times a multiply
for divides, `aie::to_float` for int → float, integers for index math;
reduce a per-chunk max element-wise and horizontally once at the end.
Check: helper gone, loop gets an II. `rms_norm` 1597 → 438.

**L03 Avoid the f32 vector multiply (AIE2P).** Skip multiplies by a known 1
or 0. If one operand is exact in bf16 (a tanh result, a chosen constant),
split only the f32 operand into **three** bf16 limbs (residuals via
`aie::msc`; two limbs change bits) and mac each against it. Matmul silu
epilogue 8098 → 3213, bit-identical. When both operands are f32, three
limbs each (six products) still beat the emulation: `bf16_exp` 13459 →
1866 and correctly rounded (c785995b735); `exp2f_vec` 14125 → 2174, max
ulp 1476 → 115 (f376889eab8). Issue the high-limb products first and the
low ones last (2226 → 1937). A numpy bit model (float64 product, one f32
rounding) predicted the hardware words exactly. Clamp without f32
`min`/`max`: compare bf16 results against bounds rounded the same way, or
select on the integer bit pattern (`mm` relu 1106 → 274, 2f76486caed);
NaN compares as a large integer, so handle it explicitly if the contract
needs it (1937 → 2174).

**L04 Run emulated f32 chains 32 lanes wide (AIE2P).** A 16-lane emulated
f32 multiply pays for 32 lanes. Step 32 lanes; replace a vector
`to_fixed`/`to_float` floor with the magic-number floor `x + 1.5*2^23`.
Check: bundles per element drop. `bf16_exp` 32482 → 15425 (with other
changes).

**L05 Fold constants into one mac.** Write `(tanh+1)/2` as one mac into an
accumulator preloaded with 0.5; fold constant scales together
(power-of-two scaling is bit-exact). `sigmoid` 498 → 118.

**L06 Walking `__restrict` cursors.** Peano doesn't strength-reduce
`base[i*stride]`. Advance a `T *__restrict` cursor, mark non-aliasing
in/out pointers `__restrict` (not for in-place kernels), and use
`add_2d_byte`/`add_3d_byte` for multi-dimensional walks
(`aie_kernels/quant/q4nx_dequant.cc`). Check: II and body `[sp` drop.
`zero<float,4096>` 519 → 262. `AIE2_RESTRICT` expands to nothing on AIE2P;
an AIE2P loop at `ns` 1 with a load → op → store chain wants plain
`__restrict` too: `bitwise_and` 306 → 40 (3714284145d), `threshold` 400 →
103 (b797494776c), `add_weighted` 1134 → 136 (50ee7db49d8), a walked
`reduce_add` 215 → 85 (0b841e34862), `bn_conv2dk3` II 47 → 27
(8bd78ee2771). Gate it per architecture so the other `.text` doesn't move.

**L07 Unroll latency-bound bodies**, after the unroll screen passes. Not
on load-, resource- or register-bound bodies. Check: II per element drops,
no new `[sp`. `leaky_relu` 298 → 86. `UNROLL(2)` on `fused_mm`'s epilogue
chunk loop: gelu 2582 → 1919. A loop at `ns` 1 whose vectors are each one
long serial chain (split, tanh, exp, mul) gets its overlap only from
independent vectors in one body: unroll under `VERSIONED_LOOP` (L22) with
a minimum count ≥ the unroll. `mm` epilogue silu ×4 3143 → 1101, gelu ×2
3090 → 1597 (eb01ca2eef1); `bf16_exp` 3021 → 1866 (c785995b735);
`expand` 162 → 92 (de5c90cb91d). Stop where the ZOL or the stack goes
(silu ×8 lost its ZOL and slowed).

**L08 Fill the accumulator.** Step 64 lanes on 8/16-bit data and
`AIE_BF16_LANES` (`aie_kernels/aie_arch.h`) on bf16 arithmetic. Check: trip
count halves at a similar II. `rope` 1932 → 298 (with L06). On AIE2P
a bf16 `vmac.f` writes a whole 2048-bit accumulator, so 64 lanes cost
what 32 do: `layer_norm` 581 → 380 (b34923542bc), `rms_norm` 438 → 308
(aed4fa9129f), `dwconv1d_cl` 98 → 75 (e3cbc517fe7). 8-bit `sliding_mul`
at 64 lanes over whole blocks: `filter2d` 4702 → 387 (5dfab7ef261). int32
into `acc64` at 32 lanes, where the tell was one 64 B load per four V-ops:
`scale` 338 → 180 (2d22b268d28).

**L09 Split dependency chains.** Two independent accumulators added at the
end, or one broadcast feeding several row blocks; stop before spills.
int16 `mv/32x32` 291 → 127. Name them (`acc0, acc1, …`): an `acc[4]` array
did not scalarize, and bf16 `reduce_add` went 140 → 84 once named
(1a21fed7cab). The same for mmuls: five output-channel accumulators
sharing each input load took `xy_pool` 4447 → 1366 (7bf129aa3da);
requantize each to a small vector before the next group's epilogue, or
the live accumulators spill.

**L10 Make the hot loop innermost and single-block**, the only loops Peano
pipelines: fully unroll a short K reduction into its parent, fold nested
tile loops into one counter, split a per-tile `if` into straight loops. Opt
in per dtype. int8 `mm` 993 → 737. `UNROLL(2)` on an unpipelined outer loop
overlaps two trips' loads and stores: `fused_mm` 1084.5 → 991.5.

**L11 Unsigned counted trip.** Compute the trip count up front as unsigned;
a signed divide or shift by 2^k is `__divsi3`. `axpy` 317 → 178. A counted
trip alone gives a ZOL, not overlap; see L22.

**L12 Wide stores, never byte loops.** Peano won't pipeline an
`lda.s8`/`st.s8` loop. Copy with `uint64_t`/`uint32_t` or 32 B vector
stores, both ends aligned. bfp16 shuffle 10.4x per call.

**L13 `UNROLL_FULL`, not `RANGE`, on a loop that switches on its counter.**
`AIE_LOOP_RANGE` is only a trip-count hint and leaves the branch. 5.31 →
2.84 ms (model).

**L14 Vector int8 epilogue (AIE2P).** Add the bias on an int32 vector, then
`acc.to_vector<int8>(shift)` under `aie::rounding_mode::conv_even`, which
is bit-exact with scalar banker's-rounding SRS. -42% on one block (model).

**L15 Producer writes mmul-A order (AIE2P).** `aie::concat` won't join
vectors under 128 bits, so a strided A operand becomes a scalar byte copy.
When you own both ends, have the producer store `to_vector<int8>(shift)`
(already mmul-A order) and the consumer do one aligned load, or
`aie::shuffle_down(aie::concat(lo, hi), shift)` over two aligned blocks.
-23.6% on one block (model).

**L16 Pure rearrangement → DMA, or wide shuffles.** A body that is only a
deinterleave or transpose belongs in a memtile `dims_to_stream` transform
(element ≥ 512 B; int8 vector loads need a 32 B-aligned start). Hand off to
`aie-dataflow-opt`. `programming_examples/basic/transposes/transposes.py`
shows `--strategy dma` and `--strategy combined`. When it must stay in the
core, widen the body to whole 512-bit `shuffle` stages: llama transpose
3344 → 880, uint32 818 → 150 (ef03238bcc1).

**L17 Raise the pipeliner stage cap on one long chain.** The loop is
pipelined at `ns` 3 and no single step dominates. Add
`["-mllvm", "--aie-pipeliner-max-stagecount=5"]` to that factory's
`compile_flags` only (as `python/iron/kernels/quant.py` does). Check: `ns`
rises; the final II may barely move, so measure anyway. `q4nx_dequant`
2245 → 2115.

**L18 Two 8x8 tiles on one 64-lane accumulator (AIE2P, bf16).** For Y +=
S·V, keep two neighbouring output tiles on one `accfloat` accumulator so
one `vmac.f` advances both; build the S operand once per row block; load
the next pair's y before this pair's store (clamp the last pointer to
itself). Prefill `fv` 2034 → 1265, bit-identical.

**L19 One bfp16 stream per operand (AIE2P).** Output streams share one `sf`
register and inputs two `lf` FIFOs, so extra live streams spill every step.
Write output through one contiguous stream; hop one A and one B stream
between rows with `pop_seek`. `mm_bfp` 4243 → 951; `q4nx_dequant` 2.10x.

**L20 Batch horizontal reductions.** Mac four rows per load of b, pack the
four accumulators so one `interleave_unzip` + `add` tree folds all rows,
and finish group g's tree after group g+1's macs issue; keep the single-row
code for the tail. Rows per group are bounded by the stack (8 overflowed
1 KB in `mv`). bf16 `mv/32x256` 1154 → 788; `mha` `partial_softmax` 8171 →
1696, both bit-identical.

**L21 Port the other architecture's tuned branch.** Enable it on this
target before inventing a new body: `add` 150 → 46 (9929a7a7202), `mul`
142 → 46 (9b193aea86e), `mul_add` 911 → 52 (41c0876239d), `axpy` 178 → 55
(c2570669139); `bn_conv2dk1_i8` 298868 → 1767 (ddf0f4dcbd7), `bn_conv2dk3`
293058 → 1543 (eae144bf389). Then re-check L01, L06 and alignment on the
new target: loops AIE2 unrolls by itself stay rolled on AIE2P (bn 1x1
stack 64 → 1408 B, II 20-22; `UNROLL_FULL` gave 128-192 B, II 8-12), and
AIE2's `& 31` alignment guards are wrong for 512-bit loads (see the traps).

**L22 Promise a minimum trip count.** A runtime-count loop Peano can't
prove runs ≥ n times is left unpipelined or at `ns` 1. Wrap it in
`VERSIONED_LOOP(MinIters, count, body)` (`aie_kernel_utils.h`), which
keeps a plain loop for shorter counts: `mm` identity copy 518 → 144
(2f76486caed). A bare `AIE_LOOP_MIN_ITERATION_COUNT(n)` needs its own
guard and a plain loop for shorter counts: `rope` 298 → 167
(5279bd73571), `dwconv1d_cf` k9 1505 → 501 (954e1ba1f71); on AIE2 `axpy`
269 → 87. It stringifies its argument, so pass a literal or a macro, and
re-check `non_zol_loops`.

**L23 Native `mmul<8,8,8>` for int8 (AIE2P).** `mmul<4,8,8>` on 8-bit data
is an 8x8x8 op with half of A zero. Make the chunk 8 pixels (64 B): bn 1x1
`i8` 56/64/24 1179 → 693 (d90990786a9), `relu` 112/16/64 1838 → 1142
(147bbd32f8d), `skip` 56/72/24 1499 → 890 (558a406e7d5); `bn_conv2dk3`
1115 → 683, then a 64-lane carry 683 → 455 (8bd78ee2771). Unaligned
64-lane loads with a shallow input-channel loop lost (relu 28/40/120 2236
→ 2853), and static rows didn't predict it; measure every case.

**L24 Aligned granule stores (AIE2P).** `store_unaligned_v` reads back the
32 B granule it ends in, so neighbouring blocks form a store → load
recurrence and RecMII holds the loop whatever the body costs. Store whole
aligned granules from `at & ~31` with the previous block's tail shuffled in
front, and one store after the loop: dw `out_split` 7x480 3692 → 1771
(1c32c246a33).

**L25 Keep helper objects in registers.** A LUT or approximation helper
object whose address is taken lives on the stack; the tell is
`kernel_stack_bytes` > 0 on a leaf kernel. Write the lookup out inline:
tanh LUT 1643 → 816, silu 2658 → 1960, swiglu 3202 → 2504, stack 512 → 0
(83ceb77cf0f).

**L26 Split a table gather from its arithmetic.** A gather loop with math
in it runs at the math's II; one pass that only gathers and a second in
place that only computes each pipeline: `sigmoid` 1769 → 925
(88958a67e0a), `silu` 1960 → 962 (fd10d6df7b7), `swiglu` 2504 → 1135
(573a3719e9a).

**L27 Pipeline the epilogue, not only the MAC loop.** Widen the A loads,
load and store C tiles in pairs, and keep one accumulator live at a time
through the store: `cascade_mm` i16 64³ 9532 → 7260 (117707e388f), 7260
→ 5404 (aa99f6b92ce). Forming both tiles' accumulators first measured
6300.

**L28 Start the chain with `mul`; add the loaded C last.** When C is wide
(2048 bits for int32 tiles), the first mac waits on four C loads.
`MMUL(aie::add(C.to_accum(), acc))` after a mul-first K chain folds into
one `vaddmac`: int8 `mm` 64x32x64 733 → 593 (d3b4b66fd7a). It needs more
live accumulators: ungated it lost at K=128 (1741 → 2181), so it is gated
on `colA ≤ 4`. Integer sums only; float order changes bits.

**L29 Compile-time shape.** When the factory knows the shape, pass it as a
`-D` flag the kernel reads on one architecture: dead chunk-count bodies
and the scalar fallback fold away. dw stride 2 14x336 10012 → 6376, `.text`
8112 → 1344-4416 per width (234d9d8fc18); a runtime-trip loop in
`conv2dk1_skip_init` at `ns` 1 1458 → 771 (504d188ee5e). Each shape is its
own symbol, so a core with two shapes links two copies. A constant can
also slow a loop the pipeliner scheduled well at run time (dw8 28x120 1840
→ 1985): keep the runtime value for that loop.

**AIE2 (npu1).** Gate AIE2-only code with `AIE_TUNED_AIE2` or a capability
from `aie_kernels/aie_arch.h`; the other target's `.text` must not change.

- The AIE2 pipeliner needs `__restrict` to overlap a streaming loop; library
  kernels spell it `AIE2_RESTRICT`, which AIE2P ignores (L06).
- An unrolled body above MII 27 gets no overlap. One chain under
  `AIE_LOOP_NO_UNROLL` with `__restrict` pipelined at II1: `add` 342 → 78.
- A LUT loop runs in series when its reads are ordered against its stores:
  rotate it one vector ahead, or use `lut_map_bf16`.
- There is no native bf16 `sliding_mul`; use `mac_elem_16_2`.
- Replace `aie::transpose` in a loop with two-register `shuffle` stages.

## Traps

Code that compiles cleanly and then does nothing, runs slowly or produces
wrong data. None of these is a compiler bug. A Peano crash or miscompile
is: reduce it to a repro, file it against llvm-aie, and name the issue next
to any workaround rather than recording it here. Wrong words clustered by
address class (every other block, one pixel column) or only past a shape
threshold point at addressing before arithmetic: add a device case at the
failing shape (6e57e9dfc6d) and diff the raw words.

- **Libcalls (AIE2 and AIE2P).** The scalar unit has no float multiplier
  or divider: `float * float` (`__mulsf3`), `float / float` (`__divsf3`),
  `(float)int` (`__floatsisf`), float compares (`__ltsf2`) and any `double`
  are runtime calls, as are 64-bit multiply and divide and any integer
  divide or modulo that isn't by a power-of-two constant. A libcall in a
  loop also blocks pipelining. Float add/sub and bf16 ↔ float casts are
  native. The remarks report lists every libcall a kernel makes.
- **No f32 vector multiplier.** `aie::mul`/`mac` on `vector<float,N>` is a
  bf16 emulation (224 B of code where one bf16 mac is 4 B). f32 `aie::max`
  is emulated too (~17 cycles a vector).
- **512-bit accesses round the address down to 64 B (AIE2P).** A `load_v`
  or `store_v` of 64 B at 32 mod 64 silently reads or overwrites the 32
  bytes before it. Buffers ≥ 64 B are 64-aligned, but views into them and
  row offsets are not: weights at 32 B gave 8 of 10 `bn_conv2dk3` cases
  wrong (b86cb6bd6d9), packed weights at 16 mod 64 gave 88/3920
  (a6b8d06ba3a), and a row stride that is an odd multiple of 32 gave
  510/1536 until 8bd78ee2771 split those stores. Guard every pointer the
  path accesses wide, weights too; grep `& 31` near `load_v<N>` whenever
  N × bits > 256.
- **Rounding and saturation are core-global.** The default rounding is
  floor, so a bf16 conversion lands 1 ulp low: `expand` 27% of outputs
  (b20ab7dfcd2); declare the contract's rounding setup, which now runs in
  every Worker (`scale_shift` 38116/65536 wrong → 0, 0d677e9f72d). A mode
  set with `set_saturation`/`set_rounding` outlives the call and breaks
  code that relied on the reset value (an unsigned cascade `lsrs` read
  25-30% wrong under `saturate`); use `swap_*` and restore on exit.
- **`vconv.bf16.fp32` misrounds f32 subnormals** (61.9 ulp), and AIE2P
  `vadd.f` has no guard bit, so a Sterbenz-exact subtraction isn't: do
  exact residuals with `msc` in the accumulator (f376889eab8).
- **Pragmas.** Under Peano `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_FLATTEN`
  and the other Chess hints in `aie_kernels/aie_kernel_utils.h` expand to
  nothing, and `AIE_PREPARE_FOR_POSTPIPELINING` disables pipelining. The
  ones that act are `AIE_LOOP_UNROLL(n)`/`_FULL`/`NO_UNROLL` and the
  trip-count hints (`AIE_LOOP_RANGE` is only a hint). Put each immediately
  before the `for`; `pass_failed` lists any the compiler dropped.
- **Stack overflow is silent.** It corrupts the neighbouring buffer:
  suspect it when errors are small, scattered and row-local. The IRON
  Worker default is 1024 B; aiecc errors with `this core needs N bytes`
  when the declaration is short, and the remarks `kernel_stack_bytes` row
  warns over the contract's `stack_bytes` (it leaves out libcall frames and
  main's). Anything that grows the frame (unroll, accumulators, markers,
  a new size) needs a new declaration in the same change, and a design that
  builds its own Worker needs it too.
- **`to_vector<int32>` is a raw accumulator dump**; `to_vector<int8>(shift)`
  applies the row-major permutation. Indexing one as the other got 61454 of
  65536 values wrong.
- **Software pipelining stops at MII 27** (`SwpMaxMii` in `schedule_notes`);
  above it only the postpipeliner runs. A loop over the cap can still win
  if it does more work per trip.
- **A list-scheduled loop's bundle count moves with edits beside it.**
  bn2's dw block loop (too much register pressure to pipeline) went 84 →
  91 → 97 bundles, HW 778 / 840 / 891, from a guard in the caller, a
  dropped min-trip hint and pointer bumps (eb89013e5d8). Measure every
  shape after each edit.
- **`aie::exp2` (AIE2P) is an interpolant** that overshoots true `exp2` by
  up to 6.15%. Keep `np.exp2` in references with a derived envelope
  (`test/python/npu/test_mha_e2e.py`).
- **Native `vtanh` returns x for |x| ≤ 0.5**, and tanh wants its f32
  argument (a bf16-rounded one raised silu's error 1.35x). The native build
  is latency-bound, the LUT build (`-DACTIVATIONS_TANH_LUT=1`) load-bound:
  gate unrolls on `ACTIVATIONS_NATIVE_TANH` and measure both.

## Changes that measured worse

- Fewer lanes, 8 rows per group, or hoisting operands in bf16 `mv`: slower
  or spilled (register pressure).
- `UNROLL_FULL` on `fused_mm`'s i loop: accumulators spilled, k_step 216 →
  287, and three builds failed their stack check.
- `mmul<8,8,8>` fed by a scalar A gather in a 1x1 int8 conv: 3.34 → 3.58 ms
  (model).
- An `if` or ternary inside a mac loop: 7% slower (model).
- Unrolling a load-, resource- or register-bound body (tanh stayed at II4;
  swiglu ×2 doubled its stack), and `AIE_LOOP_MIN_ITERATION_COUNT` on AIE2P
  loops that then lost their zero-overhead loop. The compiler reports
  these before hardware does; trust it.
- Unrolling far enough that LICM parks invariant loads in the frame:
  `conv2dk1_skip` `UNROLL(8)` 946 and 3264 B against 777 (ebbb20bb013).
- Rolling int8 `mm`'s fully unrolled K: static II 4 and a 64 B stack, HW
  +3..+28% on N=64 and column-major B (51d0a8e000a).
- Turning off the pipeliner's register-pressure tracking: II 12 = ResMII,
  HW 1378 → 1658 from spills (c0e7bb857d0).
- An out-of-line helper for a second path in a hot function: callee-saved
  spills and a longer dispatch, dw 28x120 1606 → 1669, 14x184 2495 → 2551
  (1c32c246a33).
- A new stride inside an existing loop: dw stride 1 reached 778, but the
  min-trip hint also reached stride 2 (+10 cycles) and needed a guard for
  short stride-1 rows. It got its own loop at 840 (eb89013e5d8).

## When to stop

Report NO-CHANGE with the bound when the II equals a resource or latency
bound. `mm_bfp_mixed`'s k loop sits at II6, the `vmac.f` acc → acc latency,
with no fifth accumulator register for another chain. bf16 `mm`'s k loop
needs 34 mv-slot ops per step, so II35 is 97% slot efficiency. Before
calling a loop port-bound, count slot users per iteration in the object; if
the II sits above that count, the bound is something else. On AIE2P one
slot carries `vshuffle`, `vsel`, `vadd`/`vsub`, `veqz`, `vmax_lt`/`vmin_ge`,
`vbor` and `vmov`: `gray2rgba` sits at II 9 against a ResMII of 8
(43a2b1c3b2c), and a load the compiler merges into a `vmov` competes for
it (dw stride 2 2234 → 1378 once the loads stayed, c0e7bb857d0).

## Background

`programming_guide/section-4/section-4c/README.md` (how Peano schedules a
loop, the pragma reference) and `section-4d/README.md` (this workflow for
humans, with a worked `add` example).
