---
name: aie-kernel-opt
description: Make one compiled AIE kernel faster, and prove it. For C++ kernels built by Peano (llvm-aie) for AIE2P (npu2) or AIE2 (npu1) in bf16, float, int8 or int16, from elementwise and normalization kernels to matmul, GEMV, attention and conv, in aie_kernels/ or the user's own .cc. Use when a loop has a high II, won't pipeline, spills, overflows its stack or calls __mulsf3, __divsi3 or another libcall; when the user wants to vectorize or restructure a kernel; or when they want to know whether a kernel change is really faster or bit-identical on the NPU. Drives the in-repo aie.utils.compile.remarks report and its base-arm diff, the trace-marker audit, the test_kernels_e2e gate and the test_kernels_perf back-to-back A/B. Carries the levers that measured faster on hardware and the Peano traps. Not for tile placement or DMA bandwidth (aie-dataflow-opt), or for writing a first kernel (aie-code-creator).
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
edited a symbol no factory calls), and an II37 → 31 change measured slower.

## Setup

```bash
source /opt/xilinx/xrt/setup.sh            # XRT first, or pyxrt is missing
source <venv>/bin/activate
source utils/env_setup.sh install          # from the repository root (docs/Building.md)
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
python -m aie.utils.compile.remarks --target aie2p --only "^$K" --out $W/rows.json --meta $W/meta.json
```

- Remarks prints `[OK] <build>: <symbol> from <source>` per build. Follow
  that symbol to the function it calls and edit only that. A `.cc` often
  holds siblings no factory selects (in-place, `_scalar`, another dtype).
- Remarks builds each factory at its defaults and `.dtypes`; if a case sets
  another `-D` flag, the rows describe the default build.
- Read the contract's `trace=`. Only `Trace.whole_call()` gets a cycles row.
  `Trace.none(reason)` and `Trace.partial(reason)` leave you with wall clock.

### 2. Gate, and prove the gate

```bash
pytest test/python/test_kernel_contracts.py -k "$K" -q           # host only
pytest test/python/npu/test_kernels_e2e.py -m extensive -k "$CASE" --seeds 3
```

- `-m extensive` runs every case, every data case (random plus the
  contract's edge cases) and every seed. `-k` is a substring match (`-k add`
  also runs `mul_add`); check the selection with `--collect-only -q`.
- **Mutation-prove it.** Break the kernel on purpose (drop a term, skip the
  tail, swap two loads), watch the gate fail, revert. Report a mutation
  that survives, with the reason; don't swap in a louder one.
- **Derive every tolerance** from the arithmetic, over the longest case
  (one kernel's error was 0.07 at 4 calls and 0.25 at 256).
- **The reference must not replay the kernel.** Write a single-pass fp32
  reference, and judge against the mathematics (see the exp2 and tanh traps
  below).
- Keep outputs poisoned (`kd.upload(..., poison=True)`) and guarded
  (`kd.design(..., guard=True)`), so an unwritten output or a write past a
  tile's end fails. Gate every entry point alone, and the composition once.

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
change along with the arm.

### 4. Read the static report

```bash
python -m aie.utils.compile.remarks --target aie2p --only "^$K" \
  --out $W/rows.json --meta $W/meta.json --keep $W/objs --baseline-sources $BASE
```

- Per build: `[OK]`, then any `dropped pragma:`, `calls the runtime
  library: <symbols>`, and a `stack:` warning over the contract's budget.
  It exits 3 if a build fails to compile; that is a finding.
  With `--baseline-sources` it ends with `N rows differ` and one
  `name: before -> after` line per row: that is the static A/B.
- Rows: `unpipelined_loops`, `non_zol_loops`, `missing_bank_loads`,
  `pass_failed_warnings`, `pm_bytes`, `libcalls`, `stack_bytes`, and per loop
  `loop/<fn>/<bb>/II` and `not_zol`. They cover only functions the entry
  symbol reaches.
- Meta per loop: `ii`, `ns` (stage count), `pipelined`, `zol`,
  `bundle_count`, `byte_count`, `file`, `line`; per build `pass_failed` and
  `schedule_notes` (`MII`, `SwpMaxMii`). Meta covers this tree only; for
  the base arm's loops run again with `MLIR_AIE_KERNEL_SOURCES=$BASE`.
- Spills: `[sp, #` inside a loop body in the kept object.
  ```bash
  $PEANO_INSTALL_DIR/bin/llvm-objdump -d --no-show-raw-insn <object> | grep -c '\[sp, #'
  ```
- Size: use `pm_bytes` / `pm_bytes_by_function`, not an object's `.text`,
  which counts siblings the link drops.
- A `noinline` body reports its caller's II; count its bundles yourself.

### 5. Diagnose, then change one thing

| Report or object shows | Lever |
|---|---|
| `libcalls` names a helper called in a loop | L02; L11 for `__divsi3` |
| `vector<float>` multiply, min or max in a loop | L03; L04 on AIE2P |
| Array of accumulators or vectors indexed by a loop counter; `[sp` traffic | L01 |
| The loop you care about isn't innermost or single-block; unpipelined parent | L10 |
| Loop body branches on its counter | L13 |
| Address arithmetic in the body; no `__restrict` | L06 |
| Pipelined, many empty bundles; `byte_count` flat from ×1 to ×4 | L07 |
| Stepping 16 lanes on 8/16-bit, or on bf16 on AIE2P | L08 |
| One long mac chain, or broadcasts spilled per block | L09 |
| Pipelined, `ns` 3, one long latency chain with no dominant step | L17 |
| Several `mmul<8,8,8>` accumulators for Y += S·V; high II, big frame | L18 |
| A `reduce_add`/`reduce_max` per row that the row loop never overlaps | L20 |
| bfp16 stream state (`sfl/sfh`, FIFO) spilled around `vst.push`/`vldb.pop` | L19 |
| `(tanh+1)/2`, or separate constant multiplies | L05 |
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
  `[sp, #`, or more `stack_bytes`: stop. Try 2 and 8 when 4 is borderline.
- If the change creates a path no case reaches (an unroll remainder, a
  non-square shape, an odd block count), add a `Case` to
  `test/python/npu/kernel_cases.py` and its name to
  `test/python/npu/perf_series.txt` (`test_perf_series_names.py` checks).
- A change that hides work from LLVM (an opaque pointer bump, `volatile`, a
  LICM blocker) to get a lower II has measured slower (prefill `fv`: static
  II37 → 31, HW 1374-1576 against 1265).
- Write the per-call prediction before measuring:
  `bundles_outside_loop + 5 + (trips - 1) × II`, per loop, trips from the
  loop-count setup in the object. Compare II per unit of work (per row,
  value or tile), not per loop: a raw II that rises can still win. Never
  turn an II ratio into a speedup (prefill `fv`'s said -70%; HW gave -35%).
- Re-run `test_kernel_contracts.py` and the gate.

### 6. Measure both arms back to back

```bash
taskset -c $CPUS pytest test/python/npu/test_kernels_perf.py -m perf -k "[$CASE] or [<unchanged case>]" \
  --no-compile --baseline-sources $BASE --perf-out $W/perf.json --perf-meta $W/meta.json
```

- Each case runs from this tree, then from `$BASE`, with the same inputs.
  The summary prints `cycles base -> cur`, `npu_us min base -> cur` and
  `same` or `N differ` raw output words per case; `--perf-meta` holds the
  same under `baseline.cases.<case>`.
- `-k "[$CASE]"`, brackets included, selects exactly one case. `-k`
  rejects `=`; select a prefix and check with `--collect-only -q`.
- To screen a candidate kept in its own copy, set
  `MLIR_AIE_KERNEL_SOURCES=$W/cand-<name>`; the JIT cache key includes it,
  so no cache wipe is needed. Drop `--no-compile` for the ELF byte rows.

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
  Mutation-prove the diff too: a broken candidate must show words that
  differ. If the change reorders accumulation, say so and give the max |Δ|.
- One commit per kernel and change, with base → after cycles per case, n,
  the arm sources and the gate line. List each rejected variant with the
  number that killed it. NO-CHANGE with a measured bound is a valid result.
- Record provenance with every number: the row's `extra` (commit, Peano,
  `kernels` digest), `git status --short` over the include closure, and
  what `$BASE` came from. When two numbers conflict, compare digests first.
- Several agents on one NPU: wrap every hardware command in `flock` on one
  shared lock file, and build base arms by copying the live tree.
- Keep every `extern "C"` name, signature and buffer layout. Leave
  `*_scalar` variants alone.

## Reading hardware rows

`<case>/cycles` is the **min** of the kernel's own intervals, split from its
initializers' by position (`kd.cycles_per_call`). Its `range` gives
`median X max Y n=N`, `init[i] min M` per initializer, and `truncated`.

| You see | Do |
|---|---|
| No `<case>/cycles` row | The contract is `none`/`partial` (in the library, only `set_rounding`). Use wall clock and say so, or move one marker pair to bracket the whole call and declare `Trace.whole_call()` |
| `expected E trace intervals, got M` | A marker the contract doesn't declare. `pytest test/python/test_kernel_trace_markers.py` names the build |
| `truncated`, or `none of the kernel's` | Compare `n=` with `calls`; flag n < calls/2. Run the marker audit |
| A row moved but its file didn't | Something in its include closure changed: `git diff <rev> -- <closure>`. A 33% "matmul" gain was all `zero.cc` |
| A stream you traced yourself, K kernels per call | Population j is `intervals[j::K]`. Never quote min or median over the mixed stream |

- A cycle row names the design, not the kernel. Before crediting a delta,
  confirm the `[OK]` line names the entry you changed.
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

## Levers that measured faster

Each gives the change, what must move in the static report where it isn't
obvious, and one hardware result (cycles per call unless noted; "model"
means wall clock in a model outside this repository).

**L01 Fully unroll loops that index register arrays.** A short loop
indexes `acc[i]` or calls `insert`/`extract` with its counter; `[sp` around
the accumulators. `AIE_LOOP_UNROLL_FULL` makes each index constant. Check:
stack traffic gone, the enclosing loop gets an II. `conv2dk1_i8` 7623 → 504.

**L02 No libcalls in hot loops.** The `libcalls` row names a helper (see
the traps). Use `aie::max`/`min` for compares, `aie::inv` times a multiply
for divides, `aie::to_float` for int → float, integers for index math;
reduce a per-chunk max element-wise and horizontally once at the end.
Check: helper gone, loop gets an II. `rms_norm` 1597 → 438.

**L03 Avoid the f32 vector multiply (AIE2P).** Skip multiplies by a known 1
or 0. If one operand is exact in bf16 (a tanh result, a chosen constant),
split only the f32 operand into **three** bf16 limbs (residuals via
`aie::msc`; two limbs change bits) and mac each against it. Clamp the
rounded bf16 result against bounds rounded the same way instead of an f32
min/max. Matmul silu epilogue 8098 → 3213, bit-identical.

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
`zero<float,4096>` 519 → 262.

**L07 `AIE_LOOP_UNROLL(4)` on latency-bound bodies**, after the unroll
screen passes. Not on load-, resource- or register-bound bodies. Check: II
per element drops, no new `[sp`. `leaky_relu` 298 → 86, `add` 390 → 150.
`UNROLL(2)` on `fused_mm`'s epilogue chunk loop: gelu 2582 → 1919.

**L08 Full register width.** Step 64 lanes on 8/16-bit data and
`AIE_BF16_LANES` (`aie_kernels/aie_arch.h`) on bf16 arithmetic. Check: trip
count halves at a similar II. `rope` 1932 → 298 (with L06).

**L09 Split dependency chains.** Two independent accumulators added at the
end, or one broadcast feeding several row blocks; stop before spills.
int16 `mv/32x32` 291 → 127.

**L10 Make the hot loop innermost and single-block**, the only loops Peano
pipelines: fully unroll a short K reduction into its parent, fold nested
tile loops into one counter, split a per-tile `if` into straight loops. Opt
in per dtype. int8 `mm` 993 → 737. `UNROLL(2)` on an unpipelined outer loop
overlaps two trips' loads and stores: `fused_mm` 1084.5 → 991.5.

**L11 Unsigned counted trip.** Compute the trip count up front as unsigned;
a signed divide or shift by 2^k is `__divsi3`. `AIE_LOOP_MIN_ITERATION_COUNT`
can help but can cost the zero-overhead loop, so re-check `non_zol_loops`.
`axpy` 317 → 178.

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

**L16 Pure rearrangement → DMA.** A body that is only a deinterleave or
transpose belongs in a memtile `dims_to_stream` transform (element ≥ 512 B;
int8 vector loads need a 32 B-aligned start). Hand off to
`aie-dataflow-opt`. `programming_examples/basic/transposes/transposes.py`
shows `--strategy dma` and `--strategy combined`.

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
itself). More tiles per accumulator crashed Peano. Prefill `fv` 2034 →
1265, bit-identical.

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

**AIE2 (npu1).** Gate AIE2-only code with `AIE_TUNED_AIE2` or a capability
from `aie_kernels/aie_arch.h`; the other target's `.text` must not change.

- The AIE2 pipeliner needs `__restrict` to overlap a streaming loop; library
  kernels spell it `AIE2_RESTRICT` so AIE2P code stays as tuned.
- An unrolled body above MII 27 gets no overlap. One chain under
  `AIE_LOOP_NO_UNROLL` with `__restrict` pipelined at II1: `add` 342 → 78.
- `AIE_LOOP_MIN_ITERATION_COUNT(n)` on a runtime-count loop let it overlap,
  with a plain loop kept for shorter rows: `axpy` 269 → 87.
- A LUT loop runs in series when its reads are ordered against its stores:
  rotate it one vector ahead, or use `lut_map_bf16`.
- There is no native bf16 `sliding_mul`; use `mac_elem_16_2`.
- Replace `aie::transpose` in a loop with two-register `shuffle` stages.

## Traps

Code that compiles cleanly and then does nothing, crashes, or produces
wrong data. Compiler behavior depends on the Peano version
(`utils/peano-requirements.txt`); re-test a crash workaround before relying
on it.

- **Libcalls (AIE2 and AIE2P).** `float * float` (`__mulsf3`), `float /
  float` (`__divsf3`), `(float)int` (`__floatsisf`), float compares
  (`__ltsf2`), any `double`, 64-bit multiply including a constant divide
  (`__muldi3`), and signed divide even by 2^k (`__divsi3`). A libcall in a
  loop also blocks pipelining. Float add/sub and bf16 ↔ float casts are
  native.
- **No f32 vector multiplier.** `aie::mul`/`mac` on `vector<float,N>` is a
  bf16 emulation (224 B of code where one bf16 mac is 4 B).
- **Pragmas.** Under Peano `AIE_PREPARE_FOR_PIPELINING`, `AIE_LOOP_FLATTEN`
  and the other Chess hints in `aie_kernels/aie_kernel_utils.h` expand to
  nothing, and `AIE_PREPARE_FOR_POSTPIPELINING` disables pipelining. The
  ones that act are `AIE_LOOP_UNROLL(n)`/`_FULL`/`NO_UNROLL` and the
  trip-count hints (`AIE_LOOP_RANGE` is only a hint). Put each immediately
  before the `for`; `pass_failed` lists any the compiler dropped.
- **`chess_storage(...)` is a no-op under Peano.** Use `alignas(32)` or
  `alignas(aie::vector_decl_align)`.
- **Subtracting two lane-extracted products** crashes with `unable to
  legalize instruction: G_FSUB`. Keep it in an accumulator with `aie::msc`.
- **Stack overflow is silent.** It corrupts the neighbouring buffer:
  suspect it when errors are small, scattered and row-local. The IRON
  Worker default is 1024 B; aiecc errors with `this core needs N bytes`
  when the declaration is short, and the remarks `stack_bytes` row warns
  over the contract's `stack_bytes` (it leaves out libcall frames). Anything
  that grows the frame (unroll, accumulators, markers) needs a new
  declaration in the same change.
- **Store loops and derived pointer planes can miscompile.** An int16 zero
  loop at unroll 2 stored the loop offset; planes derived as `w + t*stride`
  collapsed to plane 0. Probe with poisoned outputs and one-hot inputs.
- **`to_vector<int32>` is a raw accumulator dump**; `to_vector<int8>(shift)`
  applies the row-major permutation. Indexing one as the other got 61454 of
  65536 values wrong.
- **`pop()` then `pop_seek(odd)` reads the wrong blocks** on a bfp16 input
  stream (AIE2P). Even block strides pass, so gate an odd count
  (`mm_bfp/32x24x48x4/bfp16ebs8/odd-k`).
- **Software pipelining stops at MII 27** (`SwpMaxMii` in `schedule_notes`);
  above it only the postpipeliner runs. A loop over the cap can still win
  if it does more work per trip.
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

## When to stop

Report NO-CHANGE with the bound when the II equals a resource or latency
bound. `mm_bfp_mixed`'s k loop sits at II6, the `vmac.f` acc → acc latency,
with no fifth accumulator register for another chain. bf16 `mm`'s k loop
needs 34 mv-slot ops per step, so II35 is 97% slot efficiency. Before
calling a loop port-bound, count slot users per iteration in the object; if
the II sits above that count, the bound is something else.

## Background

`programming_guide/section-4/section-4c/README.md` (how Peano schedules a
loop, the pragma reference) and `section-4d/README.md` (this workflow for
humans, with a worked `add` example).
