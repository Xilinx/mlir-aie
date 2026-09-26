<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Section 4d - Measure-First Kernel Optimization

* [Section 4 - Performance Measurement & Vector Programming](../../section-4)
    * [Section 4a - Timers](../section-4a)
    * [Section 4b - Trace](../section-4b)
    * [Section 4c - Kernel Vectorization](../section-4c)
    * Section 4d - Measure-First Kernel Optimization

-----

[Section 4c](../section-4c) showed how a vector loop becomes a VLIW schedule and how to read that schedule. This section is about making an existing kernel faster, and knowing that you did. Most wasted effort in kernel tuning does not come from a bad idea. It comes from a number that measured something other than what you thought: a different kernel on the same core, a function no design calls, or a stale build.

Everything here is built on tools that are already in this repository, and every optimization lever comes with the hardware measurement that justified it. The numbers are traced core cycles per call on an npu2 (AIE2P) device with Peano, unless a line says otherwise. The kernels are in [aie_kernels/](../../../aie_kernels/), and each before/after pair was measured when that kernel was optimized. None of the levers was measured with Chess.

## <u>The measurement tools in this repository</u>

The kernel library ([`aie.iron.kernels`](../../kernels_library.md)) comes with correctness and performance checks that you can use on your own kernel edits. [Testing, performance and static checks](../../kernels_library.md#testing-performance-and-static-checks) describes it in full, and the API is in [docs/api/kernels.md](../../../docs/api/kernels.md). In short:

| Tool | What it tells you |
|---|---|
| [`test/python/npu/kernel_cases.py`](../../../test/python/npu/kernel_cases.py) | The table of cases: kernel, tile size, number of calls, edge data, and which cases are smoke tests, nightly tests, or performance cases. |
| [`test/python/npu/test_kernels_e2e.py`](../../../test/python/npu/test_kernels_e2e.py) | Is the kernel still correct? Runs each case on the NPU with poisoned output buffers, checks that nothing was written past each output tile, and judges it against the kernel's numpy reference. |
| [`test/python/npu/test_kernels_perf.py`](../../../test/python/npu/test_kernels_perf.py) | How fast is it? Checks correctness first, then records traced core `cycles` per call, wall-clock `npu_us`, compile time and binary sizes. |
| `python -m aie.utils.compile.remarks` | What did the compiler do? Compiles each kernel exactly as the JIT does and reports every loop's II, stages, zero-overhead-loop status, program memory, dropped pragmas, runtime-library calls and stack depth. No device needed. |
| [Nightly Kernel Checks](https://xilinx.github.io/mlir-aie/kernel-checks/) | The performance history, one chart per case and metric. Its [kernels view](https://xilinx.github.io/mlir-aie/kernel-checks/#view=kernels) lists every kernel, its sources, which NPUs build it and passed it last night, and each case's latest numbers. |

All of them honour `MLIR_AIE_KERNEL_SOURCES` (see [section 4c](../section-4c#before-you-start-make-sure-your-edits-are-compiled)): point it at a checkout and they compile that checkout's `aie_kernels/`. That is also how you build a "before" version to compare against.

## <u>Markers come first</u>

A cycle count needs `event0()` before the work and `event1()` after it in the kernel source ([section 4b](../section-4b)). `test_kernels_perf.py` measures cycles with [`kd.cycles_per_call`](../../../python/iron/algorithms/kernel_design.py), and only for kernels whose `KernelContract` declares `trace=Trace.whole_call()`: **exactly one** `event0()`/`event1()` pair brackets **exactly one** whole call of the entry symbol. A kernel whose markers sit around an inner loop or behind an early return declares `Trace.partial(reason)`, and one without markers `Trace.none(reason)`. A timed case whose kernel declares either fails rather than charting wall clock alone, and the declaration must be right, because a wrong one produces a clean-looking wrong number (see [Cycles you can trust](#cycles-you-can-trust)).

Before you declare `Trace.whole_call()` on a kernel's contract (in `python/iron/kernels/`, in the Python package your tests import):

1. Find the symbol the factory actually compiles, and the file it lives in. The remarks tool prints it for every build: for the `add` factory it is `eltwise_add_bf16_vector` in [add.cc](../../../aie_kernels/eltwise/add.cc).
2. Check that the symbol's body has one marker pair around the whole call. [`test_kernel_trace_markers.py`](../../../test/python/test_kernel_trace_markers.py) does this for every library build without a device: it compiles each build to optimized IR and fails when the markers its entry symbol reaches do not match the contract's `trace`.

An initializer on the same core (such as `zero`) may have markers of its own. The harness calls the initializers and then the kernel in a fixed order, so `cycles_per_call` splits the intervals by position and returns each kernel's population separately (`kd.CallCycles`). A stream with more intervals than the contracts declare means some kernel emits markers it does not declare, and `cycles_per_call` raises instead of returning a number:

```
RuntimeError: expected <N> trace intervals, got <M>; a kernel on the core emits markers its contract's trace does not declare
```

A trace buffer that fills keeps only the first intervals. The split still labels them, and the row's `range` says `truncated`.

Adding markers has a cost of its own: on AIE2P they grew one kernel's stack frame by 64 bytes. After instrumenting, check the remarks tool's `kernel_stack_bytes` row (see [Keeping it correct](#keeping-it-correct)).

## <u>A worked example: `add`</u>

The bf16 [add.cc](../../../aie_kernels/eltwise/add.cc) kernel used to run one load/add/store chain per loop iteration. On AIE2P it now runs four. Here is how that change looks through each tool.

**Hardware first.** The `add` contract declares `Trace.whole_call()`, so the performance check times it. Time the 16-call case with the kernel in your checkout and, in the same run, with a snapshot of the old one. Run from the root of the checkout:

```bash
mkdir /tmp/before
git archive <old-commit> aie_kernels aie_runtime_lib | tar -x -C /tmp/before
MLIR_AIE_KERNEL_SOURCES=$PWD pytest test/python/npu/test_kernels_perf.py -m perf \
    -k "[add/1024x16/bfloat16]" --perf-out after.json \
    --baseline-sources /tmp/before --perf-meta meta.json
```

`--baseline-sources` measures each case a second time with its kernels from that tree, back to back and on the same inputs. The terminal summary and `meta.json` give both arms' `cycles` and minimum `npu_us`, and how many raw output words differ. The rows in `after.json` are the checkout's. Set `MLIR_AIE_KERNEL_SOURCES` for the checkout too. Without it, the tools compile the installed copy of `aie_kernels/`, so the "after" arm quietly measures whatever kernel was last installed. The brackets in `-k` match the whole case ID; without them, `mul_add/1024x16/bfloat16` would match too. `--pmode <mode>` makes the run refuse to start unless the device is in that power mode (the nightly uses `performance`); it checks the mode and does not set it.

Each row in the JSON is `<case>/<metric>`:

| Row | Before (one chain) | After (four chains) |
|---|---|---|
| `add/1024x16/bfloat16/cycles` | 390 | 150 |
| `add/1024x16/bfloat16/cycles_per_kop` | 380.859 | 146.484 |
| `add/1024x16/bfloat16/npu_us` | 85.16 | 90.53 |

`cycles` is the minimum over the 16 traced calls, with the median, maximum and count in the row's `range`. The table's cycles are medians, recorded before the performance check switched to the minimum. `npu_us` is the median device time of a whole 16-call dispatch. The kernel is 2.6x faster, yet `npu_us` got *slower*. The next section explains why that is expected, and why you should judge the kernel by `cycles`.

**Then the compiler's view.** Run the remarks tool on each tree:

```bash
MLIR_AIE_KERNEL_SOURCES=$PWD python -m aie.utils.compile.remarks --target aie2p --only '^add$' \
    --out after-static.json --meta after-meta.json
MLIR_AIE_KERNEL_SOURCES=/tmp/before python -m aie.utils.compile.remarks --target aie2p --only '^add$' \
    --out before-static.json --meta before-meta.json
```

[add.cc](../../../aie_kernels/eltwise/add.cc) holds three functions, one row per loop: `eltwise_add_bf16_scalar`, `eltwise_add_bf16_vector` and `eltwise_add_bf16_vector_size`. The `add` factory calls `eltwise_add_bf16_vector`, so that is the row to read. Its `value` and `range`, before and after:

```
"name": "add/loop/eltwise_add_bf16_vector/for.body.i/II"
before:  "value": 12, "range": "NS=1 pro=3 epi=6 zol=True via=postpipeliner at accum.hpp:936"
after:   "value": 18, "range": "NS=1 pro=3 epi=6 zol=True via=postpipeliner at accum.hpp:936"
```

The value is the II. The range gives the number of stages (`NS`), the prologue and epilogue bundles, whether it is a zero-overhead loop, which scheduler placed it, and the source line the loop was attributed to (after inlining, often an AIE API header). `NS=1` means iterations do not overlap. In both versions the meta file also carries a note for this loop, `eltwise_add_bf16_vector@L<line>: Unable to find schedule`: the modulo scheduler gave up, and the post-pass scheduler packed the body on its own.

Now check the arithmetic: the old loop does one vector (32 elements) per iteration, so 1024 / 32 = 32 iterations × 12 = 384, plus about 6 cycles to enter and leave, is 390. The new loop does four vectors per iteration: 8 × 18 = 144, plus 6, is 150. Both match the hardware to the cycle. When II × trips does not come close to the traced count, something other than this loop is running, and it is worth finding out what before changing anything.

**Then the bundles.** Disassemble the kernel object (see [section 4c](../section-4c#reading-the-schedule-disassembly)). The old loop body, with the empty slots of the long bundles trimmed:

```
<.LBB1_1>:
  vlda.conv.fp32.bf16  cml3, [p0], #0x40
  vlda.conv.fp32.bf16  cml4, [p1], #0x40
  nopa ; nopb ; nops ; nopxm ; nopv
  nopa ; nopb ; nops ; nopxm ; nopv
  nopa ; nopb ; nops ; nopxm ; nopv
  vadd.f  dm0, dm3, dm4, r0
  nopx
  nop
  nop
  nop
  nop
<.L_LEnd1>:
  nopa ; nopb ; vst.conv.bf16.fp32  cml0, [p2], #0x40 ; nopxm ; nopv
```

Twelve bundles do four useful things. Both inputs are loaded with `vlda.conv.fp32.bf16`, a load that converts bf16 to float on the way in and exists only on load unit A, so the two loads take two bundles instead of sharing one. The `vadd.f` then waits out the load latency, and the store waits for the add. Nothing else is available to fill the gaps: the loop is **latency-bound**, not limited by any unit.

The shipped loop gives the scheduler four independent chains. Its 18 bundles start with eight `vlda.conv` loads back to back, and the adds and stores of earlier chains fill the bundles that were empty. 4.5 bundles per vector instead of 12. The loads on unit A now set a floor of 2 bundles per vector, so there is still some room, but the easy win is taken.

AIE2 took the opposite route. There the four-chain body needs more bundles than the pipeliner's limit (`SwpMaxMii` 27), so it ran unoverlapped. The AIE2 branch of `add.cc` keeps one chain under `AIE_LOOP_NO_UNROLL` with `__restrict` pointers, and the pipeliner overlaps its iterations to one vector per cycle: 342 → 78 cycles on an npu1.

This is the pattern for the rest of the section: hardware says whether it got faster, the remarks say what the compiler did, and the bundles say why.

## <u>Two clocks</u>

[Section 4a](../section-4a) measures wall-clock time around a whole dispatch; [section 4b](../section-4b) measures core cycles between the markers. They answer different questions. Use traced cycles per call to judge a kernel change. Use wall clock only to say whether the *application* got faster.

A dispatch has fixed costs that have nothing to do with the kernel. On npu2 a 16-call dispatch has a floor of roughly 60-90 us of device time. Before you trust a wall-clock delta, estimate how much of it is the core working:

```
core_fraction = cycles_per_call × calls / core_clock / device_time
```

The core clock measured 1.76 GHz on npu2. (To calibrate it on your part, trace a long compute-bound kernel whose every call is instrumented: its `core_fraction` should come out at 1.0.) For `add`, 150 cycles × 16 calls / 1760 cycles per us = 1.4 us of core time in 90 us of device time, a `core_fraction` of 0.015. No kernel change can show up in that wall clock. Two more examples:
* `sigmoid` became 4.2x faster on the core and 32.5% faster in wall clock; its `core_fraction` was 0.046.
* `conv2dk1_i8` became 15.2x faster on the core and 29.5% faster in wall clock.

Wall clock is also noisy. Between builds that were byte-identical, the minimum device time moved by up to about 19 us plus 3%. The average moved much more, and once showed a 7.5% "win" between identical builds. If you must compare wall clock, pin the host process (`taskset`), compare the **minimum** over many iterations, and first check that the two builds actually differ.

## <u>Cycles you can trust</u>

`test_kernels_perf.py`'s `cycles_per_call` labels each interval with the kernel that emitted it and refuses a stream that holds more intervals than the contracts declare. `get_trace_summary.py`, which the programming examples' `make trace` runs, does not: it pairs each `event0` with the next `event1` and prints first/min/avg/max, whatever those intervals are. Check before you quote it.

**Count intervals first.** Compare the number of intervals with the number of kernel calls the design makes:

| Intervals | Meaning |
|---|---|
| = calls | One marker pair per call. The numbers describe the kernel. |
| 2 × calls (or calls + blocks) | A second instrumented kernel runs on the same core. Split the populations. |
| > calls | The markers bracket an inner region, not the whole call. |
| < calls | The trace buffer filled. Increase the trace size or trace fewer calls. |
| 0 | The kernel that runs has no markers. |

**Two kernels on one core.** A design that accumulates into an output buffer usually runs a `zero` kernel on the same core before each block, and [zero.cc](../../../aie_kernels/zero/zero.cc) has its own markers. You met this in [section 4c](../section-4c): the single-core matrix multiply reports min 70, which is `zero`, while the matrix-multiply call is 537. A median over the pooled list is no better. In a matrix-vector design, the pooled median moved from 142 to 64 because the mix of the two kernels changed, while the kernel itself went from 291 to 127.

**The clean row that is the wrong kernel.** The dangerous case is a kernel *without* markers next to an initializer *with* them. The summary then shows one tidy population, of the wrong kernel. During one optimization effort, three matmul and attention rows reported a 33% win that was really a `zero` change. Two things gave it away: six unrelated designs showed exactly 775 cycles, and the count scaled with the bytes being cleared (16384 B → 775 cycles, 8192 B → 391). To rule this out, resolve the factory to its source file and symbol and count the markers in **that** file (`grep -c 'event0()'`). A population whose cycles scale with the size of a buffer being cleared is the initializer. Known `zero` costs on npu2: 262 cycles for 4096 floats, 134 for 4096 bf16.

**Optimize the symbol that runs.** Kernel files often hold several variants of a function. In one case a variant's loop went from II 33 to II 18, and hardware reported 594 → 594 cycles: no factory calls that variant. The same change applied to the variant the `gelu` factory selects gave 594 → 318.

**Split the populations.** [`get_cycles_summary`](../../../python/utils/trace/utils.py) returns every interval per core, so a histogram takes a few lines:

```python
import collections
from aie.utils.trace.utils import get_cycles_summary

for core, *intervals in get_cycles_summary("trace_mm.json"):
    print(core, len(intervals), sorted(collections.Counter(intervals).items()))
```

For the single-core matrix multiply this shows two clusters, 70-91 cycles (32 intervals, `zero`) and 537-553 (521 intervals, the matmul). Report each population on its own.

**Kernels without markers.** When a kernel cannot be traced, difference wall clock over repeat counts instead: build the design with N and with M calls and divide the time difference by N - M. The fixed dispatch cost cancels. A bfp16 shuffle kernel measured this way went from 27.5 to 2.6 us per call, while the plain wall clock of one dispatch showed only a 56% improvement.

**Find where the time goes by removing it.** To see what one operation costs inside a kernel, replace it with something cheap and wrong, and re-time. In `mv_bf16`, replacing the horizontal reduction with a stand-in took a call from 1154 to 634 cycles, so the reduction costs 520 cycles, 45% of the kernel. Halving the K dimension saved only 144 cycles, so about 866 cycles do not depend on K at all. The target was the reduction, not the multiply-accumulate loop; [lever 3](#3-fill-the-empty-bundles) shows the change that followed.

## <u>Reading the static report</u>

Hardware runs are slow and noisy. The compiler's own report is fast and exact, so use it to screen ideas and use hardware to decide. A better II is only a candidate: the `gelu` variant above improved from II 33 to II 18 and did nothing on hardware.

The remarks tool (see [Static checks](../../kernels_library.md#static-checks)) writes two files. `--out` has one row per series, as shown for `add` above. `--meta` has one record per loop:

| Field | What to look for |
|---|---|
| `ii`, `ns` | The initiation interval and the number of stages. `ns` = 1 means iterations do not overlap. |
| `bundle_count` | The loop body's size. Compare it with the useful work per iteration: many more bundles than operations means a latency-bound loop. |
| `zol` | `false` means the loop pays for a branch every iteration. |
| `pipelined`, `missed_reason` | A loop the pipeliner declined. The hot loop should be innermost, with no `if` in its body. |
| `prologue_bundles`, `epilogue_bundles` | The fixed cost of entering the loop. It matters when the trip count is small or the loop is entered many times. |

Per kernel, `pm_bytes` is program memory, `pass_failed` lists `#pragma clang loop` or `AIE_*` hints the compiler could not apply, and `schedule_notes` holds the scheduler's messages. "Unable to find schedule" means the modulo scheduler gave up; the loop may still have been packed by the post-pass scheduler, so read `ii` and `ns` too. The tool builds each factory at its defaults and at each entry of its `.dtypes` table, not at every case's parameters.

**Library calls and stack.** Scalar code that looks harmless can compile to a function call. The tool also reads each kernel object. Its `libcalls` row counts the runtime-library routines that the entry symbol reaches, and the build prints their names (`calls the runtime library: __divsf3 __mulsf3`). A healthy kernel has none. [Peano and AIE2P traps](#peano-and-aie2p-traps) lists the calls to look for. For an object outside the library, `$PEANO_INSTALL_DIR/bin/llvm-nm -u <kernel>.o` lists the same symbols, next to any other kernels it calls. The `kernel_stack_bytes` row is the deepest chain of stack frames from the entry symbol, not counting the library routines' own frames or the core's `main`, so the core needs more than it says. When it is over the contract's `stack_bytes` (or the target's default), the tool prints a warning.

**The object.** When a number moves unexpectedly, look before you explain. In the loop body, count the `vlda`/`vldb`/`vmac`/`vst` operations and any `[sp, #...]` accesses (spills to the stack). Size code per function, not per object: `llvm-size -A` and the `.text.<entry>` section. A `static` helper that is inlined into its `extern "C"` entry point can also be emitted on its own; it counts in the object, and the linker then drops it.

## <u>Levers that measured faster</u>

Each lever below targets one kind of bound, and each comes with its hardware measurements. "Combined" means the change applied more than one lever at once, so the number is not the lever's alone. The first two groups gave the largest and most repeatable wins; start there.

### 1. Take library calls and emulated float math out of the loop

**Why it works.** On AIE2P with Peano, several scalar operations are function calls (listed in [traps](#peano-and-aie2p-traps)). A call in a loop body costs tens of cycles per element and stops the loop from being software-pipelined. Separately, AIE2P has no `float` vector multiplier: `aie::mul` on `aie::vector<float, N>` is emulated with several bf16 multiplies, roughly 41-60 bundles per vector.

* **Replace the calls with vector or integer operations.** Use `aie::max`/`aie::min` for comparisons, `aie::inv` and a multiply for a divide, `aie::to_float` to convert integers, and integer or `constexpr` tables for index math. Avoid `double`. `rms_norm` (1024): **1597 → 438** cycles. Combined with skipping multiplies by an identity scale, `layer_norm` 2421 → 581. Combined with reducing the running maximum once at the end, `softmax` 1428 → 993.
* **Split only the operand that needs it.** Where one operand is exact in bf16, split only the `float` operand into bf16 pieces and multiply-accumulate against them, and skip multiplies by known 1 or 0. `mm_activation_epilogue` (silu): **8098 → 3213**. Use three bf16 pieces: two carry only 16 of the 24 mantissa bits and change output bits.
* **Run emulated float math 32 lanes wide.** The emulation works on 32 lanes, so a 16-lane `aie::mul` pays for 32 lanes and uses 16. Also replace vector `to_fixed`/`to_float` inside the chain (each goes through the shift-round path and writes a mode register) with the magic-number floor `x + 1.5×2^23`. `bf16_exp` (1024): **32482 → 15425**; `exp2f_vec`: 31558 → 14125 (combined). The results are bit-identical except that NaN inputs now give a quiet NaN.
* **Clamp in bf16, not in float.** AIE2P has no native `float` vector min/max either, and the emulated clamp held even the identity epilogue of [the fused matrix multiply](../../../aie_kernels/fused/mm_fused_mmul.h) at II 30 per 16 lanes. Rounding is monotone and leaves representable values alone, so clamping the bf16 result against bounds rounded the same way gives identical raw output. Identity/clamp epilogue per chunk: **126 → 19** cycles (combined with cursors, 32 lanes, and the unroll by 2 in lever 3).

### 2. Keep arrays and addresses in registers

**Why it works.** Vector and accumulator registers cannot be indexed by a runtime value. If a loop indexes an array of accumulators with its counter, or calls `insert`/`extract`/`set` with it, the array has to live in memory. Likewise, Peano does not always turn `base[i * stride]` into a pointer that is bumped by the load itself, so it recomputes the address on the scalar unit every iteration.

* **Fully unroll loops that index register arrays.** `AIE_LOOP_UNROLL_FULL` makes every index a constant, and the array stays in registers. `conv2dk1_i8`: **7623 → 504**. Combined with the next lever, a `transpose` design went from 177.2 to 72.0 us wall clock. `AIE_LOOP_RANGE` is not a substitute: it only states the trip count. On an int8 convolution network, a small loop whose body switched on the loop variable went from 5.31 to 2.84 ms when `AIE_LOOP_RANGE(3,3)` was replaced with `AIE_LOOP_UNROLL_FULL`.
* **Walk `__restrict` cursors instead of indexing.** Advance a `T *__restrict` pointer or an `aie::begin_restrict_vector` iterator, and mark input and output pointers `__restrict` so the compiler may overlap loads with earlier stores. `zero`: **519 → 262** (4096 floats), 263 → 134 (4096 bf16), 150 → 78 (4608 bytes). For `zero`, 147 → about 75 and 391 → about 130 were predicted from the II before the run. In [mm_aie2p.h](../../../aie_kernels/linalg/mm_aie2p.h), stepping the B and C tile addresses with the loop took a third to half of the stack accesses out of the loop: int16 with column-major C 2873 → 2753 (-4.2%), bf16 5761 → 5713. Combined with a cheaper conversion, `expand` (576) went 839 → 162. Combined with unrolling, `convert_copy` (1024) went 652 → 88.
* **Compute the trip count up front, as unsigned.** A signed divide by 2<sup>k</sup> does not become a shift. `axpy` (1024): **317 → 178**. `AIE_LOOP_MIN_ITERATION_COUNT(1)` was part of that change, but in two other kernels the same hint cost the zero-overhead loop, so check `zol` in the remarks after adding it.

### 3. Fill the empty bundles

**Why it works.** A loop whose body is mostly `nop` at a fixed II is waiting on latency, as `add` was in the [worked example](#a-worked-example-add). Independent work fills those bundles for free.

* **Unroll a latency-bound loop by 4.** `leaky_relu`: **298 → 86**; `mul`: 454 → 142; `add`: 390 → 150; `gelu`: 594 → 318 (all AIE2P). On AIE2 an unrolled body can exceed the pipeliner's limit instead; see [the AIE2 route for `add`](#a-worked-example-add). Screen ×1, ×2 and ×4 with the remarks tool first, then confirm on hardware. The right factor depends on the body. Do not unroll a loop that is already bound by loads, stores, or the multiplier: there are no empty slots to fill.
  The same holds for a loop that is already pipelined but short: `AIE_LOOP_UNROLL(2)` on the fused matrix multiply's epilogue chunk loop, the only change between two measured versions, took the gelu chunk from 172 to 89 cycles and the whole gelu call from 2582 to 1919.
* **Unroll by 2 a loop that does not pipeline.** When a loop does not software-pipeline, each trip pays for its inner loop's entry and exit in sequence. Two trips per body let those overlap. The fused matrix multiply's k step: **216 → 192-194** cycles, with identical output. Fully unrolling the inner loop instead made it slower (216 → 287) and overflowed the 4096-byte stack in four configurations.
* **Split long dependency chains.** A single accumulator chain runs at the latency of the accumulate instruction. Two independent accumulators, or several output rows fed from one broadcast operand, let the chains overlap. Stop before the extra registers spill. `mv` (int16, 32x32): **291 → 127**. `dwconv1d`: 2810 → 1505 (two chains plus an unrolled tap setup).
* **Fold constants so one multiply-accumulate does the work.** [sigmoid.cc](../../../aie_kernels/activation/sigmoid.cc) computes `(tanh(x/2) + 1) / 2` as `0.5 × t + 0.5`, one multiply-accumulate into an accumulator preloaded with 0.5. Scaling by a power of two commutes with rounding, so the result is bit-identical. `sigmoid` (1024): **498 → 118**. Combined with wider vectors, `swiglu` 3335 → 856. Combined with an unroll, `silu` 1255 → 211.
* **Give a latency-chain-bound loop more pipeline stages, when the remarks show the stages are the cap.** Peano's pipeliner allows 3 stages by default. If one iteration is a single long dependency chain, and the remarks show the hot loop at `NS=3` with little else in its bundles, the stage limit, not a unit, sets the II. A factory can pass Peano flags to one kernel through `ExternalFunction`'s `compile_flags`. [quant.py](../../../python/iron/kernels/quant.py) builds `q4nx_dequant` with `-mllvm --aie-pipeliner-max-stagecount=5`; its iteration is a chain of about 48 cycles (load, shuffle, unpack, convert, multiply-accumulate, convert, two shuffles, store). `q4nx_dequant`: **2245 → 2115** (5120x4, -5.8%), 712 → 673 (1536x2), 205 → 201 (512x2). The output is byte-identical, the stack is unchanged, and the code is 128-224 bytes larger. Passing `=3`, the default, compiles to identical code, which is a quick check that the flag reaches the kernel. Four stages gave most of the gain. Before register allocation the pipeliner's II dropped from 16 to 11-12, but the final loop only went from II 17 to II 16, which matches the hardware. So screen stage counts with the remarks and judge them by the final II, not the pipeliner's first estimate. Pass the flag to that kernel only.

### 4. Shape the loop nest for the pipeliner

**Why it works.** Peano software-pipelines only innermost loops whose body is a single basic block. Every loop above the innermost one pays its setup and drain in sequence, on every trip.

* **Fully unroll a short inner reduction into its parent.** `mm` (int8, 64x32x64): **993 → 737**, from unrolling the short K loop. The bf16 and int16 builds did not move.
* **Fold loop nests into one counter.** In `mm_bfp_mixed`, the two outer loops over a 2x2 grid of output tiles became one tile loop: 1349 → 1313 (64x64x64), 589 → 545 (64x32x32), with bit-identical output. Here the static estimate (the loop's bundles + 5 + (trips - 1) × II) matched the hardware within 1 cycle in both versions.
* **Move `if`s out of the multiply-accumulate loop** by splitting it into straight loops. On an int8 convolution network, this took end-to-end latency from 10.48 to 9.77 ms.

* **Batch horizontal reductions.** A per-row `reduce_add` is a chain of shuffles and adds that nothing overlaps. The matrix-vector kernel [mv_bf16](../../../aie_kernels/linalg/mv_bf16.cc) now multiplies four rows against each chunk of the vector, packs their four accumulators so that one `interleave_unzip`-and-add tree reduces all four rows at once, and finishes one group's tree while the next group's multiplies run. `mv_bf16` (32x256): **1154 → 788** (-31.7%, combined with per-row cursors); 4x64: 156 → 120, with identical output on 10 shapes. The [ablation](#cycles-you-can-trust) that found the reduction also predicted this: the reduction was 45% of the call. One shape barely moved (4x2048, 420 → 408): its rows are 4 KB apart and land in the same memory banks.

### 5. Use the full register width

Step 64 lanes for 8- and 16-bit data and 32 lanes for bf16 arithmetic, not 16. `rope` (1024): **1932 → 298** (combined with cursors).

### 6. Cut memory traffic, and tell the compiler what does not alias

* **Copy with wide words, not bytes.** Peano does not software-pipeline a byte load/store loop. Copy with `uint64_t`/`uint32_t`, or merge byte blocks into 32-byte vector stores. A bfp16 shuffle kernel: **27.5 → 2.6 us** per call, and the unshuffle 28.7 → 3.2 (by repeat-count slope). On an int8 convolution network, an explicit `uint64_t` copy loop instead of a byte loop gave +14% frames per second.
* **Load the next values before storing the current ones.** If a loop stores to a buffer it also loads from, the pipeliner cannot prove the next load is independent of this store, and it serializes them. The flash-attention prefill kernel loads the next pair of output tiles before storing the current pair, and pairs two 8x8 output tiles on one 64-lane accumulator so one native `vmac.f` advances both. Head dimension 512: **2034 → 1265** (-35%); 256: 3199 → 2707 (combined).
* **Use no more block streams than there are registers for their state.** Each `aie::block_vector_input_buffer_stream` or `block_vector_output_buffer_stream` keeps its FIFO state in a register, and AIE2P has few of them: the input streams share two `lf` registers, and the output streams share one `sf` register. Every stream beyond that spills its state to the stack and reloads it on every step. Share one stream per operand and move it with `pop_seek` rather than opening one per row or tile.
    * The bfp16 [mm_bfp.cc](../../../aie_kernels/linalg/mm_bfp.cc) opened four A/B input streams and two C streams for each 2x2 group of output tiles, which meant 73 stack accesses per call. Now one A stream and one B stream hop between the group's two rows, and C gets one output stream for the whole call. `mm_bfp` (64x64x64): **4243-4529 → 951**, and 32x24x48: 850-894 → 269, with byte-identical output (33 of 33 raw hardware outputs over 11 shapes and 3 data patterns).
    * `q4nx_dequant` wrote each group through two output streams that shared the one `sf` register. It now writes one stream, in the order the output is laid out: **4719 → 2245** (2.1x), combined with processing one 8x8 tile per iteration and flattening its loops.

On an int8 convolution network, three more changes measured faster:
* doing the bias add on an int32 vector and then `acc.to_vector<int8>(shift)` with `rounding_mode::conv_even`, which is bit-exact with the scalar rounding: 20-25% per kernel;
* having the producing kernel write the matrix-multiply A operand in the order the consumer's `mmul` loads it: -23.6% on one block;
* moving a pure strided copy (a stride-2 deinterleave) out of the kernel into the memory tile's DMA (`dims_to_stream`): +12%. This helps only when each contiguous element is at least 512 bytes, and int8 vector loads still need a 32-byte-aligned start.

## <u>Peano and AIE2P traps</u>

Some things compile without a warning and then run silently slower or give wrong results.

**Library calls** (AIE2 and AIE2P, Peano): scalar `float` multiply (`__mulsf3`) and divide (`__divsf3`), `float` from a 32-bit integer (`__floatsisf`, `__floatunsisf`), scalar `float` comparisons (`__ltsf2`, `__gtsf2`), anything using `double`, 64-bit integer multiply (`__muldi3`), and 32-bit integer divide (`__divsi3`). A group size that is not a power of two can turn `/ GROUP` into `__muldi3`. Native: `float` add and subtract, `aie::inv`, `aie::invsqrt`, `aie::to_float`, `aie::max`/`aie::min`, and bf16↔float conversion. `aie::to_float` is not a call, but inside a hot vector chain it is not free either (lever 1).

**Pragmas that do nothing, or the opposite.** Under Peano, `AIE_PREPARE_FOR_PIPELINING` expands to nothing, and so do the other Chess-only controls ([pragma reference](../section-4c#loop-pragma-reference)). Code built with and without it is byte-identical. `AIE_PREPARE_FOR_POSTPIPELINING` expands to `pipeline(disable)`: it **turns pipelining off**. `AIE_LOOP_RANGE` states a trip count and unrolls nothing. `chess_storage(...)` silently drops its alignment and bank placement: the same table was 32-byte aligned under Chess and 4-byte aligned under Peano. Use `alignas`.

**Silent wrong results.**
* A stack overflow does not trap on AIE2P; it overwrites the neighbouring buffer. aiecc measures the stack each core needs and fails the build when the declared size is too small. The IRON `Worker` default is 1024 bytes; a kernel contract states a larger need in `stack_bytes`. The remarks `kernel_stack_bytes` row gives the kernel's deepest path before you build; the core adds `main`'s frame.
* On hardware, a block stream's `pop_seek` that directly follows a plain `pop()` landed on the wrong block when the seek stride was odd; even strides were correct. [mm_bfp.cc](../../../aie_kernels/linalg/mm_bfp.cc) therefore seeks after every pop when the number of k blocks is odd, and an odd-K case in [kernel_cases.py](../../../test/python/npu/kernel_cases.py) covers that path.
* An unaligned `load_v<int8, 32>` silently returns wrong bytes: use two aligned loads and `shuffle_down`. Vector stores of 64 lanes need a 64-byte-aligned buffer (32 at the least), and lookup tables need `alignas(32)`.
* `mmul::to_vector<int32>` dumps the accumulator in a different lane order from `to_vector<int8>(shift)`.

**The modulo scheduler has a limit.** Peano's modulo scheduler gives up on loops whose minimum II is above 27; those loops get only the post-pass scheduler. The remarks note says so (for example, `Minimal Initiation Interval too large: 40 > 27` on the scalar `add` loop). Splitting a large body is the way back into the pipeliner.

**Gathers.** There is no fast per-lane gather: `load_4x*` spreads its address bits across lanes, and `parallel_lookup<int8>` uses 16 distinct inputs per 32-lane fetch.

## <u>When to stop</u>

Stop when the loop's II equals its resource floor: the busiest unit has something in every bundle. A loop at its floor only gets faster by doing less work per element, or with a different data layout. Three floors from the kernels in this repository:
* The 16-bit `scale` loop in [section 4c](../section-4c) is II 1 on AIE2P. Nothing is faster than one bundle per iteration.
* The bf16 `mm` K loop is II 35 against a floor of 34 set by the move unit: AIE2P emulates the bf16 matrix multiply, and each 2x2 step needs 34 shuffles and broadcasts. That caps bf16 at about 30 MACs per cycle. The loop itself reaches 29.3 (1024 MACs every 35 cycles); a whole call reaches 22.9 (131072 MACs in 5713 cycles), the rest being the loading, storing and draining of C between K loops. Only a different emulation, which changes the numerics, gets past this floor.
* The mixed bf16 × bfp16 matrix multiply's K loop is II 6, the latency from one `vmac.f` to the next on the same accumulator. More independent chains would hide it, but AIE2P has only five accumulator registers, and removing work from the loop left it at II 6 in the compiler's report. Only the loop-nest overhead around it moved on hardware (lever 4).

The time outside the inner loop matters as much. In the fused matrix multiply, a k step is 193 cycles around a 140-cycle multiply-accumulate loop (II 35 × 4); the rest is C making a round trip through memory in `float` on every step.

When the kernel is near a floor and several reasonable variants all measure worse, stop and record it. For `mv_bf16`, six such variants measured worse (next section) before the reduction batching in lever 3 found the 31.7%. That came from the ablation, which showed where the cycles were going.

A well-measured "no change" is a result. Record it, with the number that ruled it out.

## <u>Changes that measured worse</u>

| Change | Kernel | Result |
|---|---|---|
| Optimizing a variant no factory calls | `gelu` | II 33 → 18, hardware 594 → 594 |
| Narrower vectors to shorten the reduction tree | `mv_bf16` | 1154 → 1321 (32 lanes) or 1641 (16 lanes) |
| `AIE_LOOP_UNROLL_FULL` on a loop whose trip count varies by shape | `mv_bf16` | 1154 → 1062 at one shape, 1010 → 1122 at another, does not build at a third |
| Eight rows per group, or hoisting an operand | `mv_bf16` | stack frame 0x100 → 0xE00, or three times the stores |
| Exact `float` log2(e) scale | flash-attention prefill | error unchanged, cost within noise, about 110 more instructions |
| Splitting a `float` operand into two bf16 pieces | fused epilogue | 12 and 11 of 32,768 output words changed |
| An 8x8x8 `mmul` fed by a scalar gather of A | int8 1x1 convolution | 3.34 → 3.58 ms |
| An `if` or ternary inside a multiply-accumulate loop | int8 convolution | 7% slower |
| `AIE_LOOP_UNROLL_FULL` on the mmul row loop | fused matrix multiply | k step 216 → 287, and four configurations overflow the 4096-byte stack |
| An opaque pointer bump to keep operand loads in the loop | flash-attention prefill | compiler's II 37 → 31, but hardware 1265 → 1374-1576 |
| Averaging host wall clock | any | a phantom 7.5% between identical builds |

Many more variants were rejected on the compiler's report alone, because the II grew or registers spilled. Those are useful screens, not measurements, and are not listed here.

## <u>Keeping it correct</u>

A faster kernel that is sometimes wrong is not faster. `test_kernels_e2e.py` is the gate, and the performance check refuses to time a kernel that fails it. These habits catch what an ordinary test misses:

* **Test the tails.** Every unroll, widening or blocking adds a remainder path. Add a case to [kernel_cases.py](../../../test/python/npu/kernel_cases.py) whose size reaches it; the activation kernels have 160-element cases for this. Square shapes can hide index-wrap bugs: in `mm_bfp_mixed`, a wrap mutation failed 5 of 6 runs of a non-square 64x32x32 case, and every square case passed.
* **"Bit-exact" means raw words.** A tolerance check can pass a broken kernel. In the fused epilogue, a test at 4% relative tolerance passed with a dropped term, with a 6% shift in an activation's argument, and with the clamp bounds rounded the wrong way, which changed 45,159 output words. To claim "bit-identical", run both builds on hardware and diff the raw output words.
* **Prove the gate.** Break the kernel on purpose (drop a term, skip the tail) and confirm the test fails. A test that survives the mutation is not testing that path.
* **Poison the outputs.** An unwritten element must not pass as a zero. The kernel test and performance check upload poisoned outputs for you, and the kernel test also fails a kernel that writes past the end of its output tile.
* **Derive tolerances from the error model**, and measure them over the longest case. A generous default hid an overrun in `swiglu` that grew from 0.07 at 4 calls to 0.25 at 256.
* **Keep references independent.** A reference must not replay the kernel's own recurrence. Judge an approximation against the true function: AIE2P's `aie::exp2<bfloat16>` is a piecewise-linear `2^floor(u) × (1 + frac(u))` that overshoots by up to 6.15%, so the attention tests compare with `np.exp2` and a derived error envelope. The native `vtanh` returns x itself for |x| ≤ 0.5, so test both tanh configurations.
* **Watch what you feed an approximation.** Feeding tanh a bf16-rounded argument amplified `silu`'s error by 1.35x. In softmax, take the row maximum over the raw input and scale it through the same multiply as the data; a kernel that did otherwise returned all-zero rows on large inputs.
* **Test each entry point on its own**, plus one test that composes them.
* **Keep the ABI.** Every `extern "C"` name, signature and buffer layout is pinned by `test/python/test_kernel_contracts.py` and by the designs that call it.

## <u>Build hygiene</u>

Make sure every number comes from the source you think it does.

* Point `MLIR_AIE_KERNEL_SOURCES` at the tree you are editing. For the "before" version, extract a snapshot (`git archive <commit> aie_kernels aie_runtime_lib`), including every file the kernel includes: an attention kernel that includes `mm.cc` changes when `mm.cc` does. Snapshot any Python-side parameters the old kernel needs, too, such as its stack size.
* Run both versions back to back. Kernels you did not change must reproduce to the cycle; if they do not, the setup is the problem.
* Kernel source bytes, the `MLIR_AIE_KERNEL_SOURCES` path and the library harness's stack size are part of the JIT cache key, but not every Python-side design parameter is. Clear the cache (`NPU_CACHE_HOME`) before a run you intend to report. A run that finishes suspiciously fast probably compiled nothing.
* Record the kernel tree, its uncommitted changes and the time with each number. The performance check's provenance line records the commit, the Peano version, `MLIR_AIE_KERNEL_SOURCES` when it is set, and a digest of the kernel sources the factories compiled, uncommitted edits included. Two rows with the same digest ran the same kernel sources.
* Check the numbers in comments and commit messages against the object and the hardware before you rely on them.
* Change one thing at a time, and keep the rejected variants with the number that rejected them.
* Leave the `*_scalar` variants alone: they are the references the vector kernels are tested against.

## <u>Exercises</u>

1. Reproduce the [worked example](#a-worked-example-add): time `add/1024x16/bfloat16` with the shipped kernel and with the one-chain loop below in a copy of [add.cc](../../../aie_kernels/eltwise/add.cc), and read both remarks rows.
    ```C++
    T_in *__restrict pA1 = a;
    T_in *__restrict pB1 = b;
    T_out *__restrict pC1 = c;
    const int F = N / vec_factor;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
      aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
      pA1 += vec_factor;
      aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
      pB1 += vec_factor;
      aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
      aie::store_v(pC1, cout);
      pC1 += vec_factor;
    }
    ```
    <details markdown="1"><summary>Show answer</summary>
    390 and 150 cycles; II 12 and II 18, both with `NS=1`. The wall-clock `npu_us` does not move beyond noise, because the core is busy for about 1.5% of the device time.
    </details>

1. The performance check sizes the trace buffer from the number of intervals the contracts declare (`kd.traced_intervals`). Call `kd.cycles_per_call` for the `add/1024x256/bfloat16` case with `trace_size=16384` instead. What comes back, and how can you tell?
    <details markdown="1"><summary>Show answer</summary>
    A 16 KB buffer fills after 91 intervals of this design, so `kernel` holds the first 91 calls and `truncated` is `True`. Those 91 are still labelled correctly, because the buffer keeps the start of the stream in call order. In a performance row, `truncated` appears in the `range`.
    </details>

1. In your copy of [scale.cc](../../../aie_kernels/eltwise/scale.cc), change the scalar kernel's `c[i] = factor * a[i];` to `c[i] = a[i] / factor;`. Rebuild the 32-bit scalar design from [section 4c](../section-4c) and run `llvm-nm -u` on the kernel object. What do you see, and what does it cost?
    <details markdown="1"><summary>Show answer</summary>
    `U __divsi3`: Peano lowers the 32-bit integer divide to a library routine, so every element makes a function call. The multiply version lists no undefined symbols. On npu2 the call goes from 10761 to 51972 cycles, almost 5x slower. (The host check now reports errors because the result changed, so `make trace` stops before parsing. Run `parse.py` and `get_trace_summary.py` by hand as in [section 4b](../section-4b) to see the count.)
    </details>

1. Run `make trace` in the [single-core matrix multiply](../../../programming_examples/basic/matrix_multiplication/single_core) example and split the intervals with the histogram snippet above. How many populations are there, which kernel is each, and how many intervals did the trace capture compared with the calls the design makes?
    <details markdown="1"><summary>Show answer</summary>
    Two: 70-91 cycles (32 intervals) is `zero`, and 537-553 (521 intervals) is the matmul. That is 553 intervals against 4096 matmul calls plus 256 `zero` calls: the trace buffer filled. The matmul call's first and typical value is 537; the pooled min of 70 describes the other kernel.
    </details>

## <u>Automating this workflow</u>

The [aie-kernel-opt](../../../skills/aie-kernel-opt/SKILL.md) agent skill automates this workflow: the static report, the gate, a back-to-back traced A/B and one commit per kernel.

-----
[Prev](../section-4c) &middot; [Top](../../section-4) &middot; [Next](../../section-5)
