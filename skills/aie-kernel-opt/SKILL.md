---
name: aie-kernel-opt
description: Standalone guide to optimizing AIE / Peano-compiled kernels (INT8 conv, matmul, attention, elementwise). Covers the measure-first methodology (baseline, bit-exact gate, ablation, verify-in-.o) AND the catalog of concrete levers in priority order — loop hints, compile-time constants, killing __divsi3, branch-splitting, vectorized epilogue, operand-layout pre-pack, explicit wide packing, wider mmul, DMA layout offload — each with the constraints to respect and a measured delta.
license: Apache-2.0 WITH LLVM-exception
---

<!--
This file is licensed under the Apache License v2.0 with LLVM Exceptions.
See https://llvm.org/LICENSE.txt for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

Copyright (C) 2026, Advanced Micro Devices, Inc.
-->

# AIE kernel optimization (the "what to change" menu)

This is a catalog of the levers that actually move AIE/Peano kernels —
in rough priority order, each with the constraint to respect so it pays
off. Use it once you've identified a specific kernel worth optimizing.

The deltas cited are illustrative results from hand-optimizing example
INT8 kernels (convs, 1x1 projections, depthwise, GEMM, softmax-style
attention) — they show the headroom a given lever can unlock relative to a
straightforward first implementation, not a benchmark of any shipping
product. The *mechanisms* are general to AIE/Peano; the conv-shaped
examples are just illustrations — read "inner reduction loop", "output
loop", "epilogue" generically for your kernel. Concrete vector widths and
`mmul<...>` shapes in the examples reflect one AIE generation (the numbers
differ across AIE/AIE-ML/AIE2P); take them as worked examples and check the
aie-api headers for your target's actual widths.

## Architecture names

The codebase, marketing material, and silicon teams use different names for
the same architecture. The internal name (left column) is what you pass to
the toolchain (`--target=aie2-none-unknown-elf`, `--dev npu2`, etc.) and
what gates code in the aie-api headers.

| internal name(s)   | marketing name   | chips with this architecture          |
|--------------------|------------------|---------------------------------------|
| `aie`              | AIE              | Versal VCK5000                        |
| `aie2` (== NPU1)   | AIE-ML (XDNA)    | Phoenix                               |
| `aie2p` (== NPU2)  | XDNA2            | Strix Point, Strix Halo, Krackan      |
| `aie2ps`           | AIE-MLv2         | Telluride                             |

## Before you start (methodology — don't skip)

**Measure before modeling; verify before trusting.** Static analysis of
MLIR or kernel C is unreliable on AIE — paper-compute estimates have
mispredicted real HW time by 5–300× depending on the kernel. Every claim
about where time goes must be backed by an HW measurement or an ELF
inspection, not a cycle count in your head.

1. **Baseline first.** Build clean, run on HW, record the median wall time
   over ≥20 iterations. If the std-dev is >1% of the mean, fix that
   (background load, throttling, noisy runtime) before optimizing — you
   can't measure a 5% win against 10% noise.
2. **Establish a bit-exact gate from the start.** Capture the exact
   command that produces a PASS against a reference. Every change below
   must keep it passing. Don't take wall-time numbers from a broken build
   — wrong data computes in a different amount of time than correct data.
3. **Confirm the kernel is on the critical path before optimizing it.**
   The cheapest reliable attribution is *ablation*: gate the kernel call
   behind a flag (leave the locks/acquires/releases intact, just skip the
   compute), re-time, and look at the delta. A kernel contributing 99 ms
   is 20× more worth your time than one contributing 4 ms. HW packet
   tracing is more difficult to interpret; prefer ablation for attribution.
4. **One change at a time.** If you edit the C source AND the build flags
   AND the dataflow at once, you can't tell which moved the needle.
5. **Verify the mechanism in the `.o` before believing your cycle-model.**
   In practice, *both* "obvious" explanations for a given regression
   (register spill; doubled scalar gather) have turned out wrong on
   objdump inspection. A regression you can't see in the codegen is one you don't
   understand yet. Useful checks:
   - `llvm-nm --print-size build/X.o` — spill ≈ noticeably bigger function.
   - `llvm-objdump -d build/X.o | grep -coE '\bv(lda|ldb|mac|mul|st)\b'` —
     confirm vector ops actually appear (scalar-only `mac`/`lda`/`st` means
     the kernel didn't vectorize at all — fix that before anything else).
   - `llvm-nm build/X.o | grep __div` — should be empty (see lever #3).

   **Read the disassembly — for a compute-bound kernel it can replace HW
   timing.** AIE cores don't stall (no cache, no dynamic scheduling, no
   branch misprediction), so the hot loop's instruction count ÷ core clock
   ≈ its execution time. This is *not* the source-level paper-compute
   modeling warned against above: that fails because you can't know what the
   compiler emitted, whereas counting instructions in the emitted assembly
   is counting exactly what runs. Disassemble with relocations (the `-r`
   marks the start/end of the zero-overhead loop):
   ```shell
   $PEANO_INSTALL_DIR/bin/llvm-objdump -d -r build/X.o
   ```
   Or compile straight to assembly, skipping the object file:
   ```shell
   $PEANO_INSTALL_DIR/bin/clang++ -O2 -std=c++20 \
     -I$MLIR_AIE_INSTALL_DIR/include --target=aie2-none-unknown-elf \
     -S my_kernel.cc -o kernel.s
   # (swap aie2 for aie2p/etc. per the architecture table above)
   ```
   In the **zero-overhead loop** body (a hardware loop whose bounds are
   programmed once, so it repeats a fixed instruction block with no
   per-iteration branch/counter overhead), count two things: total
   instructions, and inserted **`nop`s** — padding emitted when the compiler
   can't fill a VLIW slot, i.e. visible evidence the loop isn't packed
   tight. Fewer of each is directly faster.
6. **Don't trust "I cleaned the build."** Make tracks file mtimes, not
   flag changes, and `.prj/` dirs cache stale ELFs. Confirm the `.o`
   actually rebuilt and contains your edit (see "Verifying any change took
   effect" at the end).

---

## Priority order

Bigger, more reliable levers first. Don't jump to the wider-mmul lever
before doing the loop-hint / branch / epilogue levers — the structural
wins live there.

### 1. Loop hints — cheapest, often biggest

Include `aie_kernel_utils.h`. Decorate every non-trivial for-loop. **The
macro goes BEFORE the `for` statement, never inside the parens** (it's a
clang pragma, not the chess attribute form).

- `AIE_PREPARE_FOR_PIPELINING` on the loop you want software-pipelined
  (usually the outermost loop over output elements/tiles). Its effect is
  toolchain-dependent; leave it in for portability across toolchains, but
  treat the other hints below as the ones doing the work.
- `AIE_LOOP_RANGE(N, N)` on every fixed-trip-count loop. This is a
  **trip-count hint only** — Peano still emits a runtime loop, it just
  knows the bounds.
- `AIE_LOOP_RANGE(lo, hi)` on bounded-but-variable loops (e.g. a reduction
  whose count depends on a border/edge condition).
- **`AIE_LOOP_UNROLL_FULL` on small fixed-count inner loops** (trip count
  is a small constant like 3) — *mandatory* if the loop body switches or
  branches on the loop variable (a helper that does `switch(i)` on the
  index, an index-dependent address calc). `AIE_LOOP_RANGE` alone leaves
  the switch as a runtime branch and blocks VLIW pipelining of the body.
  **Verified on a 3x3 conv: changing the window loop from
  `AIE_LOOP_RANGE(3,3)` to `AIE_LOOP_UNROLL_FULL` was a one-line diff for
  −47% kernel time.** RANGE alone was insufficient.

### 2. Make shapes and trip counts compile-time constants

This is the lever that unlocks most of the others, so do it early. Peano
optimizes far more aggressively when the bounds, strides, and shapes it
needs are visible as constants at compile time rather than arriving as
runtime kernel arguments. `constexpr` (not just `const`) is what guarantees
that visibility: a `const int` initialized from a runtime argument is still
a runtime value, whereas a `constexpr` — or a value threaded in through a
template parameter or a `-D` compile definition — is a literal the
optimizer can fold against.

What you get when the shapes are `constexpr`:
- **Loop hints actually bind.** The `AIE_LOOP_RANGE` / `AIE_LOOP_UNROLL_FULL`
  hints from lever #1 only help when the trip count is a known constant;
  fed a runtime bound, they have nothing to fold.
- **Address arithmetic folds to immediates.** `ic_t * stride + x * 64` with
  constant strides becomes immediate-offset loads instead of runtime
  multiplies.
- **Power-of-2 divides and multiplies become shifts** instead of `__divsi3`
  software calls (see lever #3).

How to apply: prefer template parameters or `constexpr` for per-call-site
shapes. If the same kernel serves multiple shapes, pass them as `-D`
compile definitions (one specialization per shape) with a runtime-argument
fallback path, so the hot configurations get the constant-folded code and
the rest still build. Compile-time shapes alone are typically a small
direct win (~2%), but they are the precondition that makes levers #1, #3,
and the addressing in #6 pay off.

### 3. Eliminate `__divsi3` / signed power-of-2 ops on the hot path

A signed power-of-2 divide is not equivalent to a shift (`int32_t a / 8` ≠
`a >> 3` — rounding toward zero differs for negatives), so it lowers to a
`__divsi3` software call rather than a single shift. That external call
also acts as a **vectorization barrier** for the whole enclosing function,
so removing it can unlock much more than the divide itself.

```c
const int n_tiles = (uint32_t)channel_count / 8u;  // → shift, not __divsi3
```

The same applies to signed multiply by a power of 2 when the operand can be
negative: `idx * STRIDE` where `idx` can be −1 (e.g. a left/top border
offset) won't fold to a shift even with `STRIDE` a compile-time
power-of-2 (negative-shift would be UB). Fix by **peeling the borders** out of the
hot loop — handle the edge iterations separately so the interior path has
a provably non-negative index and the address arithmetic folds to shifts.
Verify with `llvm-nm build/X.o | grep __div` — the call should disappear.

### 4. No branches inside the inner compute loop

A per-iteration `if`/`else`/ternary inside the VLIW inner loop blocks
Peano's back-to-back `vld + vld + vmac` (or `vld + vop`) issue — **even
when the predicate is a compile-time-known constant array** indexed by the
loop variable.

The common offender: a kernel that reduces over inputs from **mixed
sources** (some operands in one memory layout, some in another) and
selects per-iteration with `if (is_packed[i]) vec_load else scalar_pack`.
Don't. Split into **K straight-line per-source loops**, each body uniform:

```cpp
for (int i = 0; i < n_src0; ++i) { /* pure vec_load + op */ }
for (int i = 0; i < n_src1; ++i) { /* pure scalar-pack + op */ }
```

**Verified: branched → split was −7% on a 3-source reduction kernel, on
top of staying bit-exact.** A `(i < n) ? a : b` ternary inside the loop
counts as a branch too — split per-source even when both branches load.

### 5. Vectorize the epilogue (bias + requantize/SRS + clamp + activation)

Once the inner loop is tight, the scalar tail becomes the bottleneck:
processing the output elements one at a time (add bias, shift-round-
saturate, clamp, activation lookup). In practice, **vectorizing the
epilogue was the single biggest per-kernel lever** — roughly 20–25% per
kernel even after loop hints (#1) and compile-time shapes (#2) were
already in.
**In one measured case a pipeline stage went 27.75 → 15.99 ms (−42% /
+74% fps) across three kernels, almost entirely from epilogue
vectorization.**

Pattern (for any kernel with a scalar
`acc[i] + bias; round_shift; clamp; out[i] = lut[...]` tail):

1. **Match the rounding mode to your reference.** If the scalar reference
   uses round-half-to-even (banker's rounding), set
   `aie::set_rounding(aie::rounding_mode::conv_even)` so the vector
   `acc.to_vector<int8>(shift)` is bit-identical. A mismatched mode
   (e.g. `positive_inf`) produced ~0.2% element mismatch with max diff 3 —
   small, but an activation LUT amplifies ±1 rounding diffs into larger
   output errors, so it fails a bit-exact gate.
2. **Bias-init / fold the add into a vector op.** Load the bias values,
   broadcast/`concat` them to the accumulator width, `aie::add` on the
   int32 vector, then `acc.from_vector(sum); acc.to_vector<int8>(shift)`
   to do shift-round-saturate in one shot. Replaces N scalar bias-adds +
   N scalar SRS.
3. **Keep the epilogue helper `always_inline` and pass the accumulator
   by reference.** A non-inline pass-by-ref of an `aie::mmul`/`aie::accum`
   breaks the ABI. Only break it out of inline (and watch tile program
   memory) if PM pressure forces it.
4. **The activation-table lookup (`lut[...]`) stays scalar.** Vectorize the
   bias + SRS + clamp around it; the per-element table lookup and store
   remain a scalar loop.
5. **For a residual/skip add, use a wide-enough intermediate.** Accumulate
   into an **int16** vector and add non-saturating when the value range
   fits, then `acc.to_vector<int8>(0)` for the saturating narrow. `aie::add`
   on int8 vectors wraps rather than saturates, so the saturation has to
   come from the narrowing step.

### 6. Pre-pack operand layout to kill the scalar gather feeding mmul

The scalar operand-build that feeds a matmul — `for(p,k) a_buf[p*W+k] =
src[...]; load_v<...>(a_buf)` into a stack scratch — is frequently the
**dominant cost** of an mmul-based kernel: the per-mmul cost is several
times what the same mmul achieves once the operand is delivered in
mmul-ready layout. It's a non-issue in the canonical matmul example
(`aie_kernels/aie*/mm.cc`) because there DMA delivers the operand already
in mmul-ready layout; kernels that hand-assemble the operand from
L1-shared buffers (not via DMA) pay the repack cost instead.

Root cause: `aie::concat` has a minimum operand width (on current AIE it
won't concat sub-128-bit vectors), so a wide operand vector can't be
assembled from strided narrow loads — the scalar byte-copy fallback is the
only option *when the source layout doesn't match the mmul's expected
operand layout*.

**Fix — requires controlling BOTH the producer and the consumer of the
intermediate buffer:**
- **Producer (free):** an mmul result `acc.to_vector<int8>(shift)` is
  *already* in the byte order the next mmul wants as its A operand. Have
  the producer write it as one vector store into the intermediate buffer
  in that layout. No transform on the producer side.
- **Consumer:** reads the operand with a single aligned `vld`. If the next
  kernel needs a shifted window of that data (e.g. a sliding conv window),
  build the shifted vector with `concat` + `shuffle_down` + `extract`
  rather than re-gathering scalar-wise:
  ```cpp
  auto combined = aie::concat(lo_block, hi_block);   // ≥128-bit
  return aie::shuffle_down(combined, shift_amt).template extract<N>(0);
  ```
  Total bytes are identical to the naive layout; both ends must agree on
  the stride.

**Verified: a multi-kernel design went 14.97 → 11.44 ms (−23.6%)
standalone; a standalone conv went 207 → 521 fps (+135%).** The estimate (0.8–1.5 ms)
badly undershot the actual (3.5 ms) because removing the scalar pack also
let Peano schedule the surrounding loop more tightly — a secondary
register-allocation effect invisible in any bytecount model. Only applies
when both ends are in your code (not a library/external consumer that
expects the naive layout).

### 7. Pack wide copies explicitly

For a small fixed-size element copy, write the wide move explicitly rather
than relying on a scalar byte loop to be widened for you — in deeply-nested
or stack-allocated-buffer contexts the explicit form is what guarantees a
single wide load/store:

```cpp
*reinterpret_cast<uint64_t *>(d) = *reinterpret_cast<const uint64_t *>(src);  // 8 bytes
// 4 bytes → uint32_t; 16 → two uint64; 32 → four. alignas(N) both ends.
```

**Verified: a per-row scratch pack went 3.44 → 2.95 ms (+14%) from this
change alone.** If the explicit form produces wrong output, check for a
Worker stack overflow into an adjacent buffer — bump the worker's stack
size (e.g. `Worker(stack_size=2048)`).

### 8. Wider mmul shape — only where it amortizes

An int8 `mmul<r, s, t>` comes in several widths (the exact set is
arch-dependent; check the aie-api headers for your target — e.g. AIE2P int8
offers `mmul<4,8,8>`, `<4,16,8>`, `<8,8,8>`, `<8,16,8>`). A wider M (r)
doubles output rows per call; a wider K (s=16) needs the weight operand
re-laid-out (e.g. `...I8O8` → `...I16O8`), which isn't free.

- **When there's a deep inner reduction loop per output (e.g. 3x3+ conv
  with a kx·ky·channels reduction): usually a direct win** — halving the
  outer call count cuts a large per-output workload, swamping any extra
  operand-gather cost.
- **When the per-output work is tiny (1x1 conv / GEMV-ish) AND the operand
  is built by scalar gather: it can REGRESS.** `mmul<M,8,8>` does `M×8`
  byte copies per reduction step; going M=4→8 doubles the gather while
  only halving an already-small call count. **Verified: a 1x1 kernel went
  3.34 → 3.58 ms (+7%, reverted).** Vectorize the operand gather (lever
  #6) *first*; only then does the wider shape pay back.
- **Non-contiguous operands (strided/gathered inputs): tried + reverted.**
  The wider shape can't help if the operand can't be loaded contiguously.

### 9. Move pure data-movement into DMA (free a whole tile)

If a kernel body is *only* a data rearrangement — a strided copy,
transpose (`dst[a,b] = src[b,a]`), deinterleave, or gather, with **no
arithmetic** — it's exactly what the memtile DMA `dims_to_stream` /
`dims_from_stream` machinery was built for. Move it onto the upstream
ObjectFifo (`cons().forward(dims_to_stream=…)`) instead of running a
kernel. This drops an extern call + a `.o` and frees the consumer tile's
program memory — **worth doing even when the kernel isn't on the critical
path** (verified: pure-transpose packer kernels that ablation showed were
off the critical path were still worth removing for the PM/tile savings).

**Don't** do this if the kernel also computes (mac, activation, requant) —
the rearrangement is essentially free when overlaid on real compute, so
leave it in the kernel.

Related DMA layout wins:
- **Deinterleave a strided input** so the kernel reads aligned vectors
  instead of a strided scalar gather (e.g. split a stride-2 input into
  even/odd halves per row). **Verified +12% fps** on a strided-input conv.
- A layout transform can live on the **shim tap**, the **compute
  toStream/fromStream**, OR the **memtile forward** — pick the cheapest
  for your dataflow, it's not memtile-only.

**Constraints to respect when forwarding (otherwise expect wrong or zero
data):**
- Aligned vector loads need their natural alignment (a load of an
  N-byte vector wants N-byte alignment, e.g. `load_v<int8,32>` → 32-byte).
  If a deinterleave lands data at an unaligned offset, do 2 aligned loads +
  `shuffle_down` rather than one unaligned load.
- In a multi-kernel chain, pin the forward to a specific column memtile
  (`.forward(tile=Tile(col, 1))`) so the ObjectFifoLink has a shared tile;
  a placement that works standalone can otherwise fail in the chain.

---

## Register-pressure / power note

More compute isn't always the goal. On a kernel that was *off* the
critical path, removing an explicit `AIE_LOOP_UNROLL(2)` from a 4-
accumulator reduction loop cut in-loop register-spill traffic by ~87% and
`.text` by ~8% at **net-zero throughput** — pure power savings, which can
matter for deployment. When adding explicit unroll to a kernel with 4+
simultaneous accumulators, count in-loop spill stores/loads (`vst`/`vld`
of accumulator-state registers) in the disasm before adopting: spill
inside a hot loop is a power tax even when wall time is flat.

---

## Verifying any change took effect

Make's incremental rebuild tracks file mtimes, not flag changes, and
`.prj/` directories cache stale ELFs — a source edit may not reach the
`.o`. After each change:
1. Confirm the `.o` rebuilt (check mtime) and, when in doubt,
   `make clean` / `rm -rf build/*.prj`.
2. Confirm the change is *in* the binary: did the `vmac` count change, did
   `__divsi3` disappear, did the function size move the way you expected?
3. Only then trust the wall-time delta.

The aie-api headers are the ground truth for what each intrinsic lowers to.
In this repo they're vendored at `third_party/aie_api/include/aie_api/`
(`aie.hpp` is the umbrella header; per-arch lowerings live under
`detail/aie*/`); after a build they're also copied to
`install/include/aie_api/`. The matmul programming example
(`programming_examples/.../matrix_multiplication/`) is a good reference for
DMA-delivered mmul-ready operand layouts.
