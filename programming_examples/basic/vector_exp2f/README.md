<!---//===- README.md -----------------------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Vector 2^x (software minimax poly)

Demonstrates `kernels.exp2f_vec`, a software degree-5 minimax polynomial
approximation of `2**x` for `float32`, aie2p only. This is the accuracy
alternative to [`basic/vector_exp`](../vector_exp)'s LUT-based `bf16_exp`:
the LUT's relative error is domain-dependent and grows sharply for
negative inputs (up to 49.1% on `[-100, 0]`, measured), which is exactly
softmax's input range after the row-max shift. The poly holds ~8.5e-5
relative error across its full documented domain, `x` in `(-100, 127.999)`
(and a bounded, documented ~7.8e-4 in the narrow `[127.999, 128)` sliver
right at the upper clamp). See
[`aie_kernels/aie2p/exp2f_vec.cc`](../../../aie_kernels/aie2p/exp2f_vec.cc)
for the full domain contract, the accuracy table, and the derivation
(`x = k + f`, `2^k` built directly in the float32 exponent field, `2^f`
from the poly). That file's header also documents two device-only bugs
found and fixed in this kernel's history: an unclamped upper bound that
wrapped `2^x` into a wrong-signed finite value for `x >= 129`, and a
device-measured hardware quirk where `aie::mul` returns NaN, not +inf, on
f32 overflow, which is why the fix is an explicit compare against the
true `x` plus a select rather than a wider clamp.

Four cores each operate on `1024` `float32` numbers.

## Source Files

1. [`vector_exp2f.py`](vector_exp2f.py) - IRON structural design plus the
   host driver, mirroring [`basic/vector_exp`](../vector_exp)'s structure
   with `kernels.exp2f_vec` / `kernels.exp2f_vec_ref` in place of
   `kernels.bf16_exp` / `kernels.bf16_exp_ref`, and `float32` in place of
   `bfloat16`.
2. [`exp2f_vec.cc`](../../../aie_kernels/aie2p/exp2f_vec.cc) - the kernel.
   `__attribute__((noinline))` is load-bearing there (Peano -O2 miscompiles
   the always-inlined form to NaN on a register-pressure-heavy caller); see
   that file's header comment, not repeated here.

## Usage

```shell
python3 vector_exp2f.py
```

The IRON JIT runtime detects the attached NPU generation automatically
(aie2p / NPU2 required; the kernel raises `NotImplementedError` on aie2).

The host driver builds four input blocks: a dense grid over `[-100, 0]`
(the kernel's characterized softmax domain, both endpoints included), a
random sample over the same domain, a block from `-500` to just below
`-100` that exercises the kernel's `x = max(x, -100)` lower clamp, and a
positive-domain block (a dense grid over `[0, 127]` plus explicit values
straddling the upper-clamp boundary at k = 127, 128, 129, ... up to 1e30).
It reports max-abs-error and rel-L2 against a float64 `2**x` reference for
the characterized domains (gated at `5e-4` max-abs or max-rel, well above
the kernel's measured ~8.5e-5, some margin for cross-run float32
accumulation noise), separately checks the `< -100` clamp block stays
finite (this kernel family's known failure mode on this target is silent
inf/nan, not just imprecision) without gating on tight accuracy there,
since `[-500, -100)` is outside the characterized domain, and gates the
positive-boundary block on VALUES, not just finiteness: no negative-signed
output anywhere, and bit-exact `+inf` for every k >= 128 (2**x already
exceeds `FLT_MAX` there, so the kernel's contract is exact +inf, not an
approximation). That last block is the regression test for this kernel's
history: a pre-fix build fails it outright (see the .cc header for the
device-measured symptoms), which a plain `isfinite()` check would not
have caught.
