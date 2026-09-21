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
softmax's input range after the row-max shift. The poly holds ~8.9e-5
relative error across its default domain, `x` in `[-111, 127.999)` (and a
bounded ~7.8e-4 in the narrow `[127.999, 128)` sliver right at the upper
clamp). The lower clamp is movable down to the hard floor of `-126` at up
to 6.5e-3; see
[`aie_kernels/aie2p/exp2f_vec.cc`](../../../aie_kernels/aie2p/exp2f_vec.cc)
for the measured accuracy table and the derivation (`x = k + f`, `2^k`
built directly in the float32 exponent field, `2^f` from the poly).

Four cores each operate on `1024` `float32` numbers.

## Source Files

1. [`vector_exp2f.py`](vector_exp2f.py) - IRON structural design plus the
   host driver, mirroring [`basic/vector_exp`](../vector_exp)'s structure
   with `kernels.exp2f_vec` / `kernels.exp2f_vec_ref` in place of
   `kernels.bf16_exp` / `kernels.bf16_exp_ref`, and `float32` in place of
   `bfloat16`.
2. [`exp2f_vec.cc`](../../../aie_kernels/aie2p/exp2f_vec.cc) - the kernel.
   Its `__attribute__((noinline))` is load-bearing; see the comment there.

## Usage

```shell
python3 vector_exp2f.py
```

The IRON JIT runtime detects the attached NPU generation automatically
(aie2p / NPU2 required; the kernel raises `NotImplementedError` on aie2).

The host driver builds four input blocks: a dense grid over `[-111, 0]`, a
random sample over the same range, a block from `-500` to just below
`-111` that exercises the `x = max(x, -111)` lower clamp, and a
positive-domain block (a dense grid over `[0, 127]` plus explicit values
straddling the upper clamp at k = 127, 128, 129, ... up to 1e30).

It gates the two graded blocks on max relative error against a float64
`2**x` reference at `5e-4`, well above the measured ~8.9e-5. The clamp
block is checked only for NaN/Inf, since its contract is
`2**max(x, -111)`. The boundary block is gated on VALUES rather than
finiteness: no negative-signed output anywhere, and bit-exact `+inf` for
every k >= 128, where `2**x` exceeds `FLT_MAX`. Finiteness alone would
miss the exponent-field failure mode, which is a finite wrong-signed
result.
