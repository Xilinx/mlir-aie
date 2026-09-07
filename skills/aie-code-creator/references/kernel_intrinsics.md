<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# AIE C++ Kernel Intrinsics Cheatsheet

> Before writing a kernel from this cheatsheet, check
> [`builtin_kernels.md`](builtin_kernels.md) — `aie.iron.kernels` already ships maintained,
> vectorized kernels for element-wise ops, reductions, activations, matmul, conv2d, and vision.
> This file is for the cases those don't cover.

All code targets `aie_api/aie.hpp`. Always include `aie_kernel_utils.h` for portable loop annotations.

## Standard kernel file boilerplate

```cpp
#define NOCPP
#include <cstdint>
#include <type_traits>
#include "aie_kernel_utils.h"        // copy from aie_kernels/aie_kernel_utils.h, or adjust path
#include <aie_api/aie.hpp>

// Templated implementation
template <typename T_in, typename T_out, int N>
static inline void eltwise_add_impl(const T_in *__restrict a,
                                    const T_in *__restrict b,
                                    T_out      *__restrict c) {
    constexpr int VEC = 32;              // natural bf16 width; see architecture.md
    static_assert(N % VEC == 0, "N must be divisible by VEC");
    constexpr int F = N / VEC;

    event0();
    AIE_PREPARE_FOR_PIPELINING           // no-op under Peano; only Chess honors it
    AIE_LOOP_MIN_ITERATION_COUNT(F)      // guaranteed minimum trip count of this loop
    for (int i = 0; i < F; ++i) {
        aie::vector<T_in,  VEC> va = aie::load_v<VEC>(a); a += VEC;
        aie::vector<T_in,  VEC> vb = aie::load_v<VEC>(b); b += VEC;
        aie::vector<T_out, VEC> vc = aie::add(va, vb);
        aie::store_v(c, vc); c += VEC;
    }
    event1();
}

// extern "C" wrapper(s) — these names go into the IRON Kernel(...) declaration
// Keep __restrict (and const on inputs) on the wrapper itself, not just the
// templated impl — this is the symbol IRON links against, and __restrict here
// is what lets the compiler pipeline the loop.
extern "C" {
void eltwise_add_bf16_vector(const bfloat16 *__restrict a,
                              const bfloat16 *__restrict b,
                              bfloat16 *__restrict c) {
    eltwise_add_impl<bfloat16, bfloat16, 1024>(a, b, c);
}
} // extern "C"
```

## Loop annotations (from `aie_kernel_utils.h`)

| Macro | What it does |
|-------|--------------|
| `AIE_PREPARE_FOR_PIPELINING` | Ask the modulo scheduler to make the loop body a single-cycle pipeline |
| `AIE_LOOP_MIN_ITERATION_COUNT(n)` | Promise the loop runs at least `n` times — required for scheduling |
| `AIE_LOOP_MAX_ITERATION_COUNT(n)` | Promise the loop runs at most `n` times |
| `AIE_LOOP_RANGE(min, max)` | Combo of the two above |
| `AIE_LOOP_UNROLL(n)` | Unroll by factor `n` |
| `AIE_LOOP_UNROLL_FULL` | Fully unroll |
| `AIE_LOOP_NO_UNROLL` | Block unrolling |
| `AIE_TRY_INITIATION_INTERVAL(n)` | Request an initiation interval of `n` (**Peano only**) |
| `AIE_PREPARE_FOR_POSTPIPELINING` | Disable Peano's pipeliner for this loop (**Peano only**) |
| `AIE_NO_PREPARE_FOR_PIPELINING` | Block pipelining (rare; for setup loops) |

Use the macros rather than backend-specific pragmas. Note they don't all map onto both backends: `AIE_LOOP_MIN_ITERATION_COUNT` / `AIE_LOOP_RANGE` / `AIE_LOOP_UNROLL*` become `clang loop` pragmas under Peano/AIECC and the equivalent under Chess, but `AIE_PREPARE_FOR_PIPELINING` and `AIE_LOOP_FLATTEN` map to `chess::` hints only — they are **no-ops under Peano**. All expand to nothing on host builds. Keep them in (free under Chess) but rely on `AIE_LOOP_MIN_ITERATION_COUNT` for pipelining on Peano.

## Vector load / store

```cpp
aie::vector<T, N> v = aie::load_v<N>(ptr);   // contiguous load of N elements
aie::store_v(ptr, v);                         // contiguous store
// Aligned loads are required for N ≥ natural width; use restrict + aligned pointers from ObjectFifo.
```

## Element-wise arithmetic

```cpp
auto sum  = aie::add(va, vb);              // vector + vector
auto prod = aie::mul(va, vb);              // returns accumulator for integer types
auto sc   = aie::add(va, scalar);          // vector + scalar (broadcasts)
auto neg  = aie::neg(va);
auto mx   = aie::max(va, vb);              // element-wise max  (use for ReLU vs zeros)
auto mn   = aie::min(va, vb);
auto absv = aie::abs(va);
```

## Multiply-accumulate (non-MMUL)

```cpp
aie::accum<acc32, VEC> acc;
acc.from_vector(aie::load_v<VEC>(c_ptr));
acc = aie::mac(acc, va, vb);                // acc += va * vb
aie::store_v(c_ptr, acc.template to_vector<T_out>(shift));  // shift = number of fractional bits to drop
```

## Broadcast / zeros / ones

```cpp
auto zeros  = aie::zeros<T, VEC>();
auto allone = aie::broadcast<T, VEC>(T(1));
auto sclrv  = aie::broadcast<T, VEC>(scalar_value);
```

## Type conversion

```cpp
aie::vector<float, 16>  vf  = ...;
aie::vector<bfloat16, 32> vbf = aie::to_v32bfloat16(vf);   // (when widths line up)
auto v_i8  = acc.template to_vector<int8_t>(shift);
acc.template to_vector<bfloat16>();                         // no shift for float-acc → bf16
```

## MMUL — matrix multiply intrinsic (the perf path)

```cpp
// 1. Define the MMUL shape: r × s @ s × t → r × t   (see architecture.md for valid shapes)
using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;

// 2. Sizes of each operand fragment in elements:
//    MMUL::size_A = r*s, MMUL::size_B = s*t, MMUL::size_C = r*t

// 3. Construct an MMUL initialized from a partial accumulator in C
aie::vector<T_out, MMUL::size_C> c_init = aie::load_v<MMUL::size_C>(pC);
MMUL C00(c_init);

// 4. Inner reduction loop
for (unsigned k = 0; k < K_TILES; ++k) {
    auto A = aie::load_v<MMUL::size_A>(pA); pA += MMUL::size_A;
    auto B = aie::load_v<MMUL::size_B>(pB); pB += MMUL::size_B;
    C00.mac(A, B);                          // C00 += A · B
}

// 5. Write back (optionally with shift for fixed-point)
aie::store_v(pC, C00.template to_vector<T_out>());
// or: aie::store_v(pC, C00.template to_vector<T_out>(SHIFT));
```

### Outer expansion (per-tile macro-kernel)

For real GEMMs you unroll the MMUL across an outer block (e.g., 4×4 expansion: hold 16 MMUL accumulators in registers, share each loaded A across 4 Bs and vice versa). See `mlir-aie/aie_kernels/aie2/mm.cc` for canonical implementations.

```cpp
// Sketch of 2x2 expansion
MMUL C00(load_v<MMUL::size_C>(pC1));
MMUL C01(load_v<MMUL::size_C>(pC1 + MMUL::size_C));
MMUL C10(load_v<MMUL::size_C>(pC2));
MMUL C11(load_v<MMUL::size_C>(pC2 + MMUL::size_C));

AIE_PREPARE_FOR_PIPELINING
AIE_LOOP_MIN_ITERATION_COUNT(2)
for (unsigned k = 0; k < K_TILES; ++k) {
    auto A0 = load_v<MMUL::size_A>(pA1); pA1 += MMUL::size_A;
    auto A1 = load_v<MMUL::size_A>(pA2); pA2 += MMUL::size_A;
    auto B0 = load_v<MMUL::size_B>(pB1); pB1 += MMUL::size_B * colB;
    auto B1 = load_v<MMUL::size_B>(pB2); pB2 += MMUL::size_B * colB;
    C00.mac(A0, B0); C01.mac(A0, B1);
    C10.mac(A1, B0); C11.mac(A1, B1);
}
```

## Reduction

Horizontal vector reduction is done by repeated `shift_bytes` + `aie::add/max`:

```cpp
template <typename T, int N>
T reduce_sum(const T *__restrict in, int total) {
    using V = aie::vector<T, N>;
    V acc = aie::zeros<T, N>();
    event0();
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(8)
    for (int i = 0; i < total; i += N) {
        acc = aie::add(acc, aie::load_v<N>(in + i));
    }
    event1();
    return aie::reduce_add(acc);            // final horizontal sum
}

// reduce_max: same shape, use aie::max() and aie::reduce_max()
```

## Saturation / rounding (for integer kernels)

```cpp
::aie::set_saturation(aie::saturation_mode::saturate);    // clamp on overflow (vs wrap)
::aie::set_rounding  (aie::rounding_mode::symmetric_inf); // shift rounding mode
```

Call once at the top of the kernel (or in a wrapper) before any vector op that could overflow.

## Accumulator type quick-pick

| Input | Multiply result | Accumulator |
|-------|-----------------|-------------|
| `int8`  | `int16` / `int32`  | `acc32` |
| `int16` | `int32`            | `acc32` (or `acc64` for wide) |
| `int32` | `int64`            | `acc64` |
| `bfloat16` | `float`         | `accfloat` (`accauto` in MMUL) |
| `float`  | `float`           | `accfloat` |

If you're unsure, use `accauto` inside `aie::mmul<>`; the compiler picks.

## Common operations table

| Op | Call |
|----|------|
| Vector add | `aie::add(va, vb)` |
| Vector mul (returns acc) | `aie::mul(va, vb)` |
| Vector MAC | `aie::mac(acc, va, vb)` |
| Vector load | `aie::load_v<N>(ptr)` |
| Vector store | `aie::store_v(ptr, v)` |
| Broadcast scalar | `aie::broadcast<T, N>(s)` |
| Zero vector | `aie::zeros<T, N>()` |
| Element max | `aie::max(va, vb)` |
| Element min | `aie::min(va, vb)` |
| Element abs | `aie::abs(va)` |
| Horizontal sum | `aie::reduce_add(v)` |
| Horizontal max | `aie::reduce_max(v)` |
| Acc → vector | `acc.template to_vector<T>(shift)` |
| Vector → acc | `acc.from_vector(v)` |
| Matrix multiply tile | `MMUL C(c0); C.mac(A, B); store_v(p, C.to_vector<T>());` |
