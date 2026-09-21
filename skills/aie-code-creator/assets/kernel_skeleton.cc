//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Templated, vectorized AIE kernel skeleton.
// Adapt:
//   - the templated `impl` function for your op (currently element-wise add)
//   - the extern "C" wrapper name and concrete types
//   - the VEC factor based on data type (see references/architecture.md)
//
// Compile with the Peano / AIECC toolchain or Chess. Note
// AIE_PREPARE_FOR_PIPELINING is a no-op under Peano/AIECC (only Chess honors
// it); AIE_LOOP_MIN_ITERATION_COUNT is the annotation that enables pipelining
// on both.

#define NOCPP
#include <cstdint>
#include <type_traits>

// Copy `aie_kernels/aie_kernel_utils.h` from the mlir-aie repo next to this
// file (or adjust the path below to wherever it ends up relative to your
// kernel source).
#include "aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// ---------------------------------------------------------------------------
// Templated implementation
// ---------------------------------------------------------------------------
template <typename T_in, typename T_out, int32_t N>
static inline void eltwise_add_impl(const T_in *__restrict a,
                                    const T_in *__restrict b,
                                    T_out *__restrict c) {
  constexpr int32_t VEC =
      32; // natural bf16 width; int8 uses 64 (see references/architecture.md)
  static_assert(N % VEC == 0, "N must be divisible by VEC");
  constexpr int32_t F = N / VEC;

  event0();
  AIE_PREPARE_FOR_PIPELINING      // no-op under Peano; only Chess honors it
  AIE_LOOP_MIN_ITERATION_COUNT(F) // guaranteed minimum trip count of this loop
      for (int32_t i = 0; i < F; ++i) {
    aie::vector<T_in, VEC> va = aie::load_v<VEC>(a);
    a += VEC;
    aie::vector<T_in, VEC> vb = aie::load_v<VEC>(b);
    b += VEC;
    aie::vector<T_out, VEC> vc = aie::add(va, vb);
    aie::store_v(c, vc);
    c += VEC;
  }
  event1();
}

// ---------------------------------------------------------------------------
// extern "C" wrappers — these are the symbols the IRON Kernel(...) refers to
// ---------------------------------------------------------------------------
extern "C" {

void eltwise_add_bf16_vector(const bfloat16 *__restrict a,
                             const bfloat16 *__restrict b,
                             bfloat16 *__restrict c) {
  eltwise_add_impl<bfloat16, bfloat16, /*N=*/1024>(a, b, c);
}

void eltwise_add_i32_vector(const int32_t *__restrict a,
                            const int32_t *__restrict b,
                            int32_t *__restrict c) {
  eltwise_add_impl<int32_t, int32_t, /*N=*/1024>(a, b, c);
}

} // extern "C"
