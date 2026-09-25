//===- swiglu.cc --------------------------------------------*- C++
//-*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include "../common/activations.h" // tanh_bf16_v16
#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#ifndef SWIGLU_ELEMS
#define SWIGLU_ELEMS vector_size
#endif

// out = (x * w1) * silu(x * w2), one vector register per iteration. See
// silu.cc for the lane template and sigmoid.cc for the 0.5 * (1 + tanh) mac.
template <int lanes>
static inline void swiglu_impl(bfloat16 *restrict input_vector,
                               bfloat16 *restrict weight_vector_1,
                               bfloat16 *restrict weight_vector_2,
                               bfloat16 *restrict output_vector) {
  const int num_elems = SWIGLU_ELEMS;
  auto it_in = aie::begin_restrict_vector<lanes>((bfloat16 *)input_vector);
  auto it_wt_1 = aie::begin_restrict_vector<lanes>((bfloat16 *)weight_vector_1);
  auto it_wt_2 = aie::begin_restrict_vector<lanes>((bfloat16 *)weight_vector_2);
  auto it_out = aie::begin_restrict_vector<lanes>((bfloat16 *)output_vector);

  aie::vector<bfloat16, 16> register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::vector<bfloat16, lanes> register_0_5_wide =
      aie::broadcast<bfloat16, lanes>(0.5f);
  aie::accum<accfloat, lanes> half;
  half.from_vector(register_0_5_wide);
  AIE_PREPARE_FOR_PIPELINING
  for (int i = 0; i < num_elems; i += lanes) {
    aie::vector<bfloat16, lanes> input = *it_in++;
    aie::vector<bfloat16, lanes> weight_1 = *it_wt_1++;
    aie::vector<bfloat16, lanes> weight_2 = *it_wt_2++;

    aie::vector<bfloat16, lanes> mul_input_weight_1 = aie::mul(input, weight_1);
    aie::vector<bfloat16, lanes> mul_input_weight_2 = aie::mul(input, weight_2);

    aie::vector<bfloat16, lanes> tanh_half_x;
    if constexpr (lanes == 32) {
      auto lo = tanh_bf16_v16(
          aie::mul(mul_input_weight_2.template extract<16>(0), register_0_5));
      auto hi = tanh_bf16_v16(
          aie::mul(mul_input_weight_2.template extract<16>(1), register_0_5));
      tanh_half_x = aie::concat(lo, hi);
    } else {
      tanh_half_x = tanh_bf16_v16(aie::mul(mul_input_weight_2, register_0_5));
    }

    aie::vector<bfloat16, lanes> sigmoid_approx =
        aie::mac(half, tanh_half_x, register_0_5_wide)
            .template to_vector<bfloat16>();
    aie::vector<bfloat16, lanes> silu_output =
        aie::mul(mul_input_weight_2, sigmoid_approx);

    auto mul_output = aie::mul(mul_input_weight_1, silu_output);

    *it_out++ = mul_output.template to_vector<bfloat16>();
  }
}

#if AIE_TUNED_AIE2
// AIE2's table reads are ordered against every load and store, so the next
// trip's inputs load before this trip's lookups. At K = 4 they overflow the
// 1 KiB stack. silu(x * w2) is 0 from x * w2 = -8 down, so the output is 0
// there rather than NaN where x * w1 overflowed.
static inline void swiglu_aie2(const bfloat16 *restrict x,
                               const bfloat16 *restrict w1,
                               const bfloat16 *restrict w2,
                               bfloat16 *restrict out) {
  constexpr int K = 2;
  constexpr int n = SWIGLU_ELEMS;
  constexpr int trips = n / (16 * K);
  static_assert(trips > 0 && n % (16 * K) == 0, "swiglu tiles are 1024");
  using V = aie::vector<bfloat16, 16>;
  V register_0_5 = aie::broadcast<bfloat16, 16>(0.5f);
  aie::accum<accfloat, 16> half;
  half.from_vector(register_0_5);
  auto f = [&](V in, V wt_1, V wt_2) {
    V mul_input_weight_1 = aie::mul(in, wt_1);
    V mul_input_weight_2 = aie::mul(in, wt_2);
    V sigmoid_approx =
        aie::mac(half,
                 tanh_bf16_v16(aie::mul(mul_input_weight_2, register_0_5)),
                 register_0_5)
            .to_vector<bfloat16>();
    V silu_output =
        aie::mul(aie::max(mul_input_weight_2, bfloat16(-8.0f)), sigmoid_approx);
    V y = aie::mul(mul_input_weight_1, silu_output).to_vector<bfloat16>();
    return aie::select(y, aie::zeros<bfloat16, 16>(),
                       aie::eq(silu_output.cast_to<int16_t>(), int16_t(0)));
  };
  auto it_out = aie::begin_restrict_vector<16>(out);
  V nx[K], n1[K], n2[K];
  for (int j = 0; j < K; j++) {
    nx[j] = aie::load_v<16>(x + 16 * j);
    n1[j] = aie::load_v<16>(w1 + 16 * j);
    n2[j] = aie::load_v<16>(w2 + 16 * j);
  }
  for (int i = 0; i < trips; i++) {
    V a[K], b[K], c[K], y[K];
    for (int j = 0; j < K; j++) {
      a[j] = nx[j];
      b[j] = n1[j];
      c[j] = n2[j];
    }
    const int o = (i + 1 < trips ? i + 1 : i) * 16 * K;
    for (int j = 0; j < K; j++) {
      nx[j] = aie::load_v<16>(x + o + 16 * j);
      n1[j] = aie::load_v<16>(w1 + o + 16 * j);
      n2[j] = aie::load_v<16>(w2 + o + 16 * j);
    }
    for (int j = 0; j < K; j++)
      y[j] = f(a[j], b[j], c[j]);
    for (int j = 0; j < K; j++)
      *it_out++ = y[j];
  }
}
#endif

void swiglu_tanh_approx_bf16(bfloat16 *restrict input_vector,
                             bfloat16 *restrict weight_vector_1,
                             bfloat16 *restrict weight_vector_2,
                             bfloat16 *restrict output_vector,
                             const int32_t vector_size) {
  event0();
#if AIE_TUNED_AIE2
  swiglu_aie2(input_vector, weight_vector_1, weight_vector_2, output_vector);
#else
  swiglu_impl<AIE_BF16_LANES>(input_vector, weight_vector_1, weight_vector_2,
                              output_vector);
#endif
  event1();

  return;
}

extern "C" {

void swiglu_bf16(bfloat16 *restrict input, bfloat16 *restrict weights_1,
                 bfloat16 *restrict weights_2, bfloat16 *restrict output) {
  // Assuming input size is a multiple of AIE_BF16_LANES
  int32_t input_size = 1024;
  swiglu_tanh_approx_bf16(input, weights_1, weights_2, output, input_size);
}

} // extern "C"
