//===- softmax_aie2p.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-------------------------------------------------- --------===//

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>
#include <limits>
#include <stdint.h>

#include "../common/exp2_bf16.h"
#include "../common/f32_split.h"

#define SM_VEC_LEN 32   // 32
#define log2e 1.4453125 // 1.44269504089

using namespace aie;

// x * log2(e) in f32. Under EXP2_BF16_ACCURATE log2(e) is three bf16 limbs,
// to 2^-26, summed smallest product first; otherwise it is bf16(log2(e)),
// 1.8e-3 low. A row's maximum is scaled through this too, so its own exponent
// argument is exactly 0.
static inline aie::accum<accfloat, SM_VEC_LEN>
scale_log2e(aie::vector<bfloat16, SM_VEC_LEN> x) {
#ifdef EXP2_BF16_ACCURATE
  auto limb = [](int16_t bits) {
    return aie::broadcast<int16_t, SM_VEC_LEN>(bits).cast_to<bfloat16>();
  };
  aie::accum<accfloat, SM_VEC_LEN> t = aie::mul(x, limb(0x36ec));
  t = aie::mac(t, x, limb(0xbb2c));
  return aie::mac(t, x, limb(0x3fb9));
#else
  return aie::mul(x, aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)log2e));
#endif
}

void softmax_simple_bf16(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  // VJUNG: We do 3 passes on the vector:
  // 1. Find the max value scaled by log2e in the vector
  // 2. Calculate the exponentials of the scaled values minus the maximum
  // 3. Calculate the softmax by dividing each exponential by the sum of all
  // exponentials Note: The multiplication by log2e is very sensitive, casting
  // it to bf16 before exponentiation leads to wrong output.

  auto it_log_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_log_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_in =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)input_vector);
  auto it_exp_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_scale =
      aie::cbegin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);
  auto it_soft_out =
      aie::begin_restrict_vector<SM_VEC_LEN>((bfloat16 *)output_vector);

  aie::vector<bfloat16, SM_VEC_LEN> in_elems, exp_val, input_bf16, max_val_vec;
  aie::accum<accfloat, SM_VEC_LEN> out_vals, exp_val_accum, exp_in_accum;

  float accum_exp_val = 0;
  bfloat16 col_sum_inv;
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  exp_val_accum = aie::zeros<accfloat, SM_VEC_LEN>();

  // First pass - Optimized: element-wise max + single final reduce_max
  // Use vector max accumulation, then reduce once at the end.
  //
  // Take the max of the raw input, not of the scaled value. Scaling by log2e
  // is monotonic, so the largest element is the same either way, and a bf16
  // max over bf16 inputs is exact -- rounding the scaled value to bf16 first
  // was not. When that rounding landed below the true maximum, the largest
  // element's exponent argument came out positive instead of zero and exp2
  // ran away with it.
  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(
          std::numeric_limits<bfloat16>::lowest());
  AIE_LOOP_UNROLL(2)
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_log_in++);
  }
  max_val_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(aie::reduce_max(max_accum_vec));
  // Scale the maximum through the same multiply its element takes below, so
  // the subtraction cancels exactly there. That is what the whole
  // formulation rests on: the largest exponent argument is exactly 0 and
  // every other one is negative, so exp2 <= 1, the sum is >= 1, and the
  // reciprocal of the sum cannot overflow.
  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(max_val_vec).to_vector<float>();

  // Second pass
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {

    input_bf16 = *it_exp_in++;

    exp_in_accum = aie::sub(scale_log2e(input_bf16), max_scaled);
    exp_val = exp2_bf16(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);

    *it_exp_out++ = exp_val;
  }

  // Final reduction after loop
  aie::vector<float, SM_VEC_LEN> reduce = exp_val_accum.to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);
  col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int c = 0; c < elem_iters; c++) {
    in_elems = *it_scale++;
    out_vals = aie::mul(in_elems, col_sum_inv);
    *it_soft_out++ = out_vals.to_vector<bfloat16>();
  }

  ::aie::set_rounding(saved_rounding);

  return;
}

// Online (flash-attention) partial softmax over one key-block row.  Unlike
// softmax_simple_bf16 this does NOT normalize: it applies the running-max
// rescale used by streaming attention and stashes the block's new max and
// exp-sum into scale_buffer so the caller (see ../linalg/mha.cc) can combine
// blocks.  scale_buffer layout, indexed by row: [0*num_rows + r] = prev max
// m_{i-1}; [1*num_rows + r] = new max m_i (written here), both unscaled
// scores; [3*num_rows + r] = this block's exp-sum l_i (written here).
// `scale` is the positive log2-domain factor (1/sqrt(d) folded with log2e)
// broadcast in place of the plain log2e.  The row is read W lanes at a time,
// so vector_size is a multiple of W.
template <unsigned W = SM_VEC_LEN>
void partial_softmax_alias_bf16(bfloat16 *restrict input_vector,
                                bfloat16 *restrict output_vector,
                                float *restrict scale_buffer,
                                const int32_t vector_size,
                                const int32_t row_idx, const int32_t num_rows,
                                const bfloat16 scale) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_log_in = aie::cbegin_restrict_vector<W>((bfloat16 *)input_vector);
  auto it_exp_in = aie::cbegin_restrict_vector<W>((bfloat16 *)input_vector);
  auto it_exp_out = aie::begin_restrict_vector<W>((bfloat16 *)output_vector);

  aie::vector<bfloat16, W> in_elems, exp_val, input_bf16, log2e_vec,
      max_val_vec;
  aie::accum<accfloat, W> out_vals, exp_val_accum, scaled_accum, exp_in_accum;

  float accum_exp_val = 0;
  const int elem_iters = (uint32_t)vector_size / W;

  exp_val_accum = aie::zeros<accfloat, W>();

  log2e_vec = aie::broadcast<bfloat16, W>((bfloat16)scale);

  // First pass - the block's largest score, reduced once at the end: a
  // scalar float compare is a libcall. It stays unscaled; see
  // partial_softmax_rows in ../linalg/mha.cc.
  aie::vector<bfloat16, W> max_accum_vec =
      aie::broadcast<bfloat16, W>(std::numeric_limits<bfloat16>::lowest());
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_log_in++);
  }
  bfloat16 max_val = aie::reduce_max(max_accum_vec);

  // Compute m_{i}: max of this block and the carried-in running max.
  max_val = aie::max(max_val, (bfloat16)scale_buffer[row_idx]);
  scale_buffer[num_rows + row_idx] = (float)max_val;

  max_val_vec = aie::broadcast<bfloat16, W>(max_val);

  // Second pass - unnormalized exponentials, accumulating the block sum.
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    input_bf16 = *it_exp_in++;
    scaled_accum = aie::mul(input_bf16, log2e_vec);
    exp_in_accum = aie::msc(scaled_accum, max_val_vec, log2e_vec);
    exp_val = exp2_nonpos_bf16(exp_in_accum.template to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
    *it_exp_out++ = exp_val;
  }

  aie::vector<float, W> reduce = exp_val_accum.template to_vector<float>();
  accum_exp_val = aie::reduce_add(reduce);

  scale_buffer[3 * num_rows + row_idx] = accum_exp_val;

  ::aie::set_rounding(saved_rounding);

  return;
}

// One row of float32 scores: the max and A - m in float32, then (A - m) * s
// as three bf16 limbs, so the largest argument is exactly 0.
template <unsigned W>
static inline void
partial_softmax_row_f32(const float *__restrict A, bfloat16 *__restrict P,
                        float *__restrict scale_buffer, int32_t n, int32_t row,
                        int32_t num_rows, bfloat16 scale) {
  aie::vector<float, W> top =
      aie::broadcast<float, W>((float)std::numeric_limits<bfloat16>::lowest());
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int32_t c = 0; c < n; c += W)
    top = aie::max(top, aie::load_v<W>(A + c));
  const float m = aie::max(aie::broadcast<float, 16>(aie::reduce_max(top)),
                           aie::broadcast<float, 16>(scale_buffer[row]))
                      .get(0);
  scale_buffer[num_rows + row] = m;

  const auto s = aie::broadcast<bfloat16, W>(scale);
  const auto m_vec = aie::broadcast<float, W>(m);
  aie::accum<accfloat, W> sum = aie::zeros<accfloat, W>();
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int32_t c = 0; c < n; c += W) {
    aie::accum<accfloat, W> d;
    d.from_vector(aie::load_v<W>(A + c));
    const Limbs<W> dl = bf16_limbs(aie::sub(d, m_vec));
    aie::accum<accfloat, W> x = aie::mul(dl.hi, s);
    x = aie::mac(x, dl.mid, s);
    x = aie::mac(x, dl.lo, s);
    const aie::vector<bfloat16, W> p =
        exp2_nonpos_bf16(x.template to_vector<float>());
    sum = aie::add(sum, p);
    aie::store_v(P + c, p);
  }
  scale_buffer[3 * num_rows + row] =
      aie::reduce_add(sum.template to_vector<float>());
}

// c * tanh(s / c) * log2(e) for f32 s, as k0 (2r - 1) with k0 = c log2(e),
// r = 1 / (1 + 2^a) and a = -|s| * 2 log2(e) / c <= 0; 2^a flushing to 0
// saturates it at k0. a_limbs is a's factor 2 log2(e) / c (times any scale s
// takes first), k1_limbs 2 k0.
static inline __attribute__((always_inline)) aie::accum<accfloat, SM_VEC_LEN>
soft_cap_log2(aie::accum<accfloat, SM_VEC_LEN> s,
              const Limbs<SM_VEC_LEN> &a_limbs, float k0,
              const Limbs<SM_VEC_LEN> &k1_limbs) {
  using Acc = aie::accum<accfloat, SM_VEC_LEN>;
  auto splat = [](float c) {
    return Acc(aie::broadcast<float, SM_VEC_LEN>(c));
  };
  auto limb = [](int16_t bits) {
    return aie::broadcast<int16_t, SM_VEC_LEN>(bits).cast_to<bfloat16>();
  };
  // The sign by the bits' compare and -s in the accumulator: Peano cannot
  // legalize a 32-lane f32 fneg or 1024-bit OR.
  const auto sv = s.to_vector<float>();
  const auto negative =
      aie::lt(sv.cast_to<int32_t>(), aie::zeros<int32_t, SM_VEC_LEN>());
  const auto neg = aie::sub(aie::zeros<accfloat, SM_VEC_LEN>(), sv);
  const Acc neg_abs(aie::select(neg.to_vector<float>(), sv, negative));
  const aie::vector<float, SM_VEC_LEN> a =
      mac_limbs(aie::zeros<accfloat, SM_VEC_LEN>(), bf16_limbs(neg_abs),
                a_limbs)
          .to_vector<float>();
  const Exp2F32 e2a = exp2_f32_32(a);
  const Acc d = aie::add(
      splat(1.0f), aie::select(e2a.bits.cast_to<float>(),
                               aie::zeros<float, SM_VEC_LEN>(), e2a.below));

  // r = 1 / d, d in [1, 2]: a quadratic within 2%, then two Newton steps
  // r -= r (d r - 1), the first from r in bf16.
  const Limbs<SM_VEC_LEN> dl = bf16_limbs(d);
  const auto q =
      aie::mac(splat(-1.4515766f), dl.hi, limb(0x3ea5)).to_vector<bfloat16>();
  const auto r0 = aie::mac(splat(2.1190739f), q, dl.hi).to_vector<bfloat16>();
  Limbs<SM_VEC_LEN> e = bf16_limbs(aie::mac(
      aie::mac(aie::mac(splat(-1.0f), dl.hi, r0), dl.mid, r0), dl.lo, r0));
  const Acc r1 = aie::msc(aie::msc(Acc(r0), r0, e.hi), r0, e.mid);
  const Limbs<SM_VEC_LEN> rl = bf16_limbs(r1);
  e = bf16_limbs(mac_limbs(splat(-1.0f), dl, rl));
  const Acc r =
      aie::msc(aie::msc(aie::msc(r1, rl.hi, e.hi), rl.hi, e.mid), rl.mid, e.hi);

  const auto t = mac_limbs(splat(-k0), bf16_limbs(r), k1_limbs);
  return Acc(aie::select(
      t.to_vector<float>(),
      aie::sub(aie::zeros<accfloat, SM_VEC_LEN>(), t.to_vector<float>())
          .to_vector<float>(),
      negative));
}

#ifdef SOFTMAX_F32_INPUT
#ifndef EXP2_BF16_ACCURATE
#error "SOFTMAX_F32_INPUT takes 2^x from the accurate exp2: EXP2_BF16_ACCURATE"
#endif
#ifndef SOFTMAX_SOFT_CAP
#define SOFTMAX_SOFT_CAP 0.0f
#endif

// softmax(cap * tanh((x + addend) / cap) + mask) of one float32 row into
// bf16, with the scaled logits written over x. A mask of lowest or -inf drops
// its element; x + addend must be finite.
static void softmax_f32_row(float *__restrict x, const float *__restrict addend,
                            const bfloat16 *__restrict mask,
                            bfloat16 *__restrict y, const int32_t n) {
  using Acc = aie::accum<accfloat, SM_VEC_LEN>;
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);
  auto splat = [](float c) {
    return Acc(aie::broadcast<float, SM_VEC_LEN>(c));
  };
  auto limb = [](int16_t bits) {
    return aie::broadcast<int16_t, SM_VEC_LEN>(bits).cast_to<bfloat16>();
  };
  auto bits = [](uint32_t b) {
    return aie::broadcast<uint32_t, SM_VEC_LEN>(b);
  };
  const Limbs<SM_VEC_LEN> log2e_limbs{limb(0x3fb9), limb(0xbb2c), limb(0x36ec)};
  constexpr float cap = SOFTMAX_SOFT_CAP;
  constexpr float k0 = cap * 1.44269504088896341f;
  const Limbs<SM_VEC_LEN> a_limbs =
      bf16_limbs(splat(cap > 0.0f ? 2.88539008177792682f / cap : 0.0f));
  const Limbs<SM_VEC_LEN> k1_limbs = bf16_limbs(splat(2.0f * k0));

  // Uncapped, logits are centred on the row's largest before scaling, so no
  // finite input overflows log2(e) or a difference.
  aie::vector<float, SM_VEC_LEN> centre = aie::zeros<float, SM_VEC_LEN>();
  if constexpr (!(cap > 0.0f)) {
    aie::vector<float, SM_VEC_LEN> top =
        aie::broadcast<float, SM_VEC_LEN>(std::numeric_limits<float>::lowest());
    AIE_LOOP_MIN_ITERATION_COUNT(1)
    for (int32_t c = 0; c < n; c += SM_VEC_LEN) {
      Acc s(aie::load_v<SM_VEC_LEN>(x + c));
      if (addend)
        s = aie::add(s, aie::load_v<SM_VEC_LEN>(addend + c));
      if (mask)
        s = aie::add(s, aie::load_v<SM_VEC_LEN>(mask + c));
      top = aie::max(top, s.to_vector<float>());
    }
    centre = aie::broadcast<float, SM_VEC_LEN>(aie::reduce_max(top));
  }
  const auto neg_big = aie::broadcast<float, SM_VEC_LEN>(-0x1p126f);

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int32_t c = 0; c < n; c += SM_VEC_LEN) {
    Acc s(aie::load_v<SM_VEC_LEN>(x + c));
    if (addend)
      s = aie::add(s, aie::load_v<SM_VEC_LEN>(addend + c));
    Acc z;
    if constexpr (cap > 0.0f) {
      z = soft_cap_log2(s, a_limbs, k0, k1_limbs);
      if (mask) {
        const auto m = aie::load_v<SM_VEC_LEN>(mask + c);
        z = aie::mac(z, m, log2e_limbs.hi);
        z = aie::mac(z, m, log2e_limbs.mid);
        z = aie::mac(z, m, log2e_limbs.lo);
      }
    } else {
      // The same s + mask as the pre-pass, so t <= 0; -inf where it
      // overflows is clamped to -2^126 by the bits.
      if (mask)
        s = aie::add(s, aie::load_v<SM_VEC_LEN>(mask + c));
      auto t = aie::sub(s, centre).to_vector<float>();
      t = aie::select(
          t, neg_big,
          aie::gt(t.cast_to<uint32_t>(), neg_big.cast_to<uint32_t>()));
      z = mac_limbs(aie::zeros<accfloat, SM_VEC_LEN>(), bf16_limbs(Acc(t)),
                    log2e_limbs);
    }
    // lowest * log2(e) is -inf, which the limbs would turn into NaN: clamp to
    // lowest, whose 2^x is 0.
    const auto z_bits = z.to_vector<float>().cast_to<uint32_t>();
    aie::store_v(x + c, aie::select(z_bits, bits(0xff7f0000u),
                                    aie::gt(z_bits, bits(0xff7f0000u)))
                            .cast_to<float>());
  }

  alignas(aie::vector_decl_align) float state[4] = {
      (float)std::numeric_limits<bfloat16>::lowest(), 0.0f, 0.0f, 0.0f};
  partial_softmax_row_f32<SM_VEC_LEN>(x, y, state, n, 0, 1, (bfloat16)1.0f);

  const bfloat16 sum_inv = (bfloat16)aie::inv(state[3]);
  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(y);
  auto it_out = aie::begin_restrict_vector<SM_VEC_LEN>(y);
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int32_t c = 0; c < n; c += SM_VEC_LEN)
    *it_out++ = aie::mul(*it_in++, sum_inv).to_vector<bfloat16>();

  ::aie::set_rounding(saved_rounding);
}
#endif

// The three passes of softmax_simple_bf16 over a row split into blocks, one
// call per block and pass. Between calls state[0, SM_VEC_LEN) holds the
// per-lane exp sums and state[SM_VEC_LEN] the row maximum, so every lane sums
// its elements in the same order and the row rounds exactly as one call
// would.
void softmax_max_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t first, const int32_t vector_size) {
  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<bfloat16, SM_VEC_LEN> max_accum_vec =
      aie::broadcast<bfloat16, SM_VEC_LEN>(
          first ? std::numeric_limits<bfloat16>::lowest()
                : (bfloat16)state[SM_VEC_LEN]);
  if (first)
    aie::store_v(state, aie::zeros<float, SM_VEC_LEN>());

  AIE_LOOP_UNROLL(2)
  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    max_accum_vec = aie::max(max_accum_vec, *it_in++);
  }
  state[SM_VEC_LEN] = (float)aie::reduce_max(max_accum_vec);
}

void softmax_sum_block(bfloat16 *restrict input_vector, float *restrict state,
                       const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(
          aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)state[SM_VEC_LEN]))
          .to_vector<float>();

  aie::accum<accfloat, SM_VEC_LEN> exp_val_accum;
  exp_val_accum.from_vector(aie::load_v<SM_VEC_LEN>(state));

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    aie::accum<accfloat, SM_VEC_LEN> exp_in_accum =
        aie::sub(scale_log2e(*it_in++), max_scaled);
    aie::vector<bfloat16, SM_VEC_LEN> exp_val =
        exp2_bf16(exp_in_accum.to_vector<float>());
    exp_val_accum = add(exp_val_accum, exp_val);
  }
  aie::store_v(state, exp_val_accum.to_vector<float>());

  ::aie::set_rounding(saved_rounding);
}

void softmax_scale_block(bfloat16 *restrict input_vector,
                         bfloat16 *restrict output_vector,
                         float *restrict state, const int32_t vector_size) {
  ::aie::rounding_mode saved_rounding =
      ::aie::swap_rounding(aie::rounding_mode::conv_even);

  auto it_in = aie::cbegin_restrict_vector<SM_VEC_LEN>(input_vector);
  auto it_out = aie::begin_restrict_vector<SM_VEC_LEN>(output_vector);
  const int elem_iters = (uint32_t)vector_size / SM_VEC_LEN;

  aie::vector<float, SM_VEC_LEN> max_scaled =
      scale_log2e(
          aie::broadcast<bfloat16, SM_VEC_LEN>((bfloat16)state[SM_VEC_LEN]))
          .to_vector<float>();

  aie::vector<float, SM_VEC_LEN> reduce = aie::load_v<SM_VEC_LEN>(state);
  float accum_exp_val = aie::reduce_add(reduce);
  bfloat16 col_sum_inv = (bfloat16)aie::inv(accum_exp_val);

  AIE_LOOP_MIN_ITERATION_COUNT(1)
  for (int i = 0; i < elem_iters; i++) {
    aie::accum<accfloat, SM_VEC_LEN> exp_in_accum =
        aie::sub(scale_log2e(*it_in++), max_scaled);
    aie::vector<bfloat16, SM_VEC_LEN> exp_val =
        exp2_bf16(exp_in_accum.to_vector<float>());
    *it_out++ = aie::mul(exp_val, col_sum_inv).to_vector<bfloat16>();
  }

  ::aie::set_rounding(saved_rounding);
}

extern "C" {

void softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                  const int32_t input_size) {
  event0();
  softmax_simple_bf16(input, output, input_size);
  event1();
}

void partial_softmax_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                          float *restrict scale_buffer,
                          const int32_t input_size, const int32_t row_idx,
                          const int32_t num_rows, const bfloat16 scale) {
  // Not in partial_softmax_alias_bf16, so mha.cc can time a whole block.
  event0();
  partial_softmax_alias_bf16(input, output, scale_buffer, input_size, row_idx,
                             num_rows, scale);
  event1();
}

void softmax_max_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t first, const int32_t input_size) {
  event0();
  softmax_max_block(input, state, first, input_size);
  event1();
}

void softmax_sum_bf16(bfloat16 *restrict input, float *restrict state,
                      const int32_t input_size) {
  event0();
  softmax_sum_block(input, state, input_size);
  event1();
}

void softmax_scale_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                        float *restrict state, const int32_t input_size) {
  event0();
  softmax_scale_block(input, output, state, input_size);
  event1();
}

// Fill [unmasked_size, total_size) with -inf so a subsequent softmax drops the
// masked tail (causal / padding mask).
void mask_bf16(bfloat16 *inout, const int32_t unmasked_size,
               const int32_t total_size) {
  for (int32_t i = unmasked_size; i < total_size; i++) {
    inout[i] = std::numeric_limits<bfloat16>::lowest();
  }
}

// softmax_bf16 over `rows` rows of `row_len` each.
void softmax_rows_bf16(bfloat16 *restrict input, bfloat16 *restrict output,
                       const int32_t rows, const int32_t row_len) {
  event0();
  for (int32_t r = 0; r < rows; r++) {
    softmax_simple_bf16(input + r * row_len, output + r * row_len, row_len);
  }
  event1();
}

// softmax_rows_bf16 under a causal mask, the first row being query row_offset.
void softmax_rows_causal_bf16(bfloat16 *restrict input,
                              bfloat16 *restrict output, const int32_t rows,
                              const int32_t row_len, const int32_t row_offset) {
  event0();
  for (int32_t r = 0; r < rows; r++) {
    mask_bf16(input + r * row_len, row_offset + r + 1, row_len);
    softmax_simple_bf16(input + r * row_len, output + r * row_len, row_len);
  }
  event1();
}

#ifdef SOFTMAX_F32_INPUT
// The row is input_size float32 logits, a multiple of SM_VEC_LEN; input is
// overwritten.
void softmax_f32_bf16(float *restrict input,
#ifdef SOFTMAX_ADDEND
                      float *restrict addend,
#endif
#ifdef SOFTMAX_MASKED
                      bfloat16 *restrict mask,
#endif
                      bfloat16 *restrict output, const int32_t input_size) {
#ifndef SOFTMAX_ADDEND
  constexpr float *addend = nullptr;
#endif
#ifndef SOFTMAX_MASKED
  constexpr bfloat16 *mask = nullptr;
#endif
  event0();
  softmax_f32_row(input, addend, mask, output, input_size);
  event1();
}

// Query row q's valid columns of a `window`-wide band, gathered the way
// EG2's audio attention does: `j >= max(q % 2, window - 1 - q + q % 2)`
// and `j < window + q % 2` (an odd query's band starts a column later).
// `row_len` is at most SM_VEC_LEN: one vector, no block loop.
static inline void band_mask_row(bfloat16 *restrict band, int32_t q,
                                 int32_t window, int32_t row_len) {
  const int32_t parity = q & 1;
  int32_t lo = window - 1 - q + parity;
  if (lo < parity)
    lo = parity;
  const int32_t hi = window + parity;
  for (int32_t j = 0; j < row_len; j++)
    band[j] = (j >= lo && j < hi) ? (bfloat16)0.0f
                                  : std::numeric_limits<bfloat16>::lowest();
}

// softmax_f32_row over `rows` rows, each row's mask its own band (above),
// not read from memory: query row `(row_offset + r) % period` (`period`
// wraps the band over each of several repeats of the same `period` rows,
// as EG2 tiles one band over every attention head).
void softmax_rows_banded_f32_bf16(float *restrict input, float *restrict addend,
                                  bfloat16 *restrict output, const int32_t rows,
                                  const int32_t row_len, const int32_t window,
                                  const int32_t period,
                                  const int32_t row_offset) {
  event0();
  alignas(aie::vector_decl_align) bfloat16 band[SM_VEC_LEN];
  for (int32_t r = 0; r < rows; r++) {
    const int32_t q = (row_offset + r) % period;
    band_mask_row(band, q, window, row_len);
    softmax_f32_row(input + r * row_len, addend + r * row_len, band,
                    output + r * row_len, row_len);
  }
  event1();
}
#endif

} // extern "C"
