//===- exp2f_vec.cc -------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Software f32 2^x for exp-family ops needing more accuracy than the hardware
// `aie::exp2<bfloat16>` LUT. Measured on aie2p against a
// float64 reference, the LUT's max relative error runs 6.1% on [-1, 0] to 49.1%
// on [-100, 0], softmax's range, where this kernel holds 9.2e-6 (7.4e-6 on
// aie2). See programming_examples/basic/vector_exp2f.
//
// 2^x = p(f) * 2^k, f = x - k, with k added straight into the f32 exponent
// field. The field is 8 bits wide, and that sets the hard ends of the domain:
// outside k in [-126, 127] the biased exponent carries into the sign bit and
// the result is finite but wrong-signed (k = 129 reads back as -0.0, k = -129
// as -1.7e38), which isfinite() cannot catch. The add is exact, and the
// accuracy above holds down to -126. The clamp defaults to -111; move it
// with -DEXP2F_VEC_MIN_X=<float>. 2^128 exceeds FLT_MAX, so x >= 128 gets +inf
// under a mask, and [127.999, 128) clamps onto one value (7.8e-4 relative
// error).
//
//===----------------------------------------------------------------------===//
#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/exp2f.h"

extern "C" {

// vector_size must be a multiple of 16.
void exp2f_vec_f32(float *restrict input, float *restrict output,
                   int32_t vector_size) {
  event0();
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  // f's limbs round to bf16 in this mode.
  const auto saved_rounding = aie::get_rounding();
  aie::set_rounding(aie::rounding_mode::conv_even);
  // One vector's dependency chain does not fill the schedule; two do.
  AIE_LOOP_UNROLL(2)
#endif
  for (int i = 0; i < vector_size / EXP2F_VEC_LEN; i++) {
    exp2f_vec(input, output);
    input += EXP2F_VEC_LEN;
    output += EXP2F_VEC_LEN;
  }
  // A 16-element tail runs through a scratch vector, lanes 16-31 unused.
  if (vector_size % EXP2F_VEC_LEN) {
    alignas(aie::vector_decl_align) float tail[EXP2F_VEC_LEN];
    aie::store_v(tail, aie::load_v<EXP2F_VEC_LEN / 2>(input));
    exp2f_vec(tail, tail);
    aie::store_v(output, aie::load_v<EXP2F_VEC_LEN / 2>(tail));
  }
#if AIE_TUNED_AIE2 || AIE_TUNED_AIE2P
  aie::set_rounding(saved_rounding);
#endif

  event1();
}

} // extern "C"
