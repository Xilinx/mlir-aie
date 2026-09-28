// Copyright (C) 2024-2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../aie_kernel_utils.h"
#include <lut_based_ops.h>

// Store one iteration late: the pipeliner orders the table reads after the
// previous store.
template <const int N>
void exp_bf16_func(bfloat16 *restrict in, bfloat16 *restrict out) {
  auto it_in = aie::cbegin_restrict_vector<16>(in);
  auto it_out = aie::begin_restrict_vector<16>(out);
  aie::vector<bfloat16, 16> prev = to_v16bfloat16(getExpBf16(*it_in++));
  AIE_PREPARE_FOR_PIPELINING
  AIE_LOOP_RANGE(N / 16 - 1, N / 16 - 1)
  for (int i = 1; i < N / 16; i++) {
    aie::vector<bfloat16, 16> cur = to_v16bfloat16(getExpBf16(*it_in++));
    *it_out++ = prev;
    prev = cur;
  }
  *it_out = prev;
}

extern "C" {

void exp_bf16_1024(bfloat16 *a_in, bfloat16 *c_out) {
  event0();
  exp_bf16_func<1024>(a_in, c_out);
  event1();
}

} // extern "C"
