// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET

// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

extern "C" {
void vec_hdiff(int32_t *restrict in0, int32_t *restrict in1,
               int32_t *restrict in2, int32_t *restrict in3,
               int32_t *restrict in4, int32_t *restrict out);
}
