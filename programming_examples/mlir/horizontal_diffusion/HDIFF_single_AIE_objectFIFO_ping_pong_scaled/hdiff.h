//===- hdiff.h --------------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

extern "C" {
void vec_hdiff(int32_t *restrict in0, int32_t *restrict in1,
               int32_t *restrict in2, int32_t *restrict in3,
               int32_t *restrict in4, int32_t *restrict out);
}
