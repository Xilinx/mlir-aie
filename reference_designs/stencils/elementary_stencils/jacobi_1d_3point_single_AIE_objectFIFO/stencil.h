// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET   
  
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


extern "C" {
void stencil_1d_3point(int32_t* restrict in,int32_t* restrict out);
void stencil_1d_3point_fp32(float* restrict in, float* restrict out);
}

