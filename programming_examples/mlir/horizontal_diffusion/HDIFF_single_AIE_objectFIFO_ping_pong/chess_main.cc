//===- chess_main.cc --------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

#include "hdiff.h"

int main()
{
    int32_t din1[10] = {};
    int32_t din2[10] = {};
    int32_t din3[10] = {};
    int32_t din4[10] = {};
    int32_t din5[10] = {};
    int32_t dout[10] = {};
    vec_hdiff(din1,din2,din3, din4, din5,  dout); 
    return 1;
}
