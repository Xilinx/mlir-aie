//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2024 AMD Inc.

#include <aie_api/aie.hpp>

// Required command-line definitions:
// - DIM_m
// - DIM_n
// compile e.g. as
// xchesscc_wrapper aie2 -I/tools/Xilinx/Vitis/2023.2/aietools/include -DDIM_m=96 -DDIM_n=32 -c ../kernel.cc -o kernel.o

template<typename T_in, typename T_out, int m, int n, int t>
void row_wise_bias_add(const T_in * __restrict in, const T_in * __restrict bias, T_out * __restrict out)
{
    const T_in * __restrict bias_ptr = bias;
    const T_in * __restrict in_base_ptr = in;
    const T_in * __restrict in_ptr = in_base_ptr;
    T_out * __restrict out_base_ptr = out;
    T_out * __restrict out_ptr = out_base_ptr;
    constexpr int n_div_t = n/t;

    for(int j = 0; j < n_div_t; j++) {
        aie::vector<T_in, t> bias = aie::load_v<t>(bias_ptr);
        for(int i = 0; i < m; i += 1) {
            aie::vector<T_in, t> in = aie::load_v<t>(in_ptr);
            aie::store_v(out_ptr, aie::add(in, bias));
            in_ptr += n;
            out_ptr += n;
        }
        bias_ptr += t;
        in_base_ptr += t;
        in_ptr = in_base_ptr;
        out_base_ptr += t;
        out_ptr = out_base_ptr;
    }
}

extern "C" {

void row_wise_bias_add_f32_f32(const float * __restrict in, const float * __restrict bias, float * __restrict out) {
    constexpr int t = 32;
    static_assert(DIM_n % t == 0);
    row_wise_bias_add<float, float, DIM_m, DIM_n, t>(in, bias, out);
}

}