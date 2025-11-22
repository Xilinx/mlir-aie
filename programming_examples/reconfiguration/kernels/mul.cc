// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out> void eltwise_mul(T_in *a, T_in *b, T_out *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

template <typename T_in, typename T_out> void eltwise_vmul(T_in *a, T_in *b, T_out *c, int size)
{

    event0();
    for (int i = 0; i < size; i += 16) {
        auto A = aie::load_v<16>(a + i);
        auto B = aie::load_v<16>(b + i);
        auto C = aie::mul(A, B).template to_vector<T_out>();
        aie::store_v(c + i, C);
    }
    event1();
}

extern "C" {

void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_mul<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}
void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_vmul<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}
} // extern "C"
