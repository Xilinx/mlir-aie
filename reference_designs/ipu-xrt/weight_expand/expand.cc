//===- scale.cc -------------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2023, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 2
#define NOCPP
#define __AIEARCH__ 20

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#include <aie_api/aie.hpp>

template <typename T_in, typename T_sf, typename T_out, const int N>
void expand(T_in *in, T_out *out) {

  /*
  out[0] = 0x0;
  out[1] = 0x1;
  out[2] = 0x2;
  out[3] = 0x3;
*/
  constexpr int vec_factor = 32;
  constexpr int sf_vec_factor = 8;

  event0();
  T_in *__restrict pI = in;
  T_in *__restrict pSFb =
      in + N / 2; // The scale factors are after the integer values
  T_sf *__restrict pSF =
      (T_sf *)pSFb; // But we only advance by the number of bytes not elements
  T_out *__restrict pO = out;
  const int F = N / (vec_factor * sf_vec_factor);

  for (int i = 0; i < F; i++)
    chess_prepare_for_pipelining chess_loop_range(16, ) {

      // Let's unroll this
      aie::vector<T_sf, sf_vec_factor> sfV =
          aie::load_v<sf_vec_factor>(pSF); // For example <bfloat16, 16>
      pSF += sf_vec_factor;

      for (int k = 0; k < sf_vec_factor; k++)
        chess_unroll_loop(sf_vec_factor) {
          aie::vector<T_in, vec_factor> I0 =
              aie::load_v<vec_factor>(pI); // For example <int4, 32>
          pI += vec_factor / 2;

          bfloat16 sf = sfV[k % sf_vec_factor];

          aie::vector<bfloat16, vec_factor> sf_broadcast = aie::broadcast(sf);

          // Upsize these to 8 bits -> 16 -> bfloat16
          aie::vector<int8, vec_factor> asInt8 = aie::unpack(I0);
          aie::vector<int16, vec_factor> asInt16 = aie::unpack(asInt8);
          aie::vector<bfloat16, vec_factor> as_bf16 =
              aie::to_float<bfloat16>(asInt16, 0);
          aie::vector<bfloat16, vec_factor> scaled_bf16 =
              aie::mul(as_bf16, sf_broadcast);

          aie::store_v(pO, scaled_bf16);
          pO += vec_factor;
        }
    }
  event1();
}

extern "C" {

void expand_int4_to_bfloat16(int4 *a_in, bfloat16 *c_out) {
  expand<int4, bfloat16, bfloat16, 1024>(a_in, c_out);
}

} // extern "C"
