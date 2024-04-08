//===- kernel.cc -------------------------------------------------*- C++
//-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#define __AIENGINE__ 1
#define NOCPP

#define h1 32
#define w1 32
#define w2 32

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void extern_kernel(int32_t *restrict A, int32_t *restrict B,
                   int32_t *restrict acc, int32_t *restrict C) {

  v8int32 *restrict matA = (v8int32 *)A;
  v4int32 *restrict matB = (v4int32 *)B;
  v8int32 *restrict acc_in = (v8int32 *)acc;
  v8int32 *restrict matC = (v8int32 *)C;

  v16int32 buf_matB = undef_v16int32();
  v16int32 buf_matA = undef_v16int32();

  buf_matB = upd_v(buf_matB, 0, *matB);
  matB += w1 / 4;
  buf_matB = upd_v(buf_matB, 1, *matB);
  matB += -(w1 - 4) / 4;

  buf_matA = upd_w(buf_matA, 0, *matA);
  matA += h1 / 8;

  for (unsigned int i = 0; i < 4; i++) {

    for (unsigned int k = 0; k < 16; k++) {
      int jumpA = 0, jumpB = 0, jumpC = 0;
      if (k == 15) {
        jumpC = -(w2 * h1) + h1 + 8;
        jumpB = -(w1 * w2) + 4;
        jumpA = -(w1 * h1) + h1 + 8;
      } else {
        jumpC = h1;
        jumpB = 4;
        jumpA = -(w1 * h1) + h1;
      }
      v8acc80 acc0 = lups(*acc_in, 0);
      acc_in += h1 / 8;
      v8acc80 acc1 = lups(*acc_in, 0);
      acc_in += jumpC / 8;
      for (unsigned int j = 0; j < 3; j++)
        chess_prepare_for_pipelining chess_loop_range(3, ) {
          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 0, 0x0);
          buf_matA = upd_w(buf_matA, 1, *matA);
          matA += h1 / 8;
          buf_matB = upd_v(buf_matB, 2, *matB);
          matB += w1 / 4;
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 4, 0x0);
          buf_matB = upd_v(buf_matB, 3, *matB);
          matB += -(w1 - 4) / 4;

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 1, 0x0);
          buf_matA = upd_w(buf_matA, 0, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 5, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 2, 0x0);
          buf_matA = upd_w(buf_matA, 1, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 6, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 3, 0x0);
          buf_matA = upd_w(buf_matA, 0, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 7, 0x0);

          ////////////////////////////////////////////////////////////////////////
          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 0, 0x0);
          buf_matA = upd_w(buf_matA, 1, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 4, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 1, 0x0);
          buf_matA = upd_w(buf_matA, 0, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 5, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 2, 0x0);
          buf_matA = upd_w(buf_matA, 1, *matA);
          matA += h1 / 8;
          acc1 =
              lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 6, 0x0);

          acc0 =
              lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 3, 0x0);
          buf_matA = upd_w(buf_matA, 0, *matA);
          matA += h1 / 8;
          buf_matB = upd_v(buf_matB, 0, *matB);
          matB += w1 / 4;
          acc1 =
              lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 7, 0x0);
          buf_matB = upd_v(buf_matB, 1, *matB);
          matB += -(w1 - 4) / 4;
        }
      acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 0, 0x0);
      buf_matA = upd_w(buf_matA, 1, *matA);
      matA += h1 / 8;
      buf_matB = upd_v(buf_matB, 2, *matB);
      matB += w1 / 4;
      acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 4, 0x0);
      buf_matB = upd_v(buf_matB, 3, *matB);
      matB += jumpB / 4;

      acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 1, 0x0);
      buf_matA = upd_w(buf_matA, 0, *matA);
      matA += h1 / 8;
      acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 5, 0x0);

      acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 2, 0x0);
      buf_matA = upd_w(buf_matA, 1, *matA);
      matA += h1 / 8;
      acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 0), 6, 0x0);

      acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 3, 0x0);
      buf_matA = upd_w(buf_matA, 0, *matA);
      matA += h1 / 8;
      acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 0), 7, 0x0);

      ////////////////////////////////////////////////////////////////////////
      acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 0, 0x0);
      buf_matA = upd_w(buf_matA, 1, *matA);
      matA += h1 / 8;
      acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 4, 0x0);

      acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 1, 0x0);
      buf_matA = upd_w(buf_matA, 0, *matA);
      matA += h1 / 8;
      acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 5, 0x0);

      acc0 = lmac8(acc0, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 2, 0x0);
      buf_matA = upd_w(buf_matA, 1, *matA);
      matA += jumpA / 8;
      acc1 = lmac8(acc1, buf_matA, 0, 0x76543210, ext_w(buf_matB, 1), 6, 0x0);

      acc0 = lmac8(acc0, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 3, 0x0);
      *matC = srs(acc0, 0);
      matC += h1 / 8;
      buf_matA = upd_w(buf_matA, 0, *matA);
      matA += h1 / 8;
      buf_matB = upd_v(buf_matB, 0, *matB);
      matB += w1 / 4;
      acc1 = lmac8(acc1, buf_matA, 8, 0x76543210, ext_w(buf_matB, 1), 7, 0x0);
      *matC = srs(acc1, 0);
      matC += jumpC / 8;
      buf_matB = upd_v(buf_matB, 1, *matB);
      matB += -(w1 - 4) / 4;
    }
  }
}

} // extern "C"
