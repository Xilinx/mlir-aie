//===- maxpool_2x2.cc -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#ifndef __chess_
#define __PTHREAD_API__
#define __NEW_X86Sim__
#endif

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include <stdio.h>

#include "maxpool_params.h"

inline __attribute__ ((always_inline))
v32uint8 load_u(input_window_uint8* actsIn, const unsigned int mr) {
    v32uint8 v = window_read_v32(actsIn);
    window_incr_v16(actsIn, mr/2);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
v32uint8 load_d(input_window_uint8* actsIn, const unsigned int mr) {
    v32uint8 v = window_read_v32(actsIn);
    window_decr_v16(actsIn, mr/2 - 2);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
v32uint8 load_d_skip(input_window_uint8* actsIn, const unsigned int mr, const unsigned int M) {
    v32uint8 v = window_read_v32(actsIn);

    const unsigned int cond = (((M%4) == 2) || ((M%4) == 3));
    const unsigned int read = 4 * (M / 4) + 2 * cond;
    const unsigned int remaining = MR - read;

    window_decr_v16(actsIn, mr/2 - 2 - remaining / 2 - mr/2);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
v16uint8 load_u_s(input_window_uint8* actsIn, const unsigned int mr) {
    v16uint8 v = window_read_v16(actsIn);
    window_incr_v16(actsIn, mr/2);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
v16uint8 load_d_s(input_window_uint8* actsIn, const unsigned int mr) {
    v16uint8 v = window_read_v16(actsIn);
    window_decr_v16(actsIn, mr/2 - 1);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
v16uint8 load_d_s_skip(input_window_uint8* actsIn, const unsigned int mr, const unsigned int M) {
    v16uint8 v = window_read_v16(actsIn);

    const unsigned int cond = (((M%4) == 2) || ((M%4) == 3)) && ((M%4) != 0);
    const unsigned int read = 4 * (M / 4) + 2 * cond;
    const unsigned int remaining = MR - read;

    window_decr_v16(actsIn, mr/2 - 1 - remaining / 2 - mr/2);
    return v;
    //printf("%p\n", actsIn->ptr - actsIn->head);
}

inline __attribute__ ((always_inline))
void skip_end_row_skip_row(input_window_uint8* actsIn, const unsigned int mr, const unsigned int M) {
    const unsigned int cond = (((M%4) == 2) || ((M%4) == 3)) && ((M%4) != 0);
    const unsigned int read = 4 * (M / 4) + 2 * cond;
    const unsigned int remaining = MR - read;
    window_incr_v16(actsIn, remaining / 2 + mr/2);
}

inline __attribute__ ((always_inline))
void skip_row(input_window_uint8* actsIn, const unsigned int mr) {
    window_incr_v16(actsIn, mr/2);
}

inline __attribute__ ((always_inline))
void set_c(input_window_uint8* actsIn, const unsigned int mr, const unsigned int N,
           const unsigned int C) {
    actsIn->ptr = actsIn->head + C * (N*mr*8);
}

inline __attribute__ ((always_inline))
void store(output_window_uint8* actsOut, v16uint8 v) {
    window_write(actsOut, v);
    window_incr_v16(actsOut, 1);
}

inline __attribute__ ((always_inline))
void maxpool_int8_2x2(input_window_uint8* restrict actsIn,
                      output_window_uint8* restrict actsOut,
                      const unsigned int M,
                      const unsigned int N,
                      const unsigned int C) {

    const unsigned int MP_S = 2;
    const unsigned int MP_W = 2;

    for(unsigned int c = 0; c < C/8; c++) { // chess_unroll_loop(*)
            set_c(actsIn, MR, N, c);
            for(unsigned int h = 0; h < (N-(MP_W-1)); h+=MP_S) { // chess_unroll_loop(*)
                    for(unsigned int w = 0; w < (M-(2*MP_W-1)); w+=(2*MP_S)) chess_unroll_loop(*) {
                            v32uint8 upRow = undef_v32uint8();
                            v32uint8 downRow = undef_v32uint8();
                            upRow = load_u(actsIn, MR);

                            if((w+2*MP_S) < (M-(2*MP_W-1))) {
                                downRow = load_d(actsIn, MR);
                            } else {
                                if(((M % 4) != 0) && (((M % 4) == 2) || ((M % 4) == 3))) {
                                    downRow = load_d(actsIn, MR);
                                } else {
                                    downRow = load_d_skip(actsIn, MR, M);
                                }
                            }

                            v32int16 chess_storage(xa) upRowUnpacked = unpack(upRow);
                            v32int16 chess_storage(xb) downRowUnpacked = unpack(downRow);

                            v32int16 chess_storage(xd) vMax = max32(upRowUnpacked, downRowUnpacked);
                            v32int16 chess_storage(xc) max0 = max32(vMax, 0, 0x0a080200, 0x00000000, 0x3210,0, 0x0e0c0604, 0x00000000, 0x3210);

                            store(actsOut, as_v16uint8(pack(ext_w(max0, 0))));
                        }

                    if((M % 4) != 0) { // still have some stuff to store
                        if(((M % 4) == 2) || ((M % 4) == 3)) { // can do a last MP with two last entries
                            // need to load 128 bits because then overlap with next row
                            v16uint8 upRow = undef_v16uint8();
                            v16uint8 downRow = undef_v16uint8();
                            upRow = load_u_s(actsIn, MR);

                            downRow = load_d_s_skip(actsIn, MR, M);

                            v32int16 chess_storage(xa) upRowUnpacked = concat(unpack(upRow), undef_v16int16());
                            v32int16 chess_storage(xb) downRowUnpacked = concat(unpack(downRow), undef_v16int16());

                            v32int16 chess_storage(xd) vMax = max32(upRowUnpacked, downRowUnpacked);
                            v32int16 chess_storage(xc) max0 = max32(vMax, 0, 0x0a080200, 0x00000000, 0x3210, 0, 0x0e0c0604, 0x00000000, 0x3210);

                            store(actsOut, as_v16uint8(pack(ext_w(max0, 0))));
                        } // otherwise skip last pixel as cannot do maxpool anyway
                    }
                }
        }
}

