//===- transposed_conv2d.cc -----------------------------------000---*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2024, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>
#include <stdio.h>
#include <cassert>
#include "transposed_conv2d_params.h"

namespace F2 {

#define X_C64_D8 0x38303830
#define Z_C8_D8_I_L 0x44440000
#define Z_C8_D8_I_U 0x88884444
#define Z_C8_D8_I_UU 0xcccc8888

    // TODO refactor and simplify DUP things here

    inline __attribute__ ((always_inline))
    v32int8 load_w(input_window_int8* weightsIn) {
        return window_readincr_v32(weightsIn);
        //printf("Reading at: %p\n", (weightsIn->ptr - weightsIn->head));
    }

    inline __attribute__ ((always_inline))
    void unload_2_w(input_window_int8* weightsIn) {
        window_decr_v32(weightsIn, 2);
    }

    inline __attribute__ ((always_inline))
    v32int8 load_a_dup(input_window_int8* actsIn, v32int8 v, const unsigned int i) {
        v = upd_v(v, i, window_read_v16(actsIn));
        window_incr_v16(actsIn, chess_copy(0));
        //printf("Loading at: %p\n", (actsIn->ptr - actsIn->head));
        return v;
    }

    inline __attribute__ ((always_inline))
    v32int8 load_a(input_window_int8* actsIn, v32int8 v, const unsigned int i) {
        //printf("Loading at: %p\n", (actsIn->ptr - actsIn->head));
        return upd_v(v, i, window_read_v16(actsIn));
    }

    inline __attribute__ ((always_inline))
    v16int8 load_a_simple(input_window_int8* actsIn) {
        return window_read_v16(actsIn);
        //printf("Loading at: %p\n", (actsIn->ptr - actsIn->head));
    }

    inline __attribute__ ((always_inline))
    v32int8 load_inc_a(input_window_int8* actsIn, v32int8 v, const unsigned int i) {
        //printf("Loading at: %p\n", (actsIn->ptr - actsIn->head));
        return upd_v(v, i, window_readincr_v16(actsIn));
        //window_incr_v16(actsIn, 1);
    }

    inline __attribute__ ((always_inline))
    v16int8 load_inc_a_simple(input_window_int8* actsIn) {
        return window_readincr_v16(actsIn);
        //printf("Loading at: %p\n", (actsIn->ptr - actsIn->head));
    }

    inline __attribute__ ((always_inline))
    void skip_end_a(input_window_int8* actsIn, const unsigned int mr, const unsigned int outW) {
        const unsigned int read = 4*((outW+3)/4);
        //const unsigned int read = 4*((mr)/4);
        const unsigned int remaining = ((mr - read) / 2) * 16;
        //printf("%d, %d, %d, %p, %p\n", remaining, mr, read, (actsIn->ptr - actsIn->head)/8, mr*8);
        actsIn->ptr = actsIn->ptr + chess_copy(remaining);
        //printf("%d, %d, %d, %p\n", remaining, mr, read, (actsIn->ptr - actsIn->head)/8 % mr);
        //printf("================\n");
    }

    inline __attribute__ ((always_inline))
    void seek_next_row(input_window_int8* actsIn, const unsigned int mr, const unsigned int F,
                       const unsigned int doit) {
        window_decr_v16(actsIn, (1+F/2 - (mr/2)) * doit);
    }

    inline __attribute__ ((always_inline))
    void seek_next_channels(input_window_int8* actsIn, const unsigned int mr,
                            const unsigned int F, const unsigned int N,
                            const unsigned int doit) {
        window_decr_v16(actsIn, (1+F/2 - (mr/2)*(N-(F-1))) * doit);
    }

    inline __attribute__ ((always_inline))
    void rewind_a(input_window_int8* actsIn, const unsigned int mr, const unsigned int cinUp,
                  const unsigned int F, const unsigned int N, const unsigned int S) {
        window_decr_v16(actsIn, 1+F/2 + (cinUp/8-1) * mr/2*N + (F-1)*mr/2 - 2);
    }

    inline __attribute__((always_inline))
    void next_activations(input_window_int8* actsIn) {
        window_incr_v16(actsIn, 2);
    }

    inline __attribute__ ((always_inline))
    void reset_a(input_window_int8* actsIn) {
        actsIn->ptr = actsIn->head;
    }

    inline __attribute__ ((always_inline))
    void rewind_w(input_window_int8* weightsIn, const unsigned int F,
                  const unsigned int cinUp, const unsigned int doit) {
        window_decr_v32(weightsIn, 2*(cinUp/8) * doit);
    }

    inline __attribute__ ((always_inline))
    void seek_next_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        window_incr_v32(weightsIn, 2*(cinUp/8)); // TODO generalize ot other F length
    }

    // WB acc0
    inline __attribute__ ((always_inline))
    void wb_0(v16uint8* restrict* wrPtr, v16acc48 acc, const unsigned int erase_acc1,
              const unsigned int outW) {
        v16uint8 out = ubsrs(acc, SHIFT);
        //printf("writing at: %p\n", ((actsOut->ptr - actsOut->head)/8) / outW);
        **wrPtr = out;
        (*wrPtr) = (*wrPtr) + ((1 + 1 * erase_acc1) * 2);
        //window_write(actsOut, out);
        //window_incr_v16(actsOut, (1 + 1 * erase_acc1) * 2);
    }

    // WB acc1
    inline __attribute__ ((always_inline))
    void wb_1(v16uint8* restrict* wrPtr, v16acc48 acc, const unsigned int erase_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        (*wrPtr) = (*wrPtr) + ((1 * erase_acc1) * 2);
        **wrPtr = out;
        (*wrPtr) = (*wrPtr) - ((1 * erase_acc1) * 2);
        //window_incr_v16(actsOut, (1 * erase_acc1) * 2);
        //window_write(actsOut, out);
        //window_decr_v16(actsOut, (1 * erase_acc1) * 2);
    }

    inline __attribute__((always_inline))
    void applyKernel(input_window_int8* restrict actsIn,
                     input_window_int8* restrict weightsIn,
                     v16uint8* restrict* wrPtr,
                     const unsigned int M,
                     const unsigned int N,
                     const unsigned int Cin,
                     const unsigned int Cout,
                     const unsigned int F,
                     const unsigned int S,
                     const unsigned int CASC_IN_EN,
                     const unsigned int CASC_OUT_EN,
                     const unsigned int store_acc1) {

        // TODO propagate this change to conv2d
        const unsigned int N_WRITES = ((F+1) / 2) * (CinUp/8);
        const unsigned int W_N_WRITES = F*F*(CinUp/8);

        v16acc48 acc0 = undef_v16acc48();
        v16acc48 acc1 = undef_v16acc48();

        if(CASC_IN_EN == 1) {
            acc0 = concat(get_scd(), get_scd());
            acc1 = concat(get_scd(), get_scd());
        }

        // ensure back to back weight load on XA / XD
        v32int8 chess_storage(wr0) w0 = undef_v32int8();
        v32int8 chess_storage(wr1) w1 = undef_v32int8();
        v32int8 chess_storage(wd0) w2 = undef_v32int8();
        v32int8 chess_storage(wd1) w3 = undef_v32int8();

        v64int8 weightBuff0 = undef_v64int8();

        // Need to specify storage to avoid two acts load in a cycle
        v32int8 chess_storage(wc0) v0 = undef_v32int8();
        v32int8 chess_storage(wc1) v1 = undef_v32int8();

        // If CIn%8 != 0 then last iteration has 0 activations
#pragma unroll
        for(unsigned int c = 0; c < CinUp/8; c++) {
#pragma unroll
            for(unsigned int y = 0; y < F; y++) { // F == 1 for now
#pragma unroll
                for(unsigned int x = 0; x < F; x++) { // F == 1 for now
                    const unsigned int pat = x % 2; // access pattern for pat
                    const unsigned int wPat = (x + y*F + c*F*F) % 2; // global weight access

                    v32int8 l0 = load_w(weightsIn);
                    v32int8 l1 = load_w(weightsIn);
                    v64int8 l;

                    if((W_N_WRITES % 2) == 0) {
                        if(wPat == 0) {
                            w0 = l0;
                            w1 = l1;
                            l = concat(w0, w1);
                        } else {
                            w2 = l0;
                            w3 = l1;
                            l = concat(w2, w3);
                        }
                    } else {
                        if(wPat == 0) {
                            if((x+y*F+c*F*F) == 0) {
                                w2 = l0;
                                w3 = l1;

                                v64int8 chess_storage(xa) temp1 = concat(w2, w3);
                                l = temp1;
                            } else {
                                w0 = l0;
                                w1 = l1;
                                l = concat(w0, w1);
                            }
                        } else {
                            w2 = l0;
                            w3 = l1;
                            l = concat(w2, w3);
                        }
                    }

                    weightBuff0 = l;

                    if((x == 0) || (pat == 1)) {
                        if((N_WRITES % 2) == 0) { // even number of read, register period of 1
                            const unsigned int regWrite = (((x+1)/2) + c%2) % 2;
                            if(regWrite == 0) {
                                v0 = load_inc_a(actsIn, v0, 0);
                                v0 = load_a(actsIn, v0, 1);
                            } else {
                                v1 = load_inc_a(actsIn, v1, 0);
                                v1 = load_a(actsIn, v1, 1);
                            }
                        } else { // odd number of read, register period of 2
                            const unsigned int regWrite = (((x+1)/2) + (y%2) + (c%2)) % 2;
                            if((x == 0) && (y == 0) && (c == 0)) {
                                v1 = load_inc_a(actsIn, v1, 0);
                                v1 = load_a(actsIn, v1, 1);
                                v0 = v1;
                            } else {
                                if(regWrite == 0) {
                                    v0 = load_inc_a(actsIn, v0, 0);
                                    v0 = load_a(actsIn, v0, 1);
                                } else {
                                    v1 = load_inc_a(actsIn, v1, 0);
                                    v1 = load_a(actsIn, v1, 1);
                                }
                            }
                        }
                    }

                    // Which act register to use for the first MAC
                    unsigned int regUp = ((N_WRITES % 2) == 0) ? ((x/2 + c%2) % 2) : ((x/2 + y%2 + c%2) % 2);
                    //printf("%d\n", regUp);
                    v32int8 actBuff = (regUp == 0) ? v0 : v1;
                    //chess_report(actBuff);
                    unsigned int zoff = (pat == 0) ? Z_C8_D8_I_L : Z_C8_D8_I_U;

                    if((x == 0 && y == 0 && c == 0) && CASC_IN_EN == 0) {
                        acc0 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    } else {
                        acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    }

                    // decide act register for second mac
                    unsigned int regLow = ((N_WRITES % 2) == 0) ? ((x+1)/2 + c%2) % 2 : ((x+1)/2 + y%2 + c%2) % 2;
                    //printf("%d\n", regLow);
                    actBuff = (regLow == 0) ? v0 : v1;
                    zoff = (pat == 0) ? Z_C8_D8_I_UU : Z_C8_D8_I_U;
                    if((x == 0 && y == 0 && c == 0) && CASC_IN_EN == 0) {
                        acc1 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    } else {
                        acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    }
                }

                // Only 2x2 supported
                //const unsigned int doit = y < (F-1);
                //seek_next_row(actsIn, MR, F, doit);// prefetch next activations
            }
            const unsigned int doit = c < (Cin/8 - 1);
            seek_next_channels(actsIn, MR, F, N, doit); // prefetch next channels
        }

        // WB computed values
        if(CASC_OUT_EN == 0) {
            wb_1(wrPtr, acc1, store_acc1);
            wb_0(wrPtr, acc0, store_acc1, outWidth);
        } else {
            put_mcd(ext_lo(acc0));
            put_mcd(ext_hi(acc0));

            put_mcd(ext_lo(acc1));
            put_mcd(ext_hi(acc1));
        }
    }

    inline __attribute__((always_inline))
    v32uint8 load(v32uint8 chess_storage(DM_bankA) ** loadPtr) {
        v32uint8 v = **loadPtr;
        (*loadPtr) = (*loadPtr) + 1;
        return v;
    }

    inline __attribute__((always_inline))
    void store(v32uint8 chess_storage(DM_bankA) * restrict * storePtr, v32uint8 v) {
        **storePtr = v;
        (*storePtr) = (*storePtr) + 1;
    }

    inline __attribute__((always_inline))
    void move_write_pointer(v16uint8* restrict* wrPtr, const unsigned int outW){
        (*wrPtr) = (*wrPtr) + outW / 2;
    }

    inline __attribute__((always_inline))
    void reorder(v16uint8* restrict wrPtr,
                 const unsigned int M,
                 const unsigned int N,
                 const unsigned int Cin,
                 const unsigned int Cout,
                 const unsigned int F,
                 const unsigned int S,
                 const unsigned int CASC_IN_EN,
                 const unsigned int CASC_OUT_EN) {

        clr_sat();

        // restrict puts pointer in two different alliases (memory regions for the compiler)
        // This prevents the pointers to be collapsed into one
        // We hence have two pointer registers and can expand the loop
        v32uint8 chess_storage(DM_bankA)* loadPtr = (v32uint8 chess_storage(DM_bankA)*)wrPtr;
        v32uint8 chess_storage(DM_bankA)* restrict writePtr = (v32uint8 chess_storage(DM_bankA)*)wrPtr;

        // TODO most likely unroll everything here
        for(unsigned int c = 0; c < CoutUp/8; c++) {
            for(unsigned int y = 0; y < outHeightInner; y++) chess_prepare_for_pipelining {
                for(unsigned int x = 0; x < outWidthInner; x++) chess_unroll_loop(*) {
                    v32uint8 v = load(&loadPtr);
                    v32int16 vExt = unpack(v);

                    v32int16 vShuff = shuffle32(vExt, 0, 0x0a080200, 0x0e0c0604, 0x3210);

                    v32uint8 ret = as_v32uint8(pack(vShuff));
                    store(&writePtr, ret);
                }
            }
        }
    }

    //* For the moment only support 2x2 kernels with stride of 1 on input volume
    //* Assumes special ordering for the weights
    // TODO make generic
    inline __attribute__ ((always_inline))
    void transposed_conv2d_int8(input_window_int8* restrict actsIn,
                                input_window_int8* restrict weightsIn,
                                output_window_uint8* restrict actsOut,
                                const unsigned int M,
                                const unsigned int N,
                                const unsigned int Cin,
                                const unsigned int Cout,
                                const unsigned int F,
                                const unsigned int S,
                                const unsigned int CASC_IN_EN,
                                const unsigned int CASC_OUT_EN) {
        assert(F == 2); // Only one tested for the moment

        const unsigned int Fin = 1;
        const unsigned int Sin = 1;

        v16uint8* restrict wrPtr = (v16uint8*)actsOut->head;

        // If Cout%8 != 0 then unused weights are zeros
        for(unsigned int fi = 0; fi < F * F; fi++) {
            //actsOut->ptr = actsOut->head + (fi%2) * 16 + (fi/2) * outWidth * 8;
            wrPtr = (v16uint8*)(actsOut->head + (fi%2) * 16 + (fi/2) * outWidth * 8);
            for(unsigned int f = 0; f < CoutUp/8; f++) {
                for(unsigned int h = 0; h < (outHeight)/2; h+=1) chess_prepare_for_pipelining {
                        for(unsigned int w = 0; w < (outWidth)/2; w+=4) chess_unroll_loop(*) {
                        const unsigned int cond = (((outWidth/2) % 4) == 0) || !((w+4) >= (outWidth/2));
                        applyKernel(actsIn, weightsIn, &wrPtr,
                                    M, N, Cin, Cout, Fin, Sin,
                                    CASC_IN_EN, CASC_OUT_EN, cond);

                        //rewind_a(actsIn, MR, CinUp, Fin, N, Sin);
                        //const unsigned int doit = (CinUp > 8);
                        //rewind_w(weightsIn, Fin, CinUp, doit);
                    }

                    // Because of the convolution, we skip the last pixels
                    // as the resulting image is smaller
                    //skip_end_a(actsIn, MR, MR);
                    //move_write_pointer(&wrPtr, outWidth);
                }

                // Prepare pointers for next iteration
                //reset_a(actsIn);
                //seek_next_w(weightsIn, F, CinUp);
            }
        }

        // Put our data back in its format
        //if(CASC_OUT_EN == 0) {
        //    reorder((v16uint8*)actsOut->head, M, N, Cin, Cout, F, S, CASC_IN_EN, CASC_OUT_EN);
        //}
    }
}
