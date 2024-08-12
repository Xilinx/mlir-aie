//===- conv2d_sx.cc -----------------------------------000---*- C++ -*-===//
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

#include "conv2d_params.h"

//* Register access pattern + load BW requirement are different that S == 1 case

namespace SX {

#define X_C64_D8 0x38303830
#define Z_S_2 0x88880000

    inline __attribute__ ((always_inline))
    v32int8 load_w(input_window_int8* weightsIn,
                   const unsigned int F, const unsigned int cinUp,
                   const unsigned int enRewind) {
        const unsigned int rewindOffset = ((8/4)*F*F*cinUp/8) * 32 * enRewind;

        v32int8 v = window_read_v32(weightsIn);
        weightsIn->ptr = weightsIn->ptr + 32 - rewindOffset;

        return v;
    }

    inline __attribute__ ((always_inline))
    void unload_2_w(input_window_int8* weightsIn) {
        window_decr_v32(weightsIn, 2);
    }

    inline __attribute__ ((always_inline))
    v32uint8 load_a(input_window_uint8* actsIn, v32uint8 v, const unsigned int i,
                    const unsigned int S,
                    const unsigned int F, const unsigned int mr,
                    const unsigned int N, const unsigned int cinUp,
                    const unsigned int enRewind) {
        //const unsigned int rowOffset = ((8 * S * 3 + (F-1)*8 - 16) - mr * 8) * nextRow;
        //const unsigned int cOffset = ((8 * S * 3 + (F-1)*8 - 16) - mr * 8 * (N - (F-1))) * nextChannels;
        //const unsigned int regularOffset = ((S/2)*2) * 8;

        const unsigned int filterRewind = (8 * S * 3 + (F-1)*8 - 16);
        const unsigned int rowsRewind = mr * 8 * (F - 1);
        const unsigned int chansRewind = N * mr * 8 * (cinUp/8 - 1);
        const unsigned int finalDisp = 4 * S * 8;
        const unsigned int rewindOffset = (- filterRewind - rowsRewind - chansRewind + finalDisp) * enRewind;

        v = upd_v(v, i, window_read_v16(actsIn));

        if(enRewind != 0) {
            actsIn->ptr = actsIn->ptr + ((S/2)*2) * 8;// + rewindOffset;// - rowOffset - nextChannels;
        } else {
            actsIn->ptr = actsIn->ptr + ((S/2)*2) * 8;
        }

        return v;
    }

    inline __attribute__ ((always_inline))
    void skip_end_a(input_window_uint8* actsIn, const unsigned int mr,
                    const unsigned int S, const unsigned int outW) {
        const unsigned int read = S*4*((outW+3)/4);
        const unsigned int remaining = ((mr - read) / 2) * 16;
        actsIn->ptr = actsIn->ptr + remaining + (S-1) * (mr/2) * 16;
    }

    inline __attribute__ ((always_inline))
    void seek_next_row(input_window_uint8* actsIn, const unsigned int mr, const unsigned int F,
                       const unsigned int S, const unsigned int doit) {
        // remove strided start position + rightmost kernel displacement
        // then add line size to go to the next
        const unsigned int modifier = (-(8 * S * 3 + (F-1)*8 - 16) + mr * 8);
        actsIn->ptr = actsIn->ptr + modifier;
    }

    inline __attribute__ ((always_inline))
    void seek_next_channels(input_window_uint8* actsIn, const unsigned int mr,
                            const unsigned int F, const unsigned int N,
                            const unsigned int S, const unsigned int doit) {
        const unsigned int modifier = (-(8 * S * 3 + (F-1)*8 - 16) + mr * 8 * (N - (F-1)));
        actsIn->ptr = actsIn->ptr + modifier;
    }

    inline __attribute__ ((always_inline))
    void rewind_a(input_window_uint8* actsIn, const unsigned int mr, const unsigned int cinUp,
                    const unsigned int F, const unsigned int N, const unsigned int S) {
        const unsigned int filterRewind = (8 * S * 3 + (F-1)*8 - 16);
        const unsigned int rowsRewind = mr * 8 * (F - 1);
        const unsigned int chansRewind = N * mr * 8 * (cinUp/8 - 1);
        const unsigned int finalDisp = 4 * S * 8;
        actsIn->ptr = actsIn->ptr - filterRewind - rowsRewind - chansRewind + finalDisp;
    }

    inline __attribute__ ((always_inline))
    void reset_a(input_window_uint8* actsIn) {
        actsIn->ptr = actsIn->head;
    }

    inline __attribute__ ((always_inline))
    void rewind_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        window_decr_v32(weightsIn, (8/4)*F*F*cinUp/8);
    }

    inline __attribute__ ((always_inline))
    void seek_next_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        window_incr_v32(weightsIn, (8/4)*F*F*cinUp/8);
    }

    inline __attribute__ ((always_inline))
    void wb_0(output_window_uint8* actsOut, v16acc48 acc, const unsigned int erase_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        window_write(actsOut, out);
        window_incr_v16(actsOut, 1 + 1 * erase_acc1);
    }

    // WB acc1
    inline __attribute__ ((always_inline))
    void wb_1(output_window_uint8* actsOut, v16acc48 acc, const unsigned int erase_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        window_incr_v16(actsOut, 1 * erase_acc1);
        window_write(actsOut, out);
        window_decr_v16(actsOut, 1 * erase_acc1);
    }

    // For the moment only support S == 2
    inline __attribute__((always_inline))
    void apply_kernel(input_window_uint8* actsIn,
                      input_window_int8* weightsIn,
                      output_window_uint8* restrict actsOut,
                      const unsigned int M,
                      const unsigned int N,
                      const unsigned int Cin,
                      const unsigned int Cout,
                      const unsigned int F,
                      const unsigned int S,
                      const unsigned int P,
                      const unsigned int CASC_IN_EN,
                      const unsigned int CASC_OUT_EN,
                      const unsigned int store_acc1) {

        // TODO improve that
        assert(S == 2);

        const unsigned int N_WRITES = ((F+1) / 2) * (CinUp/8);
        const unsigned int W_N_WRITES = F*F*(CinUp/8);

        v16acc48 chess_storage(BM) acc0 = undef_v16acc48();
        v16acc48 chess_storage(BM) acc1 = undef_v16acc48();

        if(CASC_IN_EN == 1) {
            acc0 = concat(get_scd(), get_scd());
            acc1 = concat(get_scd(), get_scd());
        }

        // ensure back to back weight load on XA / XD
        v32int8 chess_storage(wr0) w0 = undef_v32int8();
        v32int8 chess_storage(wr1) w1 = undef_v32int8();
        v32int8 chess_storage(wd0) w2 = undef_v32int8();
        v32int8 chess_storage(wd1) w3 = undef_v32int8();
        v32int8 chess_storage(wr2) w4 = undef_v32int8();
        v32int8 chess_storage(wr3) w5 = undef_v32int8();

        //v64int8 chess_storage(xa) xa;
        //v64int8 chess_storage(xb) xb;
        v64int8 chess_storage(xd) xd = undef_v64int8();

        v64int8 weightBuff0 = undef_v64int8();
        v64int8 weightBuff1 = undef_v64int8();

        // Need to specify storage to avoid two acts load in a cycle
        v32uint8 chess_storage(wc0) v0 = undef_v32uint8();
        v32uint8 chess_storage(wc1) v1 = undef_v32uint8();

        // If CIn%8 != 0 then last iteration has 0 padded activations
#pragma unroll
        for(unsigned int c = 0; c < CinUp/8; c++) { //chess_unroll_loop(*)
#pragma unroll
            for(unsigned int y = 0; y < F; y++) {
#pragma unroll
                for(unsigned int x = 0; x < F; x++) {
                    actsIn->ptr = chess_copy(actsIn->ptr);
                    const unsigned int pat = x % 2; // access pattern for pat
                    const unsigned int wPat = (x + y*F + c*F*F) % 2; // global weight access
                    const unsigned int enRewind = (x == (F-1)) && (y == (F-1)) && (c == (CinUp/8 - 1));

                    // Select weight to load depending on the number of reads
                    //v32int8 l0 = load_w(weightsIn);
                    //v32int8 l1 = load_w(weightsIn);
                    //v64int8 lw0;
                    //v64int8 lw1;

                    // TODO adapt to longer F, at the moment only F == 3
                    if(x == 0) {
                        v32int8 l0 = load_w(weightsIn, F, CinUp, 0);
                        v32int8 l1 = load_w(weightsIn, F, CinUp, 0);
                        w0 = l0;
                        w1 = l1;

                        weightBuff0 = concat(w0, w1);

                        l0 = load_w(weightsIn, F, CinUp, 0);
                        l1 = load_w(weightsIn, F, CinUp, enRewind);
                        w4 = l0; // load
                        w2 = w4; // mov and load in parallel
                        w3 = l1;

                        weightBuff1 = concat(w2, w3);
                    } else if(x == 2) {
                        v32int8 l0 = load_w(weightsIn, F, CinUp, 0);
                        v32int8 l1 = load_w(weightsIn, F, CinUp, enRewind);

                        w4 = l0; // load in B
                        w5 = l1;

                        // mov in D for usage
                        xd = concat(w4, w5);

                        weightBuff0 = xd;
                    }

                    //weightBuff0 = lw0;
                    //weightBuff1 = lw1;

                    const unsigned int enRows = (y < (F-1)) && (x == (F-1));
                    const unsigned int enChans = (c < (CinUp/8 - 1)) && (y == (F-1)) && (x == (F-1));

                    if(x == 0) {
                        v0 = load_a(actsIn, v0, 0, S, F, MR, N, CinUp, 0);
                        v0 = load_a(actsIn, v0, 1, S, F, MR, N, CinUp, 0);
                    } else if(x == 1) {
                        v1 = load_a(actsIn, v1, 0, S, F, MR, N, CinUp, 0);
                        v1 = load_a(actsIn, v1, 1, -S, F, MR, N, CinUp, enRewind);
                    } else if((x%4) == 2) {
                        v0 = load_a(actsIn, v0, 0, 2*S, F, MR, N, CinUp, 0);
                        v1 = load_a(actsIn, v1, 0, -S, F, MR, N, CinUp, enRewind);
                    } else if((x%4) == 0) {
                        v0 = load_a(actsIn, v0, 1, 2*S, F, MR, N, CinUp, 0);
                        v1 = load_a(actsIn, v1, 1, -S, F, MR, N, CinUp, enRewind);
                    }

                    //chess_report(v0);
                    //chess_report(v1);
                    //chess_report(weightBuff0);

                    if((x == 0 && y == 0 && c == 0) && CASC_IN_EN == 0) {
                        // start 0
                        acc0 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                     v0, 0, 0x88880000, 2, 0x3210);

                        // start 1
                        acc0 = mac16(acc0, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                     v0, 8, 0x88880000, 2, 0x3210);

                    } else if((x == 1 && y == 0 && c == 0) && CASC_IN_EN == 0) {
                        // start 0
                        acc1 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                     v1, 0, 0x88880000, 2, 0x3210);

                        // start 1
                        acc1 = mac16(acc1, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                     v1, 8, 0x88880000, 2, 0x3210);
                    } else {
                        if(x == 0) { // v0 twice start low
                            // start 0
                            acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                         v0, 0, 0x88880000, 2, 0x3210);

                            // start 1
                            acc0 = mac16(acc0, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                         v0, 8, 0x88880000, 2, 0x3210);
                        } else if(x == 1) { // v1 twice start low
                            // start 0
                            acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                         v1, 0, 0x88880000, 2, 0x3210);

                            // start 1
                            acc1 = mac16(acc1, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                         v1, 8, 0x88880000, 2, 0x3210);
                        } else if((x%2) == 0) { // mixed
                            if((x%4) == 0) { // v0 v1 start low
                                // start 0
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 0, 0x88880000, 2, 0x3210);

                                // start 0
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 8, 0x88880000, 2, 0x3210);
                            } else { // v0 v1 start high
                                // start 2
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 0, 0x00008888, 2, 0x3210);

                                // start 2
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 0, 0x00008888, 2, 0x3210);
                            }
                        } else {
                            if((x%4) == 0) {
                                // start 1
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 8, 0x88880000, 2, 0x3210);

                                // start 1
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 8, 0x88880000, 2, 0x3210);
                            } else { // v0 v1 start high
                                // start 3
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 8, 0x0000888, 2, 0x3210);

                                // start 3
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 8, 0x00008888, 2, 0x3210);
                            }
                        }
                    }

                    //chess_report(acc0);
                }
                //const unsigned int enRow = (y < (F-1));
                if(y < F-1) {
                    seek_next_row(actsIn, MR, F, S, 1);
                }
            }
            if(c < (CinUp/8 - 1)) {
                seek_next_channels(actsIn, MR, F, N, S, 1);
            }
            //const unsigned int enChs = (c < (CinUp/8 - 1));
        }

        //chess_report(acc0);
        //if(store_acc1) {
        //    chess_report(acc1);
        //}

        // WB computed values
        if(CASC_OUT_EN == 0) {
            wb_1(actsOut, acc1, store_acc1);
            wb_0(actsOut, acc0, store_acc1);
        } else {
            put_mcd(ext_lo(acc0));
            put_mcd(ext_hi(acc0));

            put_mcd(ext_lo(acc1));
            put_mcd(ext_hi(acc1));
        }
    }

    inline __attribute__ ((always_inline))
    void conv2d_int8_S(input_window_uint8* restrict actsIn,
                       input_window_int8* restrict weightsIn,
                       output_window_uint8* restrict actsOut,
                       const unsigned int M,
                       const unsigned int N,
                       const unsigned int Cin,
                       const unsigned int Cout,
                       const unsigned int F,
                       const unsigned int S,
                       const unsigned int P,
                       const unsigned int CASC_IN_EN,
                       const unsigned int CASC_OUT_EN) {

        // If Cout%8 != 0 then unused weights are zeros
        for(unsigned int f = 0; f < CoutUp/8; f++) {
            for(unsigned int h = 0; h < outHeight; h+=1) chess_prepare_for_pipelining {
                for(unsigned int w = 0; w < outWidth; w+=4) chess_prepare_for_pipelining {
                    // TODO what if we allow computation of the next line to start at the same time as the current one?
                    const unsigned int cond = ((outWidth % 4) == 0) || !((w+4) >= (outWidth));
                    apply_kernel(actsIn, weightsIn, actsOut, M, N, Cin, Cout,
                                 F, S, P, CASC_IN_EN, CASC_OUT_EN, cond);

                    // prepare pointers for next iterations
                    // rewind_w(weightsIn, F, CinUp);
                    rewind_a(actsIn, MR, CinUp, F, N, S);
                }

                // Because of the convolution, we skip the last pixels
                // as the resulting image is smaller
                // TODO What if we already computed a portion of the next line?
                skip_end_a(actsIn, MR, S, outWidth);
            }

            // Prepare pointers for next iteration
            reset_a(actsIn);
            seek_next_w(weightsIn, F, CinUp);
        }
    }
}

