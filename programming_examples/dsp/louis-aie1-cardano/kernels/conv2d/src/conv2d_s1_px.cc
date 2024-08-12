//===- conv2d_s1_px.cc -----------------------------------000---*- C++ -*-===//
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

namespace S1PX {

#define X_C64_D8 0x38303830
#define Z_C8_D8_I_L 0x44440000
#define Z_C8_D8_I_U 0x88884444
#define Z_C8_D8_I_UU 0xcccc8888

#define PAD_LEFT 0
#define PAD_RIGHT 1
#define PAD_NONE 2

    // skip first line if needed
    inline __attribute__ ((always_inline))
    v32int8 load_w_first(input_window_int8* weightsIn, const unsigned int F,
                         const unsigned int nonZeroYStart, const unsigned int startLoop) {
        const unsigned int needToSkip = nonZeroYStart && startLoop;
        const unsigned int skipW = 8*8*F*needToSkip;
        weightsIn->ptr = weightsIn->ptr + skipW;

        v32int8 v = window_read_v32(weightsIn);
        weightsIn->ptr = weightsIn->ptr + 32;
        return v;
    }

    inline __attribute__ ((always_inline))
    v32int8 load_w_last(input_window_int8* weightsIn, const unsigned int F,
                        const unsigned int cinUp, const unsigned int rewindp,
                        const unsigned int nonFYEnd, const unsigned int endLoop) {
        const unsigned int needToSkip = nonFYEnd && endLoop;
        const unsigned int skipW = 8*8*F*needToSkip;
        const unsigned int rewind_modifier = ((8*8)*F*F*(cinUp/8)) * rewindp;

        v32int8 v = window_read_v32(weightsIn);
        weightsIn->ptr = weightsIn->ptr + 32 - rewind_modifier + skipW;

        return v;
    }

    inline __attribute__ ((always_inline))
    v32uint8 load_a(input_window_uint8* actsIn, v32uint8 v, const unsigned int i,
                    const unsigned int mr, const unsigned int F, const unsigned int padded,
                    const unsigned int N, const unsigned int lines,
                    const unsigned int cinUp,
                    const unsigned int doit_lines, const unsigned int doit_channels,
                    const unsigned int doit_rewind) {

        const unsigned int next_row = (1 + F/2 + (1 - padded) - (mr/2)) * 16 * doit_lines;
        const unsigned int next_channels = (1 + F/2 + (1-padded) - (mr/2)*(N-(lines-1))) * 16 * doit_channels;
        const unsigned int rewind_offset = (1 + F/2 + (1 - padded) + (cinUp/8-1) * mr/2*N + (lines-1)*mr/2 - (2-padded)) * doit_rewind * 16;

        v = upd_v(v, i, window_read_v16(actsIn));

        if((doit_lines != 0) || (doit_channels != 0) || (doit_rewind != 0)) {
            actsIn->ptr = actsIn->ptr - next_row - next_channels - rewind_offset;
        } else {
            actsIn->ptr = actsIn->ptr + chess_copy(0);
        }

        return v;
    }

    // inserts 8 0s in the given vector at the given location
    inline __attribute__ ((always_inline))
    v32uint8 load_a_pad(input_window_uint8* actsIn, v32uint8 v, const unsigned int i) {
        //actsIn->ptr = chess_copy(actsIn->ptr);
        //v32uint8 v = upd_v(v, i, window_read_v16(actsIn));
        // TODO double check correctness
        // set zero either top half or bottom half
        v = as_v32uint8(upd_elem(as_v8int32(v), i, 0, 0));

        return v;
    }

    inline __attribute__ ((always_inline))
    v32uint8 load_inc_a(input_window_uint8* actsIn, v32uint8 v, const unsigned int i,
                        const unsigned int mr, const unsigned int F, const unsigned int padded,
                        const unsigned int N, const unsigned int lines,
                        const unsigned int cinUp,
                        const unsigned int doit_lines, const unsigned int doit_channels,
                        const unsigned int doit_rewind) {
        const unsigned int next_row = (1 + F/2 + (1 - padded) - (mr/2)) * 16 * doit_lines;
        const unsigned int next_channels = (1 + F/2 + (1-padded) - (mr/2)*(N-(lines-1))) * 16 * doit_channels;
        const unsigned int rewind_offset = (1 + F/2 + (1 - padded) + (cinUp/8-1) * mr/2*N + (lines-1)*mr/2 - (2-padded)) * doit_rewind * 16;

        v = upd_v(v, i, window_read_v16(actsIn));

        if((doit_lines != 0) || (doit_channels != 0) || (doit_rewind != 0)) {
            actsIn->ptr = actsIn->ptr - next_row - next_channels - rewind_offset + 16;
        } else {
            actsIn->ptr = actsIn->ptr + 16;
        }

        return v;
        //window_incr_v16(actsIn, 1);
    }

    inline __attribute__ ((always_inline))
    void skip_end_a(input_window_uint8* actsIn, const unsigned int mr, const unsigned int outW,
                    const unsigned int P) {
        const unsigned int read = ((outW+3)/4)*4;
        const unsigned int padW = mr + P * 2;
        const unsigned int remaining = (padW - read) - P;
        const unsigned int offset = ((remaining+1) / 2) * 2 * 8;
        actsIn->ptr = actsIn->ptr + offset;
    }

    inline __attribute__ ((always_inline))
    void rewind_line(input_window_uint8* actsIn, const unsigned int mr, const unsigned int outW,
                    const unsigned int P) {
        const unsigned int read = ((outW+3)/4)*4 - P;
        const unsigned int offset = (read / 2) * 2 * 8;
        actsIn->ptr = actsIn->ptr - offset;
    }

    inline __attribute__ ((always_inline))
    void seek_next_row(input_window_uint8* actsIn, const unsigned int mr, const unsigned int F,
                       const unsigned int doit, const unsigned int padded) {
        // F/2 as we decrement pointer to allign on the first accumulator
        // move + 2 v16 because of the accumulator start location spread
        // move + F/2 from the highest location to compute meaningful results
        // move -1 because of the dec in loading scheme
        // hence, going back to start location is - F/2 - 1
        // If you are padded, going bazck to start means out of line boundaries
        // So we dec by one less for this specific case
        window_decr_v16(actsIn, (F/2 + (1 - padded) - (mr/2)) * doit);
    }

    inline __attribute__ ((always_inline))
    void seek_next_channels(input_window_uint8* actsIn, const unsigned int mr,
                            const unsigned int F, const unsigned int lines, const unsigned int N,
                            const unsigned int doit, const unsigned int padded) {
        window_decr_v16(actsIn, (F/2 + (1-padded) - (mr/2)*(N-(lines))) * doit);
    }

    inline __attribute__ ((always_inline))
    void rewind_a(input_window_uint8* actsIn, const unsigned int mr, const unsigned int cinUp,
                  const unsigned int F, const unsigned int lines, const unsigned int N,
                  const unsigned int S, const unsigned int padded) {
        //window_decr_v16(actsIn, F/2 + (cinUp/8-1) * mr/2*N + (F-1)*mr/2 - 2);
        // CinUp/8 - 1 * .. = full channel blocks
        // (F-1)*mr/2 = kernel application size
        // slighly different form to rewind as we can underflow as we add +2 anyway at the end
        window_decr_v16(actsIn, F/2 + 1 + (cinUp/8-1) * mr/2*N + (lines-1)*mr/2 - 2);
    }

    inline __attribute__ ((always_inline))
    void reset_a(input_window_uint8* actsIn) {
        actsIn->ptr = actsIn->head;
    }

    inline __attribute__ ((always_inline))
    void rewind_w(input_window_int8* weightsIn, const unsigned int F,
                  const unsigned int lines, const unsigned int cinUp, const unsigned int all) {
        const unsigned int l = (all == 1) ? F : lines; // simplistic assumption
        weightsIn->ptr = weightsIn->ptr - (8*8)*F*F*(cinUp/8 - 1) - (8*8) * F * l;
        //printf("rewind: %p\n", weightsIn->ptr - weightsIn->head);
    }

    inline __attribute__ ((always_inline))
    void seek_next_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        //window_incr_v32(weightsIn, (8/4)*F*F*(cinUp/8));
        weightsIn->ptr = weightsIn->ptr + (8*8)*F*F*(cinUp/8);
    }

    // WB acc0
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

    // Padding related pointer changes
    /*inline __attribute__((always_inline))
    void skip_weights(input_window_int8* weightsIn, const unsigned int p,
                      const unsigned int F) {
        // skip the zero block
        weightsIn->ptr = weightsIn->ptr + 8*8*F * p;
    }*/

    // TODO add included rewindp_w
    inline __attribute__((always_inline))
    void applyKernelPad(input_window_uint8* actsIn,
                        input_window_int8* weightsIn,
                        output_window_uint8* actsOut,
                        const unsigned int M,
                        const unsigned int N,
                        const unsigned int Cin,
                        const unsigned int Cout,
                        const unsigned int F,
                        const unsigned int S,
                        const unsigned int P,
                        const unsigned int ystart,
                        const unsigned int yend,
                        const unsigned int CASC_IN_EN,
                        const unsigned int CASC_OUT_EN,
                        const unsigned int store_acc1,
                        const unsigned int padLoc,
                        const unsigned int skipWeights) {

        const unsigned int padded = padLoc == PAD_LEFT;

        const unsigned int N_WRITES = ((F+1) / 2) * (CinUp/8);
        const unsigned int W_N_WRITES = F*F*(CinUp/8);
        const unsigned int linesToSkip = (yend - ystart);
        const unsigned int lines = yend - ystart;

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
        v32int8 chess_storage(wr2) w4 = undef_v32int8();
        v32int8 chess_storage(wr3) w5 = undef_v32int8();

        v64int8 chess_storage(xd) xd = undef_v64int8();

        v64int8 weightBuff0 = undef_v64int8();
        v64int8 weightBuff1 = undef_v64int8();

        // Need to specify storage to avoid two acts load in a cycle
        v32uint8 chess_storage(wc0) v0 = undef_v32uint8();
        v32uint8 chess_storage(wc1) v1 = undef_v32uint8();

        /*v32int8 w0 = undef_v32int8();
        v32int8 w1 = undef_v32int8();
        v32int8 w2 = undef_v32int8();
        v32int8 w3 = undef_v32int8();
        v32int8 w4 = undef_v32int8();
        v32int8 w5 = undef_v32int8();

        v64int8 xd = undef_v64int8();

        v64int8 weightBuff0 = undef_v64int8();
        v64int8 weightBuff1 = undef_v64int8();

        // Need to specify storage to avoid two acts load in a cycle
        v32uint8 v0 = undef_v32uint8();
        v32uint8 v1 = undef_v32uint8();*/

        // If CIn%8 != 0 then last iteration has 0 activations
        for(unsigned int c = 0; c < CinUp/8; c++) chess_unroll_loop(*) {
            for(unsigned int y = ystart; y < yend; y++) chess_unroll_loop(*) {
#pragma unroll
                for(unsigned int x = 0; x < F; x++) {
                    //actsIn->ptr = chess_copy(actsIn->ptr);
                    const unsigned int pat = x % 2; // access pattern for pat
                    const unsigned int wPat = (x + y*F + c*F*F) % 2; // global weight access

                    // Computes loop boundaries to allow for the pointer increment concatenation
                    const unsigned int endloop = (x == (F-1)) && (y == (yend-1)) && (c == (CinUp/8-1));
                    const unsigned int nextChannels = (x == (F-1)) && (y == (yend-1)) && (c < (CinUp/8-1));
                    const unsigned int nextLine = (x == (F-1)) && (y < (yend-1));
                    const unsigned int start_loop = c == 0 && y == ystart && x == 0;
                    const unsigned int skipChannelChunkTop = y == ystart && x == 0;
                    const unsigned int skipChannelChunkBottom = (y == (yend-1)) && (x == (F-1));

                    // same load pattern as strided kernel
                    if(x == 0) {
                        v32int8 l0 = load_w_first(weightsIn, F, ystart != 0, skipChannelChunkTop);
                        v32int8 l1 = load_w_last(weightsIn, F, CinUp, 0, yend != F, 0);

                        w0 = l0;
                        w1 = l1;

                        weightBuff0 = concat(w0, w1);

                        l0 = load_w_first(weightsIn, F, ystart != 0, 0);
                        l1 = load_w_last(weightsIn, F, CinUp, 0, yend != F, 0);

                        w4 = l0;
                        w2 = w4;
                        w3 = l1;

                        weightBuff1 = concat(w2, w3);
                    } else if(x == 2){
                        v32int8 l0 = load_w_first(weightsIn, F, ystart != 0, 0);
                        v32int8 l1 = load_w_last(weightsIn, F, CinUp, endloop, yend != F, skipChannelChunkBottom);

                        w4 = l0;
                        w5 = l1;

                        xd = concat(w4, w5);
                        weightBuff0 = xd;
                    } else {
                        // TODO only support F == 3 at the moment
                    }

                    if(padLoc == PAD_LEFT) {
                        if(x == 0) {
                            v0 = load_a_pad(actsIn, v0, 1);
                            v0 = load_a(actsIn, v0, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if(x == 1) {
                            v1 = load_inc_a(actsIn, v1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if((x%4) == 2){
                            v0 = load_inc_a(actsIn, v0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 0, MR, F, padded, N, linesToSkip, CinUp,
                                        nextLine, nextChannels, endloop);
                        } else if((x%4) == 0){
                            v0 = load_inc_a(actsIn, v0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        nextLine, nextChannels, endloop);
                        }

                    } else if(padLoc == PAD_RIGHT) {
                        if(x == 0) {
                            v0 = load_inc_a(actsIn, v0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v0 = load_a(actsIn, v0, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if(x == 1) {
                            v1 = load_inc_a(actsIn, v1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if((x%4) == 2){
                            if(x == (F-1)) {
                                v0 = load_inc_a(actsIn, v0, 0, MR, F, padded, N, linesToSkip, CinUp,
                                                nextLine, nextChannels, endloop);
                                v1 = load_a_pad(actsIn, v1, 0);
                            } else {
                                v0 = load_inc_a(actsIn, v0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                                v1 = load_a(actsIn, v1, 0, MR, F, padded, N, linesToSkip, CinUp,
                                            nextLine, nextChannels, endloop);
                            }
                        } else if((x%4) == 0){
                            v0 = load_inc_a(actsIn, v0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            if(x == (F-1)) {
                                v1 = load_a_pad(actsIn, v1, 2);
                            } else {
                                v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                            nextLine, nextChannels, endloop);
                            }
                        }

                    } else {// middle no padding
                        if(x == 0) {
                            v0 = load_inc_a(actsIn, v0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v0 = load_a(actsIn, v0, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if(x == 1) {
                            v1 = load_inc_a(actsIn, v1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        0, 0, 0);
                        } else if((x%4) == 2) {
                            v0 = load_inc_a(actsIn, v0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 0, MR, F, padded, N, linesToSkip, CinUp,
                                        nextLine, nextChannels, endloop);
                        } else if((x%4) == 0) {
                            v0 = load_inc_a(actsIn, v0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
                            v1 = load_a(actsIn, v1, 1, MR, F, padded, N, linesToSkip, CinUp,
                                        nextLine, nextChannels, endloop);
                        }
                    }

                    //chess_report(v0);
                    //chess_report(v1);
                    //chess_report(weightBuff0);
                    //chess_report(acc0);
                    //chess_report(acc1);

                    if((x == 0 && y == ystart && c == 0) && CASC_IN_EN == 0) {
                        // start 8
                        acc0 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                     v0, 0, 0x88884444, 2, 0x3210);

                        // start 16
                        acc0 = mac16(acc0, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                     v0, 8, 0x88884444, 2, 0x3210);

                    } else if((x == 1 && y == ystart && c == 0) && CASC_IN_EN == 0) {
                        // start 8
                        acc1 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                     v1, 0, 0x88884444, 2, 0x3210);

                        // start 16
                        acc1 = mac16(acc1, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                     v1, 8, 0x88884444, 2, 0x3210);
                    } else {
                        if(x == 0) { // v0 twice start low
                            // start 8
                            acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                         v0, 0, 0x88884444, 2, 0x3210);

                            // start 16
                            acc0 = mac16(acc0, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                         v0, 8, 0x88884444, 2, 0x3210);
                        } else if(x == 1) { // v1 twice start low
                            // start 8
                            acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                         v1, 0, 0x88884444, 2, 0x3210);

                            // start 16
                            acc1 = mac16(acc1, weightBuff1, 0, X_C64_D8, 4, 0x3210,
                                         v1, 8, 0x88884444, 2, 0x3210);
                        } else if((x%2) == 0) { // mixed
                            if((x%4) == 0) { // v0 v1 start low
                                // start 8
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 0, 0x88884444, 2, 0x3210);

                                // start 8
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 0, 0x88884444, 2, 0x3210);
                            } else { // v0 v1 start high
                                // start 24
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 0, 0x0000cccc, 2, 0x3210);

                                // start 24
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 0, 0x0000cccc, 2, 0x3210);
                            }
                        } else {
                            if((x%4) == 1) {
                                // start 16
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 8, 0x88880000, 2, 0x3210);

                                // start 16
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 8, 0x88880000, 2, 0x3210);
                            } else { // v0 v1 start high
                                // start 0
                                acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v0, 8, 0x0000cccc, 2, 0x3210);

                                // start 0
                                acc1 = mac16(acc1, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                             v1, 8, 0x0000cccc, 2, 0x3210);
                            }
                        }
                    }
                }
                //const unsigned int doit = y < (yend-1);
                //seek_next_row(actsIn, MR, F, doit, padded);
            }
            //const unsigned int doit = c < (Cin/8 - 1);
            // linesToSkip
            //seek_next_channels(actsIn, MR, F, linesToSkip, N, doit, padded);
        }

        // WB computed values
        //chess_report(acc0);
        //chess_report(acc1);

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

    // TODO maybe not keep this inlining
    inline __attribute__((always_inline))
    void paddingKernel(input_window_uint8* actsIn,
                       input_window_int8* weightsIn,
                       output_window_uint8* actsOut,
                       const unsigned int M,
                       const unsigned int N,
                       const unsigned int Cin,
                       const unsigned int Cout,
                       const unsigned int F,
                       const unsigned int S,
                       const unsigned int P,
                       const unsigned int PE,
                       const unsigned int PW,
                       const unsigned int ystart,
                       const unsigned int yend,
                       const unsigned int CASC_IN_EN,
                       const unsigned int CASC_OUT_EN) {

        const unsigned int lines = yend - ystart;

        // PE and PW are runtime dep
        //if(PE != 0) {
        // first iteration w == 0
        const unsigned int cond0 = ((outWidth % 4) == 0) || !((0+4) >= (outWidth));

        // computes 4 output pixels + relocate pointer for the next iterations
        applyKernelPad(actsIn, weightsIn, actsOut,
                       M, N, Cin, Cout, F, S, P, ystart, yend,
                       CASC_IN_EN, CASC_OUT_EN, cond0, PAD_LEFT, ystart != 0);
        //}

        // 4 here because of 4 pixels produced per kernel application
        // NOTE if no idea check chess_copy
        for(unsigned int w = 4; w < outWidth-(4); w+=4) chess_prepare_for_pipelining {
            // Fully static calls
            const unsigned int cond1 = ((outWidth % 4) == 0) || !((w+4) >= (outWidth));
            // computes 4 output pixels + relocate pointer for the next iterations
            applyKernelPad(actsIn, weightsIn, actsOut,
                           M, N, Cin, Cout, F, S, P, ystart, yend,
                           CASC_IN_EN, CASC_OUT_EN, cond1, PAD_NONE, ystart != 0);
        }

        // Compute the current value of w to check store condition
        const unsigned int w = (outWidth - 4); // TODO this may not be correct when only even
        const unsigned int cond2 = ((outWidth % 4) == 0) || !((w+4) >= (outWidth));

        // computes 4 output pixels + relocate pointer for the next iterations
        applyKernelPad(actsIn, weightsIn, actsOut,
                       M, N, Cin, Cout, F, S, P, ystart, yend,
                       CASC_IN_EN, CASC_OUT_EN, cond2, PAD_RIGHT, ystart != 0);

        // Because of the convolution, we skip the last pixels
        // as the resulting image is smaller
        if((ystart == 0) && (yend == F)) {
            skip_end_a(actsIn, MR, outWidth, P);
        } else { // Othw start back the line
            rewind_line(actsIn, MR, outWidth, P);
        }
    }


    //* assumes P smaller than F
    inline __attribute__ ((always_inline))
    void conv2d_int8_S1(input_window_uint8* restrict actsIn,
                        input_window_int8* restrict weightsIn,
                        output_window_uint8* restrict actsOut,
                        const unsigned int M,
                        const unsigned int N,
                        const unsigned int Cin,
                        const unsigned int Cout,
                        const unsigned int F,
                        const unsigned int S,
                        const unsigned int P,// This is a template parameter
                        const unsigned int PE,// the 4 next ones are boolean runtime parameters
                        const unsigned int PW,
                        const unsigned int PN,
                        const unsigned int PS,
                        const unsigned int CASC_IN_EN,
                        const unsigned int CASC_OUT_EN) {

        //actsIn->ptr = actsIn->head;
        //weightsIn->ptr = weightsIn->head;
        //actsOut->ptr = actsOut->head;

        //* Process special loading scheme layers separately
        //* Increase code size, but if too big
        //* Analyze if it makes sense to remove some unrolling from these loops
        //* depending on parameters

        for(unsigned int f = 0; f < CoutUp/8; f++) {
            // TODO if this is too costly, make these loop iterations less effective?
            //if(PN != 0) {
            //#pragma unroll
            //for(unsigned int p = P; p > 0; p--) chess_unroll_loop(*) {
            paddingKernel(actsIn, weightsIn, actsOut,
                          M, N, Cin, Cout, F, S, P,
                          PE, PW, 1, F,
                          CASC_IN_EN, CASC_OUT_EN);
            //}
            //}

            for(unsigned int h = P; h < outHeight-P; h+=1) chess_prepare_for_pipelining {// TODO keep pipelining?
                paddingKernel(actsIn, weightsIn, actsOut,
                              M, N, Cin, Cout, F, S, P,
                              PE, PW, 0, F,
                              CASC_IN_EN, CASC_OUT_EN);
            }

            //if(PS != 0) {
            // TODO if this is too costly, make these loop iterations less effective?
           //#pragma unroll
           //for(unsigned int p = 0; p < P; p++) {
            paddingKernel(actsIn, weightsIn, actsOut,
                          M, N, Cin, Cout, F, S, P,
                          PE, PW, 0, (F-1),
                          CASC_IN_EN, CASC_OUT_EN);
                //}
                //}

            // Prepare pointers for next iteration
            reset_a(actsIn);
            seek_next_w(weightsIn, F, CinUp);
        }
    }
}

