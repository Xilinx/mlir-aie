#ifndef __chess_
#define __PTHREAD_API__
#define __NEW_X86Sim__
#endif

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include <stdio.h>
#include <stdint.h>
#include "dw_params.h"

namespace SX {

    inline __attribute__((always_inline))
    v32uint8 load_a_next_columns(input_window_uint8* actsIn, const unsigned int mr) {
        v32uint8 v = window_read_v32(actsIn);
        actsIn->ptr = actsIn->ptr + 32;
        return v;
    }

    inline __attribute__((always_inline))
    v32uint8 load_a_next_line(input_window_uint8* actsIn, const unsigned int mr) {
        v32uint8 v = window_read_v32(actsIn);
        const unsigned int n_accs = 2;
        actsIn->ptr = actsIn->ptr - (n_accs) * 32  + mr * 8;
        return v;
    }

    inline __attribute__((always_inline))
    v32uint8 load_a_next_chunk(input_window_uint8* actsIn, const unsigned int mr,
                               const unsigned int DW_W, const unsigned int outW,
                               const unsigned int DW_S, const unsigned int nextRow) {
        const unsigned int n_accs = 2; // means we process input pixels by 4
        const unsigned int readBy = 2 * n_accs;
        const unsigned int remaining = mr - DW_S * readBy * ((outW + readBy - 1) / readBy);
        const unsigned int rowMod = (remaining * 8 + (DW_S-1) * mr * 8) * rowMod;

        v32uint8 v = window_read_v32(actsIn);
        actsIn->ptr = actsIn->ptr - (DW_W-1) * mr * 8 + rowMod; //- n_accs * 32 + DW_S * (n_accs) * 32;
        return v;
    }

    inline __attribute__((always_inline))
    v32int8 load_w(input_window_int8* weightsIn) {
        // Weights are per 4 out of a 12 group grouped by 8 channels
        v32int8 v = window_read_v32(weightsIn);
        weightsIn->ptr = weightsIn->ptr + 32;
        return v;
    }

    // Next two horizontally (vertically) adjacent pixels
    inline __attribute__((always_inline))
    v32int8 load_w_rewind(input_window_int8* weightsIn) {
        v32int8 v = window_read_v32(weightsIn);
        weightsIn->ptr = weightsIn->ptr - 32 * 2;
        return v;
    }

    // Only supports stride of 1 for now
    inline __attribute__ ((always_inline))
    void skip_end_row(input_window_uint8* actsIn, const unsigned int mr,
                      const unsigned int outW, const unsigned int S) {
        const unsigned int n_accs = 2; // means we process input pixels by 4
        const unsigned int readBy = 2 * n_accs;
        const unsigned int remaining = mr - S * readBy * ((outW + readBy - 1) / readBy);
        actsIn->ptr = actsIn->ptr + remaining * 8 + (S-1) * mr * 8;
    }

    inline __attribute__((always_inline))
    void seek_next_weights(input_window_int8* weightsIn) {
        // as 8 * 3 * 4 = 32 * 3
        weightsIn->ptr = weightsIn->ptr + 32 * 3;
    }

    inline __attribute__ ((always_inline))
    void set_c(input_window_uint8* actsIn, const unsigned int mr, const unsigned int N,
               const unsigned int C) {
        actsIn->ptr = actsIn->head + C * (N*mr*8);
    }

    inline __attribute__((always_inline))
    void wb_0(output_window_uint8* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 v = ubsrs(acc, SHIFT);
        window_write(actsOut, v);
        window_incr_v16(actsOut, 1 + 1 * keep_acc1);
    }

    inline __attribute__((always_inline))
    void wb_1(output_window_uint8* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        window_incr_v16(actsOut, 1 * keep_acc1);
        window_write(actsOut, out);
        window_decr_v16(actsOut, 1 * keep_acc1);
    }

    //* Simpler version with one load from weights / acts per cycle and one accumulator used
    //* Stride == 2 here use 16x16 bits scheme here
    inline __attribute__ ((always_inline))
    void dw_conv_int8_3x3(input_window_uint8* restrict actsIn,
                          input_window_int8* restrict weightsIn,
                          output_window_uint8* restrict actsOut,
                          const unsigned int M,
                          const unsigned int N,
                          const unsigned int C,
                          const unsigned int DW_S) {

        // Only width supported so far
        const unsigned int DW_W = 3;

        // Used to convert back from 16 bits to 8 bits
        set_sat();

        // This kernel has a stride of 1
        for(unsigned int c = 0; c < C/8; c++) {
            set_c(actsIn, MR, N, c);
            for(unsigned int h = 0; h < outHeight; h++) chess_prepare_for_pipelining {
#pragma unroll
                for(unsigned int w = 0; w < outWidth; w+=4) {
                    v16acc48 acc0 = undef_v16acc48();
                    v16acc48 acc1 = undef_v16acc48();

#pragma unroll
                    for(unsigned int y = 0; y < DW_W; y++) {
                        v32int8 chess_storage(wc0) w;

                        v64int16 chess_storage(ya) ya;
                        v64int16 chess_storage(yd) yd;

                        if(y < (DW_W-1)) {
                            ya = upd_x(ya, 0, unpack(load_a_next_columns(actsIn, MR)));
                            ya = upd_x(ya, 1, unpack(load_a_next_columns(actsIn, MR)));
                            yd = upd_x(yd, 0, unpack(load_a_next_line(actsIn, MR)));
                            w = load_w(weightsIn);
                        } else  {
                            ya = upd_x(ya, 0, unpack(load_a_next_columns(actsIn, MR)));
                            ya = upd_x(ya, 1, unpack(load_a_next_columns(actsIn, MR)));
                            yd = upd_x(yd, 0, unpack(load_a_next_chunk(actsIn, MR, DW_W, outWidth, DW_S, 1)));
                            w = load_w_rewind(weightsIn);
                        }

                        //chess_report(v0);
                        //chess_report(v1);
                        //chess_report(v2);

                        //v64int16 chess_storage(ya) ya = concat(v0, v1);
                        //v64int16 chess_storage(yd) yd = concat(v2, v1);

                        if(y == 0) {
                            acc0 = mul16(ya, 0, 0x33323130, 0x3b3a3938, 16, 0x3120,
                                         w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                            acc1 = mul16(yd, 32, 0x33323130, 0x3b3a3938, 16, 0x3120,
                                         w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                        } else {
                            acc0 = mac16(acc0, ya, 0, 0x33323130, 0x3b3a3938, 16, 0x3120,
                                         w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                            acc1 = mac16(acc1, yd, 32, 0x33323130, 0x3b3a3938, 16, 0x3120,
                                         w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                        }
                    }

                    //chess_report(acc0);
                    //chess_report(acc1);

                    // store in the end
                    const unsigned int cond = ((outWidth % 4) == 0) || ((w+4) < outWidth);
                    wb_1(actsOut, acc1, cond);
                    wb_0(actsOut, acc0, cond);
                }

                //skip_end_row(actsIn, MR, outWidth, DW_S);
            }

            seek_next_weights(weightsIn);
        }
    }


}


