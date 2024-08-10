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

namespace S1 {
    inline __attribute__((always_inline))
    v32uint8 load_a_next_line(input_window_uint8* actsIn, const unsigned int mr) {
        v32uint8 v = window_read_v32(actsIn);
        actsIn->ptr = actsIn->ptr + mr * 8;
        return v;
    }

    inline __attribute__((always_inline))
    v32uint8 load_a_next_columns(input_window_uint8* actsIn, const unsigned int mr,
                                 const unsigned int F, const unsigned int outW,
                                 const unsigned int doit) {
        const unsigned int disp = (mr - outW) * 8 * doit;
        v32uint8 v = window_read_v32(actsIn);
        actsIn->ptr = actsIn->ptr - (F-1) * mr * 8 + 16 + disp;
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
    v32int8 load_w_rewind(input_window_int8* weightsIn, const unsigned int nextWeights) {
        v32int8 v = window_read_v32(weightsIn);
        if(nextWeights != 0) {
            weightsIn->ptr = weightsIn->ptr - 32 * 2 + 32 * 3;
        } else {
            weightsIn->ptr = weightsIn->ptr - 32 * 2;
        }
        return v;
    }

    // Only supports stride of 1 for now
    inline __attribute__ ((always_inline))
    void skip_end_row(input_window_uint8* actsIn, const unsigned int mr,
                      const unsigned int outW) {
        const unsigned int remaining = mr - outW;
        window_incr_v16(actsIn, remaining / 2);
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

    inline __attribute__ ((always_inline))
    void store(output_window_uint8* actsOut, v16acc48 acc) {
        v16uint8 v = ubsrs(acc, SHIFT);
        window_write(actsOut, v);
        window_incr_v16(actsOut, 1);
    }


    //* Simpler version with one load from weights / acts per cycle and one accumulator used
    //* Stride == 1 here, as stride == 2 means different loading scheme / acc usage
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
            for(unsigned int h = 0; h < outHeight; h++) {
                #pragma unroll // TODO replace that by picking out the last iteration
                for(unsigned int f = 0; f < (outWidth); f+=2) {
                    v16acc48 acc = undef_v16acc48();

                    #pragma unroll
                    for(unsigned int y = 0; y < DW_W; y++) {
                        v32int16 v;
                        v32int8 w;
                        if(y < (DW_W-1)) {
                            v = unpack(load_a_next_line(actsIn, MR));
                            w = load_w(weightsIn);
                        } else  {
                            v = unpack(load_a_next_columns(actsIn, MR, DW_W, outWidth, (f+2) >= outWidth));
                            w = load_w_rewind(weightsIn, (f+2) >= outWidth);
                        }

                        if(y == 0) {
                            acc = mul16(v, 0, 0x33323130, 0x37363534, 16, 0x3120,
                                        w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                        } else {
                            acc = mac16(acc, v, 0, 0x33323130, 0x37363534, 16, 0x3120,
                                        w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                        }
                    }

                    // store in the end
                    store(actsOut, acc);
                }

                // Last iteration have different pointer increment
                /*v16acc48 acc = undef_v16acc48();

                #pragma unroll
                for(unsigned int y = 0; y < DW_W; y++) {
                    v32int16 v;
                    v32int8 w;
                    if(y < (DW_W-1)) {
                        v = unpack(load_a_next_line(actsIn, MR));
                        w = load_w(weightsIn);
                    } else  {
                        v = unpack(load_a_next_columns(actsIn, MR, DW_W, outWidth, 1));
                        w = load_w_rewind(weightsIn);
                    }

                    if(y == 0) {
                        acc = mul16(v, 0, 0x33323130, 0x37363534, 16, 0x3120,
                                    w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                    } else {
                        acc = mac16(acc, v, 0, 0x33323130, 0x37363534, 16, 0x3120,
                                    w, 0, 0xeca86420, 0xeca86420, 2, 0x3210);
                    }
                }

                // store in the end
                store(actsOut, acc);*/

                //skip_end_row(actsIn, MR, outWidth); done on the last iteration to save a PADD
            }

            //seek_next_weights(weightsIn);
        }
    }
}
