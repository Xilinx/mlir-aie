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
#include "avg_params.h"

#define SHIFT 20

inline __attribute__((always_inline))
v32uint8 load_a(input_window_uint8* actsIn, const unsigned int mr) {
    v32uint8 v = window_read_v32(actsIn);
    actsIn->ptr = actsIn->ptr + 32;
    return v;
}

inline __attribute__((always_inline))
v32uint8 load_a_inc_line(input_window_uint8* actsIn, const unsigned int mr) {
    v32uint8 v = window_read_v32(actsIn);
    actsIn->ptr = actsIn->ptr - 32 + mr * 8;
    return v;
}

inline __attribute__((always_inline))
void next_loc(input_window_uint8* actsIn, const unsigned int mr, const unsigned int F,
              const unsigned int doit, const unsigned int outW) {
    const unsigned int remaining = mr - outW;
    actsIn->ptr = actsIn->ptr - F * mr * 8 + (2) * 8 + doit * remaining * 8;
}

// Only supports stride of 1 for now
inline __attribute__ ((always_inline))
void skip_end_row(input_window_uint8* actsIn, const unsigned int mr, const unsigned int F, const unsigned int outW) {
    const unsigned int remaining = mr - outW;
    window_incr_v16(actsIn, remaining / 2);
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

// For now only support 2x2 max pooling layers

// load 256 = X0-3Y0
// load 256 = X0-3Y1
// max vertical (100%)
// max horizontal (50%) -> can pack first 16 values
// max horizontal (50%) -> can pack second 16 values

#define SINGLE 0

// 50% usage for now, simple version
inline __attribute__ ((always_inline))
void avgpool_int8_7x7(input_window_uint8* restrict actsIn,
                      output_window_uint8* restrict actsOut,
                      const unsigned int M,
                      const unsigned int N,
                      const unsigned int C) {

    // we deal with constant stride and width in this file
    const unsigned int MP_S = 1;
    const unsigned int MP_W = 7;

    if(SINGLE == (M == 7)) {
        assert(0);
    }

    // Used to convert back from 16 bits to 8 bits
    set_sat();

    // Used to round
    set_rnd(rnd_sym_inf);

    int multiplier = (int)(((int64_t)1 << SHIFT) / 49);
    printf("multiplier: %d\n", multiplier);
    v16int16 vmult = undef_v16int16();
    vmult = upd_elem(vmult, 0, multiplier);

    // This kernel has a stride of 1
    for(unsigned int c = 0; c < C/8; c++) { // chess_unroll_loop(*)
        set_c(actsIn, MR, N, c);
        for(unsigned int h = 0; h < outHeight; h++) {
            for(unsigned int w = 0; w < chess_copy(outWidth); w+=2) {// chess_prepare_for_pipelining
#if SINGLE == 1
                // load
                v32int16 chess_storage(xa) v0 = unpack(load_a(actsIn, MR));
                v32int16 chess_storage(xb) v1 = unpack(load_a_inc_line(actsIn, MR));
#else
                v32int16 v0 = unpack(load_a(actsIn, MR));
                v32int16 v1 = unpack(load_a_inc_line(actsIn, MR));
#endif
                // Move v0 to xd
                v32int16 v0_d = v0;

                v8acc48 accR;
                if(M != 7) {
                    // set v0 LSB to 0 in xa
                    v0 = upd_v(v0, 0, null_v8int16());
                    // compute with ya
                    accR = mul8_sym(v0, 0, 0x33323130, 16, 0x3120,
                                    v1, 0, 0x3120,
                                    vmult, 0, 0x00000000, 0);
                }

                // set v1 MSB to 0
                v1 = upd_v(v1, 3, null_v8int16());
                // compute with yb
                v8acc48 accL = mul8_sym(v0_d, 0, 0x33323130, 16, 0x3120,
                                        v1, 0, 0x3120, vmult, 0, 0x00000000, 0);

                for(unsigned int fy = 1; fy < (MP_W-1); fy++) chess_unroll_loop(*) {
                        // load
                        v0 = unpack(load_a(actsIn, MR));
                        v1 = unpack(load_a_inc_line(actsIn, MR));

                        // move v0 to d
                        v0_d = v0;
                        v0 = upd_v(v0, 0, null_v8int16());
                        if(M != 7) {
                            // compute with ya
                            accR = mac8_sym(accR, v0, 0, 0x33323130, 16, 0x3120,
                                            v1, 0, 0x3120,
                                            vmult, 0, 0x00000000, 0);

                        }

                        // compute with yd
                        v1 = upd_v(v1, 3, null_v8int16());
                        accL = mac8_sym(accL, v0_d, 0, 0x33323130, 16, 0x3120,
                                        v1, 0, 0x3120,
                                        vmult, 0, 0x00000000, 0);
                }

                v0 = unpack(load_a(actsIn, MR));
                v1 = unpack(load_a_inc_line(actsIn, MR));
                v0_d = v0;

                v0 = upd_v(v0, 0, null_v8int16());
                if(M != 7) {
                    accR = mac8_sym(accR, v0, 0, 0x33323130, 16, 0x3120,
                                    v1, 0, 0x3120,
                                    vmult, 0, 0x00000000, 0);
                }

                v1 = upd_v(v1, 3, null_v8int16());
                accL = mac8_sym(accL, v0_d, 0, 0x33323130, 16, 0x3120,
                            v1, 0, 0x3120,
                            vmult, 0, 0x00000000, 0);



                if(M != 7) {
                    v16acc48 fullAcc = concat(accL, accR);
                    v16uint8 res = ubsrs(fullAcc, SHIFT);
                    store(actsOut, res);
                } else {
                    v16acc48 fullAcc = concat(accL, undef_v8acc48());
                    v16uint8 res = ubsrs(fullAcc, SHIFT);
                    store(actsOut, res);
                }


                const unsigned int doit = (w+2) >= outWidth;
                next_loc(actsIn, MR, MP_W, doit, outWidth);
            }

            //skip_end_row(actsIn, MR, MP_W, outWidth);
        }
    }
}
