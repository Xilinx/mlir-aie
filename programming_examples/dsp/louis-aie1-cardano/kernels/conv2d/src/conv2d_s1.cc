#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>
#include <stdio.h>
#include <cassert>
#include "conv2d_params.h"

namespace S1 {

#define X_C64_D8 0x38303830
#define Z_C8_D8_I_L 0x44440000
#define Z_C8_D8_I_U 0x88884444
#define Z_C8_D8_I_UU 0xcccc8888

    inline __attribute__ ((always_inline))
    v32int8 load_w(input_window_int8* weightsIn,
                   const unsigned int F, const unsigned int cinUp,
                   const unsigned int enRewind) {
        const unsigned int rewindOffset = ((8/4)*F*F*(cinUp/8)) * enRewind * 32;
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
                    const unsigned int F, const unsigned int mr, const unsigned int N,
                    const unsigned int cinUp,
                    const unsigned int nextRow, const unsigned int nextChannels,
                    const unsigned int enRewind) {

        const unsigned int rowOffset = ((1+F/2 - (mr/2)) * nextRow) * 16;
        const unsigned int channelsOffset = ((1+F/2 - (mr/2)*(N-(F-1))) * nextChannels) * 16;
        const unsigned int rewindOffset = (1+F/2 + (cinUp/8-1) * mr/2*N + (F-1)*mr/2 - 2) * enRewind * 16;

        v = upd_v(v, i, window_read_v16(actsIn));

        if((nextRow != 0) || (nextChannels != 0) || (enRewind != 0)) {
            actsIn->ptr = actsIn->ptr - rowOffset - channelsOffset - rewindOffset;
        } else {
            // Introduce a fake dependency between the load activations to prevent two activations load in a cycle
            // This way we are guaranteed to load weights in all cycles
            actsIn->ptr = actsIn->ptr + chess_copy(0);
        }

        return v;
    }

    inline __attribute__ ((always_inline))
    v16uint8 load_a_simple(input_window_uint8* actsIn) {
        return window_read_v16(actsIn);
    }

    inline __attribute__ ((always_inline))
    v32uint8 load_inc_a(input_window_uint8* actsIn, v32uint8 v, const unsigned int i) {
        return upd_v(v, i, window_readincr_v16(actsIn));
    }

    inline __attribute__ ((always_inline))
    v16uint8 load_inc_a_simple(input_window_uint8* actsIn) {
        return window_readincr_v16(actsIn);
    }

    inline __attribute__ ((always_inline))
    void skip_end_a(input_window_uint8* actsIn, const unsigned int mr, const unsigned int outW) {
        const unsigned int read = 4*((outW+3)/4);
        const unsigned int remaining = ((mr - read) / 2) * 16;
        actsIn->ptr = actsIn->ptr + chess_copy(remaining);
    }

    inline __attribute__ ((always_inline))
    void seek_next_row(input_window_uint8* actsIn, const unsigned int mr, const unsigned int F,
                       const unsigned int doit) {
        window_decr_v16(actsIn, (1+F/2 - (mr/2)) * doit);
    }

    inline __attribute__ ((always_inline))
    void seek_next_channels(input_window_uint8* actsIn, const unsigned int mr,
                            const unsigned int F, const unsigned int N,
                            const unsigned int doit) {
        window_decr_v16(actsIn, (1+F/2 - (mr/2)*(N-(F-1))) * doit);
    }

    inline __attribute__ ((always_inline))
    void rewind_a(input_window_uint8* actsIn, const unsigned int mr, const unsigned int cinUp,
                  const unsigned int F, const unsigned int N, const unsigned int S) {
        // rewind x filter displacement
        // rewind channel displacement
        // rewind y filter displacement
        // move forward by 4 pixels
        window_decr_v16(actsIn, 1+F/2 + (cinUp/8-1) * mr/2*N + (F-1)*mr/2 - 2);
    }

    inline __attribute__ ((always_inline))
    void reset_a(input_window_uint8* actsIn) {
        actsIn->ptr = actsIn->head;
    }

    inline __attribute__ ((always_inline))
    void rewind_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        window_decr_v32(weightsIn, (8/4)*F*F*(cinUp/8));
    }

    inline __attribute__ ((always_inline))
    void seek_next_w(input_window_int8* weightsIn, const unsigned int F, const unsigned int cinUp) {
        window_incr_v32(weightsIn, (8/4)*F*F*(cinUp/8));
    }

    // WB acc0
    inline __attribute__ ((always_inline))
    void wb_0(output_window_uint8* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        window_write(actsOut, out);
        window_incr_v16(actsOut, 1 + 1 * keep_acc1);
    }

    // WB acc1
    inline __attribute__ ((always_inline))
    void wb_1(output_window_uint8* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        window_incr_v16(actsOut, 1 * keep_acc1);
        window_write(actsOut, out);
        window_decr_v16(actsOut, 1 * keep_acc1);
    }

    inline __attribute__((always_inline))
    void applyKernel(input_window_uint8* actsIn,
                     input_window_int8* weightsIn,
                     output_window_uint8* actsOut,
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
        // everything is static, but store_acc1

        const unsigned int N_WRITES = ((F+1) / 2) * (CinUp/8);
        const unsigned int W_N_WRITES = F*F*(CinUp/8);

        v16acc48 acc0 = undef_v16acc48();
        v16acc48 acc1 = undef_v16acc48();

        // Preload accumulator with value from cascade
        if(CASC_IN_EN == 1) {
            acc0 = concat(get_scd(), get_scd());
            acc1 = concat(get_scd(), get_scd());
        }

        // ensure back to back weight load on XA / XD
        // As XA and XD are the only registers we can read from with mac intrinsics
        // To guide the scheduler
        v32int8 chess_storage(wr0) w0 = undef_v32int8();
        v32int8 chess_storage(wr1) w1 = undef_v32int8();
        v32int8 chess_storage(wd0) w2 = undef_v32int8();
        v32int8 chess_storage(wd1) w3 = undef_v32int8();

        // name for either XA and XB depending on the iteration
        v64int8 weightBuff0 = undef_v64int8();

        // Need to specify storage to avoid two acts load in a cycle
        v32uint8 chess_storage(wc0) v0 = undef_v32uint8();
        v32uint8 chess_storage(wc1) v1 = undef_v32uint8();

        // If CIn%8 != 0 then last iteration has 0 activations
        for(unsigned int c = 0; c < CinUp/8; c++) chess_unroll_loop(*) {
            for(unsigned int y = 0; y < F; y++) chess_unroll_loop(*) {
                for(unsigned int x = 0; x < F; x++) chess_unroll_loop(*) {
                    const unsigned int pat = x % 2; // access pattern for pat
                    const unsigned int wPat = (x + y*F + c*F*F) % 2; // global weight access
                    const unsigned int enRewind = (x == (F-1)) && (y == (F-1)) && (c == (CinUp/8 - 1));

                    v32int8 l0 = load_w(weightsIn, 0, 0, 0);
                    v32int8 l1 = load_w(weightsIn, F, CinUp, enRewind);
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
                                // MOV to allow period of 1 and efficient generated code
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
                        const unsigned int enRows = (y < (F-1)) && (x == (F-2));
                        const unsigned int enChans = (c < (CinUp/8 - 1)) && (y == (F-1)) && (x == (F-2));
                        const unsigned int enRewindA = (x == (F-2)) && (y == (F-1)) && (c == (CinUp/8 - 1));

                        if((N_WRITES % 2) == 0) { // even number of read, register period of 1
                            const unsigned int regWrite = (((x+1)/2)) % 2;
                            if(regWrite == 0) {
                                v0 = load_inc_a(actsIn, v0, 0);
                                v0 = load_a(actsIn, v0, 1, F, MR, N, CinUp, enRows, enChans, enRewindA);
                            } else {
                                v1 = load_inc_a(actsIn, v1, 0);
                                v1 = load_a(actsIn, v1, 1, F, MR, N, CinUp, enRows, enChans, enRewindA);
                            }
                        } else { // odd number of read, register period of 2
                            const unsigned int regWrite = (((x+1)/2) + (y%2) + (c%2)) % 2;
                            if((x == 0) && (y == 0) && (c == 0)) {
                                v1 = load_inc_a(actsIn, v1, 0);
                                v1 = load_a(actsIn, v1, 1, F, MR, N, CinUp, enRows, enChans, enRewindA);
                                v0 = v1;
                            } else {
                                if(regWrite == 0) {
                                    v0 = load_inc_a(actsIn, v0, 0);
                                    v0 = load_a(actsIn, v0, 1, F, MR, N, CinUp, enRows, enChans, enRewindA);
                                } else {
                                    v1 = load_inc_a(actsIn, v1, 0);
                                    v1 = load_a(actsIn, v1, 1, F, MR, N, CinUp, enRows, enChans, enRewindA);
                                }
                            }
                        }
                    }

                    // Which act register to use for the first MAC
                    unsigned int regUp = ((N_WRITES % 2) == 0) ? ((x/2) % 2) : ((x/2 + y%2 + c%2) % 2);
                    v32uint8 actBuff = (regUp == 0) ? v0 : v1;
                    unsigned int zoff = (pat == 0) ? Z_C8_D8_I_L : Z_C8_D8_I_U;

                    if((x == 0 && y == 0 && c == 0) && CASC_IN_EN == 0) {
                        acc0 = mul16(weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    } else {
                        acc0 = mac16(acc0, weightBuff0, 0, X_C64_D8, 4, 0x3210,
                                        actBuff, 0, zoff, 2, 0x3210);
                    }

                    // decide act register for second mac
                    unsigned int regLow = ((N_WRITES % 2) == 0) ? ((x+1)/2) % 2 : ((x+1)/2 + y%2 + c%2) % 2;
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
            }
        }

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
    void conv2d_int8_S1(input_window_uint8* restrict actsIn,
                        input_window_int8* restrict weightsIn,
                        output_window_uint8* restrict actsOut,
                        const unsigned int M, // Width of tile
                        const unsigned int N, // height of tile
                        const unsigned int Cin, // number of input channels
                        const unsigned int Cout, // number of output channels
                        const unsigned int F,// filter width
                        const unsigned int S,// stride
                        const unsigned int P,// padding
                        const unsigned int CASC_IN_EN, // input comes from cascade boolean
                        const unsigned int CASC_OUT_EN) {//output goes to cascade boolean

        // If Cout%8 != 0 then unused weights are zeros
        // NOTE pipelining could causes an internal error on some M and N params
        for(unsigned int f = 0; f < CoutUp/8; f++) {
            for(unsigned int h = 0; h < outHeight; h+=1) chess_prepare_for_pipelining {
                for(unsigned int w = 0; w < outWidth; w+=4) chess_prepare_for_pipelining {
                    const unsigned int cond = ((outWidth % 4) == 0) || !((w+4) >= (outWidth));
                    applyKernel(actsIn, weightsIn, actsOut,
                                M, N, Cin, Cout, F, S, P,
                                CASC_IN_EN, CASC_OUT_EN, cond);
                }

                // Because of the convolution, we skip the last pixels
                // as the resulting image is smaller
                skip_end_a(actsIn, MR, outWidth);
            }

            // Prepare pointers for next iteration
            reset_a(actsIn);
            seek_next_w(weightsIn, F, CinUp);
        }
    }
}
