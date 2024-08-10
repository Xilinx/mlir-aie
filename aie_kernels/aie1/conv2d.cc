#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <aie_api/aie.hpp>
#include "conv2d_params.h"

#define REL_WRITE 0
#define REL_READ 1

namespace S1 {

#define X_C64_D8 0x38303830
#define Z_C8_D8_I_L 0x44440000
#define Z_C8_D8_I_U 0x88884444
#define Z_C8_D8_I_UU 0xcccc8888

    inline __attribute__ ((always_inline))
    v32int8 load_w(int8_t* weightsIn) {
        v32int8 v = *(v32int8*)weightsIn;
        return v;
    }

    inline __attribute__ ((always_inline))
    v32uint8 load_a(uint8_t* actsIn, v32uint8 v, const unsigned int i) {
        v = upd_v(v, i, *(v16uint8*)actsIn);
        return v;
    }

    inline __attribute__ ((always_inline))
    uint8_t* wb_0(uint8_t* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        *(v16uint8*)actsOut = out;
        actsOut += (1 + 1 * keep_acc1) * 16;
        return actsOut;
    }

    inline __attribute__ ((always_inline))
    uint8_t* wb_1(uint8_t* actsOut, v16acc48 acc, const unsigned int keep_acc1) {
        v16uint8 out = ubsrs(acc, SHIFT);
        actsOut += (1 * keep_acc1) * 16;
        *(v16uint8*)actsOut = out;
        actsOut -= (1 * keep_acc1) * 16;
        return actsOut;
    }

    inline __attribute__((always_inline))
    void applyKernel(uint8_t* &actsIn, uint8_t* actsIn_head,
                     int8_t* &weightsIn, int8_t* weightsIn_head,
                     uint8_t* &actsOut,
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
        const unsigned int N_WRITES = ((F+1) / 2) * (CinUp/8);
        const unsigned int W_N_WRITES = F*F*(CinUp/8);

        v16acc48 acc0 = undef_v16acc48();
        v16acc48 acc1 = undef_v16acc48();

        if(CASC_IN_EN == 1) {
            acc0 = concat(get_scd(), get_scd());
            acc1 = concat(get_scd(), get_scd());
        }

        v32int8 chess_storage(wr0) w0 = undef_v32int8();
        v32int8 chess_storage(wr1) w1 = undef_v32int8();
        v32int8 chess_storage(wd0) w2 = undef_v32int8();
        v32int8 chess_storage(wd1) w3 = undef_v32int8();

        v64int8 weightBuff0 = undef_v64int8();

        v32uint8 chess_storage(wc0) v0 = undef_v32uint8();
        v32uint8 chess_storage(wc1) v1 = undef_v32uint8();

        for(unsigned int c = 0; c < CinUp/8; c++) {
            for(unsigned int y = 0; y < F; y++)  {
                for(unsigned int x = 0; x < F; x++)  {
                    const unsigned int pat = x % 2;
                    const unsigned int wPat = (x + y*F + c*F*F) % 2;
                    const unsigned int enRewind = (x == (F-1)) && (y == (F-1)) && (c == (CinUp/8 - 1));

                    // Load weights and increment pointer manually
                    v32int8 l0 = load_w(weightsIn);
                    weightsIn += 32;

                    v32int8 l1 = load_w(weightsIn);
                    weightsIn += 32 - ((8/4)*F*F*(CinUp/8)) * enRewind * 32;

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
                        const unsigned int nextRow = (y < (F-1)) && (x == (F-2));
                        const unsigned int nextChannels = (c < (CinUp/8 - 1)) && (y == (F-1)) && (x == (F-2));
                        const unsigned int enRewindA = (x == (F-2)) && (y == (F-1)) && (c == (CinUp/8 - 1));

                        if((N_WRITES % 2) == 0) {
                            const unsigned int regWrite = (((x+1)/2)) % 2;
                            if(regWrite == 0) {
                                v0 = load_a(actsIn, v0, 0);
                                actsIn += 16; // Manually increment actsIn pointer
                                v0 = load_a(actsIn, v0, 1);
                                if (nextRow) actsIn -= ((1+F/2 - (MR/2)) * nextRow) * 16;
                                if (nextChannels) actsIn -= ((1+F/2 - (MR/2)*(N-(F-1))) * nextChannels) * 16;
                                if (enRewindA) actsIn -= (1+F/2 + (CinUp/8-1) * MR/2*N + (F-1)*MR/2 - 2) * enRewindA * 16;
                            } else {
                                v1 = load_a(actsIn, v1, 0);
                                actsIn += 16; // Manually increment actsIn pointer
                                v1 = load_a(actsIn, v1, 1);
                                if (nextRow) actsIn -= ((1+F/2 - (MR/2)) * nextRow) * 16;
                                if (nextChannels) actsIn -= ((1+F/2 - (MR/2)*(N-(F-1))) * nextChannels) * 16;
                                if (enRewindA) actsIn -= (1+F/2 + (CinUp/8-1) * MR/2*N + (F-1)*MR/2 - 2) * enRewindA * 16;
                            }
                        } else {
                            const unsigned int regWrite = (((x+1)/2) + (y%2) + (c%2)) % 2;
                            if((x == 0) && (y == 0) && (c == 0)) {
                                v1 = load_a(actsIn, v1, 0);
                                actsIn += 16; // Manually increment actsIn pointer
                                v1 = load_a(actsIn, v1, 1);
                                if (nextRow) actsIn -= ((1+F/2 - (MR/2)) * nextRow) * 16;
                                if (nextChannels) actsIn -= ((1+F/2 - (MR/2)*(N-(F-1))) * nextChannels) * 16;
                                if (enRewindA) actsIn -= (1+F/2 + (CinUp/8-1) * MR/2*N + (F-1)*MR/2 - 2) * enRewindA * 16;
                                v0 = v1;
                            } else {
                                if(regWrite == 0) {
                                    v0 = load_a(actsIn, v0, 0);
                                    actsIn += 16; // Manually increment actsIn pointer
                                    v0 = load_a(actsIn, v0, 1);
                                    if (nextRow) actsIn -= ((1+F/2 - (MR/2)) * nextRow) * 16;
                                    if (nextChannels) actsIn -= ((1+F/2 - (MR/2)*(N-(F-1))) * nextChannels) * 16;
                                    if (enRewindA) actsIn -= (1+F/2 + (CinUp/8-1) * MR/2*N + (F-1)*MR/2 - 2) * enRewindA * 16;
                                } else {
                                    v1 = load_a(actsIn, v1, 0);
                                    actsIn += 16; // Manually increment actsIn pointer
                                    v1 = load_a(actsIn, v1, 1);
                                    if (nextRow) actsIn -= ((1+F/2 - (MR/2)) * nextRow) * 16;
                                    if (nextChannels) actsIn -= ((1+F/2 - (MR/2)*(N-(F-1))) * nextChannels) * 16;
                                    if (enRewindA) actsIn -= (1+F/2 + (CinUp/8-1) * MR/2*N + (F-1)*MR/2 - 2) * enRewindA * 16;
                                }
                            }
                        }
                    }

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

        if(CASC_OUT_EN == 0) {
            actsOut = wb_1(actsOut, acc1, store_acc1);
            actsOut = wb_0(actsOut, acc0, store_acc1);
        } else {
            put_mcd(ext_lo(acc0));
            put_mcd(ext_hi(acc0));

            put_mcd(ext_lo(acc1));
            put_mcd(ext_hi(acc1));
        }
    }

    inline __attribute__ ((always_inline))
    void conv2d_int8_S1(uint8_t* restrict actsIn,
                        int8_t* restrict weightsIn,
                        uint8_t* restrict actsOut,
                        const unsigned int M,
                        const unsigned int N,
                        const unsigned int Cin,
                        const unsigned int Cout,
                        const unsigned int F,
                        const unsigned int S,
                        const unsigned int P,
                        const unsigned int CASC_IN_EN,
                        const unsigned int CASC_OUT_EN) {

        uint8_t* actsIn_head = actsIn;
        int8_t* weightsIn_head = weightsIn;

        for(unsigned int f = 0; f < CoutUp/8; f++) {
            for(unsigned int h = 0; h < outHeight; h+=1) chess_prepare_for_pipelining {
                for(unsigned int w = 0; w < outWidth; w+=4) chess_prepare_for_pipelining {
                    const unsigned int cond = ((outWidth % 4) == 0) || !((w+4) >= (outWidth));

                    uint8_t* actsIn_copy = actsIn;  // Create a copy of actsIn
                    int8_t* weightsIn_copy = weightsIn;  // Create a copy of weightsIn
                    uint8_t* actsOut_copy = actsOut;  // Create a copy of actsOut

                    applyKernel(actsIn_copy, actsIn_head, weightsIn_copy, weightsIn_head, actsOut_copy,
                                M, N, Cin, Cout, F, S, P,
                                CASC_IN_EN, CASC_OUT_EN, cond);

                    actsIn = actsIn_copy;  // Update the original pointer with the modified copy
                    weightsIn = weightsIn_copy;  // Update the original pointer with the modified copy
                    actsOut = actsOut_copy;  // Update the original pointer with the modified copy
                }

                const unsigned int read = 4*((outWidth+3)/4);
                const unsigned int remaining = ((MR - read) / 2) * 16;
                actsIn += remaining;
            }

            actsIn = actsIn_head;
            weightsIn += (8/4)*F*F*(CinUp/8) * 32;
        }
    }
}

extern "C" {

    void conv2d_int8(uint8_t* restrict actsIn, 
                        int8_t* restrict weightsIn, 
                        uint8_t* restrict actsOut,
                        const unsigned int M,
                        const unsigned int N,
                        const unsigned int Cin,
                        const unsigned int Cout,
                        const unsigned int F,
                        const unsigned int S,
                        const unsigned int P,// This is a template parameter
                        const unsigned int CASC_IN_EN,
                        const unsigned int CASC_OUT_EN ) {
    S1::conv2d_int8_S1( actsIn, 
                    weightsIn, 
                    actsOut,
                    M,
                    N,
                    Cin,
                    Cout,
                    F,
                    S,
                    P,// This is a template parameter
                    CASC_IN_EN,
                    CASC_OUT_EN) ;
    }
}
