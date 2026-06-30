/* Copyright (C) 2019-2022 Xilinx, Inc.
Copyright (C) 2022-2025 Advanced Micro Devices, Inc.
SPDX-License-Identifier: LicenseRef-AMD-Proprietary */

// #include "adf.h"
#include "../inc/funcs_gelu.h"
#include "../inc/funcs_mmul.h"
#include "../inc/lut_group1.h"
#include "aie_api/aie.hpp"

void group1_init() {
  aie::tile::current().set_saturation(aie::saturation_mode::saturate);
  aie::tile::current().set_rounding(aie::rounding_mode::symmetric_inf);
}

// #define GROUP_1_INST(id)  void group1_ ## id ## _kernel(	adf::input_async_buffer<  int16_t,  adf::extents<8*512>>   & __restrict din, \
// 													        adf::output_buffer< int16_t,  adf::extents<16>>      & __restrict yout) \

#define GROUP_1_INST(id)                                                       \
  void group1_##id##_kernel(int16_t *__restrict din,                           \
                            int16_t *__restrict yout) {                        \
    int16_t *BA_L0 = magika_1_##id;                                            \
    int16_t *BA_B = (BA_L0 + 512);                                             \
    int16_t *BA_C0 = (BA_B + 16);                                              \
    int16_t *BA_Y = (BA_C0 + 256 * 6 * 5);                                     \
    int16_t *BA_L1 = (BA_Y + 508 * 12 + 16);                                   \
    int16_t *BA_C1 = (BA_L1 + 512);                                            \
    conv1d_12x8x512_init(BA_B, BA_Y);                                          \
    int16_t *pc0 = BA_C0;                                                      \
    int16_t *pc1 = BA_C1;                                                      \
    for (int igrp = 0; igrp < (256 / 8) - 1; igrp++) {                         \
      conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                  \
      pc0 += 12 * 8 * 5 / 2;                                                   \
      pc1 += 12 * 8 * 5 / 2;                                                   \
    }                                                                          \
    conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                    \
    gelu_max_12x508(BA_Y, din, BA_L0, BA_L1, yout);                            \
  }

// din += 8 * 512;                                                            \

// #define GROUP_1_CMB2(id)  void group1_ ## id ## _kernel(	adf::input_async_buffer<  int16_t,  adf::extents<8*512>>   & __restrict din, \
// 															adf::input_async_buffer<  int16_t,  adf::extents<16>>      & __restrict din2, \
// 													        adf::output_buffer< int16_t,  adf::extents<32>>      & __restrict yout) \

#define GROUP_1_CMB2(id)                                                       \
  void group1_##id##_kernel(int16_t *__restrict din, int16_t *__restrict din2, \
                            int16_t *__restrict yout) {                        \
    int16_t *BA_L0 = magika_1_##id;                                            \
    int16_t *BA_B = (BA_L0 + 512);                                             \
    int16_t *BA_C0 = (BA_B + 16);                                              \
    int16_t *BA_Y = (BA_C0 + 256 * 6 * 5);                                     \
    int16_t *BA_L1 = (BA_Y + 508 * 12 + 16);                                   \
    int16_t *BA_C1 = (BA_L1 + 512);                                            \
    conv1d_12x8x512_init(BA_B, BA_Y);                                          \
    int16_t *pc0 = BA_C0;                                                      \
    int16_t *pc1 = BA_C1;                                                      \
    for (int igrp = 0; igrp < (256 / 8) - 1; igrp++) {                         \
      conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                  \
      pc0 += 12 * 8 * 5 / 2;                                                   \
      pc1 += 12 * 8 * 5 / 2;                                                   \
    }                                                                          \
    conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                    \
    gelu_max_12x508(BA_Y, din, BA_L0, BA_L1, yout + 16);                       \
    *((v16int16 *)yout) = *((v16int16 *)din2);                                 \
  }

// din += 8 * 512;                                                            \

// #define GROUP_1_CMB3(id)  void group1_ ## id ## _kernel(	adf::input_async_buffer<  int16_t,  adf::extents<8*512>>   & __restrict din, \
// 															adf::input_async_buffer<  int16_t,  adf::extents<16>>      & __restrict din2, \
// 															adf::input_async_buffer<  int16_t,  adf::extents<16>>      & __restrict din3, \
// 													        adf::output_buffer< int16_t,  adf::extents<48>>      & __restrict yout) \

#define GROUP_1_CMB3(id)                                                       \
  void group1_##id##_kernel(int16_t *__restrict din, int16_t *__restrict din2, \
                            int16_t *__restrict din3,                          \
                            int16_t *__restrict yout) {                        \
    int16_t *BA_L0 = magika_1_##id;                                            \
    int16_t *BA_B = (BA_L0 + 512);                                             \
    int16_t *BA_C0 = (BA_B + 16);                                              \
    int16_t *BA_Y = (BA_C0 + 256 * 6 * 5);                                     \
    int16_t *BA_L1 = (BA_Y + 508 * 12 + 16);                                   \
    int16_t *BA_C1 = (BA_L1 + 512);                                            \
    conv1d_12x8x512_init(BA_B, BA_Y);                                          \
    int16_t *pc0 = BA_C0;                                                      \
    int16_t *pc1 = BA_C1;                                                      \
    for (int igrp = 0; igrp < (256 / 8) - 1; igrp++) {                         \
      conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                  \
      pc0 += 12 * 8 * 5 / 2;                                                   \
      pc1 += 12 * 8 * 5 / 2;                                                   \
    }                                                                          \
    conv1d_12x8x512x5(din, pc0, pc1, BA_Y);                                    \
    gelu_max_12x508(BA_Y, din, BA_L0, BA_L1, yout + 16);                       \
    *((v16int16 *)yout) = *((v16int16 *)din2);                                 \
    *((v16int16 *)(yout + 32)) = *((v16int16 *)din3);                          \
  }

// din += 8 * 512;                                                            \

//-----------------------8->3
GROUP_1_INST(00)
GROUP_1_CMB3(01)
GROUP_1_INST(02)

GROUP_1_INST(03)
GROUP_1_CMB3(04)
GROUP_1_INST(05)

GROUP_1_INST(06)
GROUP_1_CMB2(07)

//-----------------------8->3
GROUP_1_INST(08)
GROUP_1_CMB3(09)
GROUP_1_INST(10)

GROUP_1_INST(11)
GROUP_1_CMB3(12)
GROUP_1_INST(13)

GROUP_1_INST(14)
GROUP_1_CMB2(15)

//-----------------------8->3
GROUP_1_INST(16)
GROUP_1_CMB3(17)
GROUP_1_INST(18)

GROUP_1_INST(19)
GROUP_1_CMB3(20)
GROUP_1_INST(21)

GROUP_1_INST(22)
GROUP_1_CMB2(23)

//-----------------------8->3
GROUP_1_INST(24)
GROUP_1_CMB3(25)
GROUP_1_INST(26)

GROUP_1_INST(27)
GROUP_1_CMB3(28)
GROUP_1_INST(29)

GROUP_1_INST(30)
GROUP_1_CMB2(31)

//-----------------------8->3
GROUP_1_INST(32)
GROUP_1_CMB3(33)
GROUP_1_INST(34)

GROUP_1_INST(35)
GROUP_1_CMB3(36)
GROUP_1_INST(37)

GROUP_1_INST(38)
GROUP_1_CMB2(39)

//-----------------------3->1
GROUP_1_INST(40)
GROUP_1_CMB3(41)
GROUP_1_INST(42)
