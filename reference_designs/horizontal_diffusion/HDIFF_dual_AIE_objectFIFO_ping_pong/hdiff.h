//===- hdiff.h --------------------------------------------------*- C++ -*-===//
//
// (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET
//
// This file is licensed under the MIT License.
// SPDX-License-Identifier: MIT
//
//
//===----------------------------------------------------------------------===//

extern "C" {
void hdiff_lap(int32_t *restrict row0, int32_t *restrict row1,
               int32_t *restrict row2, int32_t *restrict row3,
               int32_t *restrict row4, int32_t *restrict out_flux1,
               int32_t *restrict out_flux2, int32_t *restrict out_flux3,
               int32_t *restrict out_flux4);
void hdiff_flux(int32_t *restrict row1, int32_t *restrict row2,
                int32_t *restrict row3, int32_t *restrict flux_forward1,
                int32_t *restrict flux_forward2,
                int32_t *restrict flux_forward3,
                int32_t *restrict flux_forward4, int32_t *restrict out);

void hdiff_lap_fp32(float *restrict row0, float *restrict row1,
                    float *restrict row2, float *restrict row3,
                    float *restrict row4, float *restrict out_flux1,
                    float *restrict out_flux2, float *restrict out_flux3,
                    float *restrict out_flux4);
void hdiff_flux_fp32(float *restrict row1, float *restrict row2,
                     float *restrict row3, float *restrict flux_forward1,
                     float *restrict flux_forward2,
                     float *restrict flux_forward3,
                     float *restrict flux_forward4, float *restrict out);
}
