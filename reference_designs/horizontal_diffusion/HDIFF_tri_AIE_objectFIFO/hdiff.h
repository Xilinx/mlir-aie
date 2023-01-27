/*  (c) 2023 SAFARI Research Group at ETH Zurich, Gagandeep Singh, D-ITET             */

//
//===----------------------------------------------------------------------===//

// #include <stdint.h>

extern "C" {
void hdiff_lap(int32_t* restrict row0, int32_t* restrict row1,int32_t* restrict row2,int32_t* restrict row3,int32_t* restrict row4, int32_t* restrict out_flux1, int32_t* restrict out_flux2, int32_t* restrict out_flux3, int32_t* restrict out_flux4 );
void hdiff_flux1(int32_t* restrict row1, int32_t* restrict row2,int32_t* restrict row3, int32_t* restrict flux_forward1,int32_t* restrict flux_forward2,int32_t* restrict flux_forward3,int32_t* restrict flux_forward4,  int32_t* restrict flux_inter1,int32_t* restrict flux_inter2,int32_t* restrict flux_inter3,int32_t* restrict flux_inter4,int32_t* restrict flux_inter5);
void hdiff_flux2(int32_t* restrict flux_inter1,int32_t* restrict flux_inter2,int32_t* restrict flux_inter3,int32_t* restrict flux_inter4,int32_t* restrict flux_inter5,  int32_t * restrict out);

}

