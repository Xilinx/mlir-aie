//===- passThrough.cc -------------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// #define __AIENGINE__ 1
#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

// template <typename T, int N>
// __attribute__((noinline)) void passThrough_aie(T *restrict in, T *restrict out,
//                                                const int32_t height,
//                                                const int32_t width) {
//   event0();

//   v64uint8 *restrict outPtr = (v64uint8 *)out;
//   v64uint8 *restrict inPtr = (v64uint8 *)in;

//   for (int j = 0; j < (height * width); j += N) // Nx samples per loop
//     chess_prepare_for_pipelining chess_loop_range(6, ) { *                                                                                                                                          outPtr++ = *inPtr++; }

//   event1();
// }

template<typename T>
__attribute__((noinline)) void passThrough_simple(uint32_t *restrict in, uint32_t*restrict out, const int32_t size){
  
  // event0()  ;

  for( int32_t j = 0; j < size; j++){
    *out = *in;
    out++;
    in++;
  }
  // event1();
}
bool parity(uint32_t n) {
    uint32_t p = 0;
    while (n) {
        p += n & 1;
        n >>= 1;
    }
    return (p % 2) == 0;
}
uint32_t control_packet_gen(int32_t stream_id, int32_t operation, int32_t beats, int32_t address){
  //operation: 0 read, 1 write
  uint32_t control_packet =
        stream_id << 24 | operation << 22 | beats << 20 | address;
  control_packet |= (0x1 & parity(control_packet)) << 31;

  return control_packet;
}

extern "C" {

void passThroughTest(uint32_t *in, uint32_t *out, 
                  int32_t buffer_size, int32_t total_passThrough_size,
                  int32_t in_buffer_prod_lock, int32_t in_buffer_con_lock,
                  int32_t out_buffer_prod_lock, int32_t out_buffer_con_lock,


                  int32_t* CT2_control_out_buffer, int32_t* CT2_control_res_buffer,
                  int32_t CT2_control_out_prod_lock, int32_t CT2_control_out_con_lock,
                  int32_t CT2_control_in_prod_lock, int32_t CT2_control_in_con_lock
                  
){
  // assume divisible and multiple of two
  for(uint32_t i = 0; i < (total_passThrough_size /buffer_size ); i += 2){

    acquire_greater_equal(in_buffer_con_lock, 1);
    if(i == 0){

        acquire_greater_equal(CT2_control_out_prod_lock, 1);
        *CT2_control_out_buffer = control_packet_gen(14, 1,0,0x001D044);
        release(CT2_control_out_con_lock, 1);

        acquire_greater_equal(CT2_control_in_con_lock, 1); // stuck here, result never return back from packet_id 14, bug in packet_flow?
        *(in) = *CT2_control_res_buffer;
        release(CT2_control_in_prod_lock, 1);
    }else{
       
    }

    // *in = 0x10001;
    acquire_greater_equal(out_buffer_prod_lock, 1);
    passThrough_simple<uint32_t>(in, out, buffer_size);
    release(in_buffer_prod_lock, 1);
    release(out_buffer_con_lock, 1);

  


    acquire_greater_equal(in_buffer_con_lock, 1);
    acquire_greater_equal(out_buffer_prod_lock, 1);
    passThrough_simple<uint32_t>(in+buffer_size, out+buffer_size, buffer_size);
    release(in_buffer_prod_lock, 1);
    release(out_buffer_con_lock, 1);

  }


}


// // #endif
// void passThroughLine_float_0(float *in, float *out, int32_t lineWidth) {
//   passThrough_simple<float>( in, out, lineWidth);
// }
// void passThroughLine_float_1(float *in, float *out, int32_t lineWidth) {
//   passThrough_simple<float>( in, out, lineWidth);
// }

// void passThroughLine_float_2(float *in, float *out, int32_t lineWidth) {
//   passThrough_simple<float>( in, out, lineWidth);
// }

// void passThroughLine_float_3(float *in, float *out, int32_t lineWidth) {
//   passThrough_simple<float>( in, out, lineWidth);
// }



} // extern "C"
