#ifndef KERNEL_H
#define KERNEL_H

#include "cardano/window/window.h"
#include "cardano/stream/streams.h"
#include <cardano/redefine.h>
#include <cardano/intrinsics.h>

#include "avgpool.cc"

template<unsigned int M,
         unsigned int N,
         unsigned int C,
         unsigned int MP_S,
         unsigned int MP_W>
void avgpool_int8(input_window_uint8* restrict actsIn,
                  output_window_uint8* restrict actsOut) {
    avgpool_int8_7x7(actsIn, actsOut, M, N, C);
}

#endif
