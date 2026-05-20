//===- yolo_m10_linear_gemm.cc ---------------------------------*- C++ -*-===//
//
// Scalar i8 Gemm 1280 → 2 for the yolo26n-cls binary classifier head.
// Weights are stored in raw `shape_2x1280` layout (row-major, no OIYX
// packing — manifest weights_layout="shape_2x1280"). Each output is
//   out[o] = SRS_i8(sum_d wts[o, d] * in[d] + bias[o], right_shift=10)
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

extern "C" {

void yolo_m10_linear_gemm_i8_i8(
    int8_t *in_vec,            // (in_dim,)
    int8_t *wts,               // (out_dim, in_dim) row-major flat
    int32_t *bias,             // (out_dim,)
    int8_t *out_vec,           // (out_dim,)
    const int32_t in_dim,       // 1280
    const int32_t out_dim,      // 2
    const int32_t right_shift) {  // 10
  event0();

  for (int o = 0; o < out_dim; o++) {
    int32_t sum = bias[o];
    for (int d = 0; d < in_dim; d++) {
      sum += (int32_t)wts[o * in_dim + d] * (int32_t)in_vec[d];
    }
    int32_t s = banker_srs(sum, right_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    out_vec[o] = (int8_t)s;
  }

  event1();
}

} // extern "C"
