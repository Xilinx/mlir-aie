//===- yolo_m9_attn_scale.cc ---------------------------------*- C++ -*-===//
//
// In-place integer scale-and-requantize step for the PSA attention pipe.
// ONNX has a Mul-by-constant node between MatMul output (i8, scale 2^-3)
// and Softmax input (i8, scale 2^-5). The constant is the quantized form
// of 1/sqrt(d) where d=key_dim=32, encoded as quantized_value=91 with
// scale 2^-9 (= 91/512 ≈ 0.1777 ≈ 1/sqrt(32)=0.1768).
//
// The integer equivalent of the DQ → Mul → QL chain is:
//   scaled_i8 = banker_srs(raw_i8 * mul_int, mul_shift)
// where mul_int=91 and mul_shift=7 for m9 (derived from
// 2^-3 * (91 * 2^-9) / 2^-5 = 91 / 128 = 91 / 2^7).
//
// Operates in-place on one row (N values) of a (chunk_rows, N) chunk.
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

void yolo_m9_attn_scale_row_i8_i8(
    int8_t *chunk_io,           // (chunk_rows, N)
    const int32_t chunk_row,
    const int32_t N,
    const int32_t mul_int,
    const int32_t mul_shift) {
  event0();

  int8_t *row = chunk_io + chunk_row * N;

  for (int j = 0; j < N; j++) {
    int32_t prod = (int32_t)row[j] * mul_int;
    int32_t s = banker_srs(prod, mul_shift);
    if (s > I8_MAX) s = I8_MAX;
    if (s < I8_MIN) s = I8_MIN;
    row[j] = (int8_t)s;
  }

  event1();
}

} // extern "C"
