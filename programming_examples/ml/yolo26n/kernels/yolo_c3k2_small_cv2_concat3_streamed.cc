//===- yolo_c3k2_small_cv2_concat3_streamed.cc ----------------*- C++ -*-===//
//
// Chunked-OC variant of yolo_c3k2_small_cv2_concat3. Three concatenated
// input rows (top/bot/m0 from upstream), chunk of OIYXI8O8 weights for
// chunk_oc output channels, full bias, SiLU LUT. Each call writes
// chunk_oc channels into the corresponding slice of the output row.
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include <aie_api/aie.hpp>

#ifndef KERNEL_SUFFIX
#define KERNEL_SUFFIX
#endif
#define _PASTE(a, b) a##b
#define _MAKE(name, suffix) _PASTE(name, suffix)
#define KERNEL_NAME(base) _MAKE(base, KERNEL_SUFFIX)

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int32_t banker_srs(int32_t sum, int32_t rs) {
  return (sum + (1 << (rs - 1)) - 1 + ((sum >> rs) & 1)) >> rs;
}

static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void KERNEL_NAME(yolo_c3k2_small_cv2_concat3_streamed_silu_bias_i8_i8)(
    int8_t *in_top,
    int8_t *in_bot,
    int8_t *in_m0,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *output,
    const int32_t input_width,
    const int32_t c,              // per-input channel count (top/bot/m0 width)
    const int32_t output_channels,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
  event0();

  const int32_t chunk_oc = output_channels / n_chunks;
  const int32_t three_c = 3 * c;
  const int32_t bias_offset = chunk_idx * chunk_oc;
  const int32_t out_offset = chunk_idx * chunk_oc;

  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[bias_offset + co];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < three_c; ic++) {
        int8_t a = (ic < c) ? in_top[x * c + ic]
                : (ic < 2 * c) ? in_bot[x * c + (ic - c)]
                               : in_m0[x * c + (ic - 2 * c)];
        sum += a * wts_chunk[wts_chunk_idx_1x1(co, ic, three_c)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      output[x * output_channels + (out_offset + co)] = silu_lut[s + 128];
    }
  }

  event1();
}

} // extern "C"
