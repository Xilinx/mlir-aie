//===- yolo_m9_cv1_split.cc -----------------------------------*- C++ -*-===//
//
// Streamed (chunked-OC) variant of m9's cv1 1x1 INT8 conv. The full weight
// tensor for 256→256 OIYXI8O8 is 64KB — too large to co-exist with input/
// output fifos on a single AIE2P tile (64KB L1). So we stream weights from
// a memtile in N chunks; this kernel is called n_chunks times per row, each
// call processing chunk_oc = twoc/n_chunks output channels from one chunk.
// SiLU + bias fused; channel-split output (top = chans [0, c), bot = [c, 2c))
// for the PSA topology (top → residual skip, bot → qkv + proj_cv2 skip).
//
// Symbol is m9-specific (no KERNEL_SUFFIX mangling); m9 is the sole consumer.
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

// Chunk is OIYXI8O8 packed for (chunk_oc, in_c, 1, 1).
static inline int wts_chunk_idx_1x1(int chunk_oc_full, int ic_full, int in_c) {
  int oc_t = chunk_oc_full >> 3;
  int oc_i = chunk_oc_full & 7;
  int ic_t = ic_full >> 3;
  int ic_i = ic_full & 7;
  return ((oc_t * (in_c >> 3) + ic_t) << 6) + ic_i * 8 + oc_i;
}

extern "C" {

void yolo_m9_cv1_split_silu_bias_i8_i8(
    int8_t *in_row,
    int8_t *wts_chunk,
    int32_t *bias_full,
    int8_t *silu_lut,
    int8_t *out_top,
    int8_t *out_bot,
    const int32_t input_width,
    const int32_t input_channels,
    const int32_t twoc,
    const int32_t n_chunks,
    const int32_t chunk_idx,
    const int32_t right_shift) {
#ifdef NOOP_KERNEL
  return;
#endif
  event0();

  const int32_t chunk_oc = twoc / n_chunks;
  const int32_t c = twoc >> 1;
  const int32_t chunks_per_half = n_chunks >> 1;
  const bool is_top = (chunk_idx < chunks_per_half);
  const int32_t dst_oc_offset = is_top
      ? chunk_idx * chunk_oc
      : (chunk_idx - chunks_per_half) * chunk_oc;
  const int32_t bias_offset = chunk_idx * chunk_oc;

  for (int co = 0; co < chunk_oc; co++) {
    const int32_t binit = bias_full[bias_offset + co];
    for (int x = 0; x < input_width; x++) {
      int32_t sum = binit;
      for (int ic = 0; ic < input_channels; ic++) {
        sum += in_row[x * input_channels + ic] *
               wts_chunk[wts_chunk_idx_1x1(co, ic, input_channels)];
      }
      int32_t s = banker_srs(sum, right_shift);
      s = s > I8_MAX ? I8_MAX : (s < I8_MIN ? I8_MIN : s);
      int8_t silu = silu_lut[s + 128];
      const int idx = x * c + (dst_oc_offset + co);
      if (is_top) {
        out_top[idx] = silu;
      } else {
        out_bot[idx] = silu;
      }
    }
  }

  event1();
}

} // extern "C"
