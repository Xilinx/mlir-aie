//===- llama_gemm_int8_srs_tiled_layer_mh.cc --------------*- C++ -*-===//
// Phase 7a multi-head GEMM entries:
//   - K=2048, N_TILE=4, kOutDim=QD=2048: q_proj with per-Q-head requant.
//     Slot prefix (448 B) holds act_scale + 32 q_inv_outs +
//     [32 q_out_scale | sv_inv_out_scale] pairs (mirrored to out tail
//     for downstream rope_mh / q-splitter / flowkv_mh).
//   - K=2048, N_TILE=4, kOutDim=D=2048: o_proj (standard v2 perchan, 64 B
//     prefix). af input already requanted to one global o_act_scale by
//     af_concat, so this is the same shape as existing _perchan_v2_o but
//     at K=2048 instead of K=64.
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

// q_proj-mh prefix layout (448 B, 64-aligned):
//   [0..4]    act_scale fp32
//   [4..8]    spare
//   [8..136]  32 q_inv_outs fp32 (kernel reads to requant per head)
//   [136..392] 32 * 8 B [q_out_scale, sv_inv_out_scale] (mirrored to out tail)
//   [392..448] padding
constexpr int kMhPrefix = 448;
constexpr int kMhTailOffset = 136;
constexpr int kMhTailBytes = 32 * 8; // 256

// Standard v2 (used by o_proj-mh at K=2048).
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_impl(int8_t *restrict act,
                                             int8_t *restrict w_tile,
                                             int8_t *restrict out_tile) {
  static_assert(kPrefix >= 8, "prefix must hold at least 8 B of scales");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale, inv_out_scale;
  memcpy(&act_scale, w_tile, 4);
  memcpy(&inv_out_scale, w_tile + 4, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// Multi-head q_proj: per-Q-head requant. N_TILE=4 divides HEAD_DIM=64
// evenly, so all 4 output rows in a tile belong to the same Q head.
// head_idx = tile_idx / (HEAD_DIM / N_TILE) = tile_idx / 16.
template <int kK, int kNTile, int kHD, int kNHeadsQ>
static inline void gemm_tile_perchan_v2_qmh_impl(int8_t *restrict act,
                                                 int8_t *restrict w_tile,
                                                 int8_t *restrict out_tile,
                                                 int8_t *restrict out_full_base,
                                                 int32_t tile_idx) {
  static_assert(kHD % kNTile == 0,
                "HEAD_DIM must be a multiple of N_TILE so each tile maps "
                "to exactly one head");
  static_assert(kMhPrefix % 64 == 0,
                "prefix must be a multiple of 64 for aligned weight loads");

  float act_scale;
  memcpy(&act_scale, w_tile, 4);

  // Per-head q_inv_outs live at prefix[8 .. 8 + kNHeadsQ*4].
  const float *q_inv_outs = reinterpret_cast<const float *>(w_tile + 8);

  int head_idx = tile_idx / (kHD / kNTile);
  float inv_out_scale = q_inv_outs[head_idx];

  // Mirror 256 B tail (per-head [q_out_scale, sv_inv_out_scale] pairs).
  // Done every tile call (idempotent, same as single-head v2_up_q pattern).
  {
    constexpr int kOutDim = kNHeadsQ * kHD;
    memcpy(out_full_base + kOutDim, w_tile + kMhTailOffset, kMhTailBytes);
  }

  int8_t *body = w_tile + kMhPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// Multi-head q_proj, activation-tail act_scale. Identical to the qmh impl
// except act_scale comes from the activation tail (act[kK..kK+4]) written by
// the upstream dyn rmsnorm, instead of the weight-slot prefix. Per-head
// q_inv_outs and the 256 B downstream tail still come from the weight prefix.
// Activation buffer must be int8[kK + 8].
template <int kK, int kNTile, int kHD, int kNHeadsQ>
static inline void gemm_tile_perchan_v2_qmh_acttail_impl(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_tile,
    int8_t *restrict out_full_base, int32_t tile_idx) {
  static_assert(kHD % kNTile == 0,
                "HEAD_DIM must be a multiple of N_TILE so each tile maps "
                "to exactly one head");
  static_assert(kMhPrefix % 64 == 0,
                "prefix must be a multiple of 64 for aligned weight loads");

  float act_scale;
  memcpy(&act_scale, act + kK, 4);

  // Per-head q_inv_outs live at prefix[8 .. 8 + kNHeadsQ*4].
  const float *q_inv_outs = reinterpret_cast<const float *>(w_tile + 8);

  int head_idx = tile_idx / (kHD / kNTile);
  float inv_out_scale = q_inv_outs[head_idx];

  // Mirror 256 B tail (per-head [q_out_scale, sv_inv_out_scale] pairs).
  {
    constexpr int kOutDim = kNHeadsQ * kHD;
    memcpy(out_full_base + kOutDim, w_tile + kMhTailOffset, kMhTailBytes);
  }

  int8_t *body = w_tile + kMhPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    float fp = (float)sum_i32 * act_scale * w_scales[n];
    float scaled = fp * inv_out_scale;

    int32_t r = (int32_t)(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
    if (r > I8_MAX)
      r = I8_MAX;
    if (r < I8_MIN)
      r = I8_MIN;
    out_tile[n] = (int8_t)r;
  }
}

// fp32-out variant: same MAC + bias + per-channel fp scale as
// gemm_tile_perchan_v2_impl, but writes the TRUE fp32 output (no
// inv_out_scale / round / clamp). Used by o_proj/down when the downstream
// rescale-add consumes the fp32 projection directly and computes the
// dynamic residual scale itself. out_tile is float* (already offset by the
// caller to tile_idx * kNTile).
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_fp32out_impl(int8_t *restrict act,
                                                     int8_t *restrict w_tile,
                                                     float *restrict out_tile) {
  static_assert(kPrefix >= 8, "prefix must hold at least 8 B of scales");
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 to keep aie::load_v<64> "
                "of the weights aligned");
  float act_scale;
  memcpy(&act_scale, w_tile, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0,
                "K must be a multiple of 64 (one aie::mac lane group)");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());

    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];

    out_tile[n] = (float)sum_i32 * act_scale * w_scales[n];
  }
}

// fp32-out, activation-tail act_scale variant: act is int8[kK+8] with the
// per-token scale in act[kK..kK+4] (the dyn-rmsnorm h1 tail). Used by k_proj
// and v_proj, whose input is h1 at the per-token dynamic act_scale1.
template <int kK, int kNTile, int kPrefix = 64>
static inline void gemm_tile_perchan_v2_fp32out_acttail_impl(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_tile) {
  static_assert(kPrefix % 64 == 0,
                "prefix must be a multiple of 64 for aligned weight loads");
  float act_scale;
  memcpy(&act_scale, act + kK, 4);

  int8_t *body = w_tile + kPrefix;
  constexpr int kVec = 64;
  static_assert(kK % kVec == 0, "K must be a multiple of 64");
  constexpr int kGroups = kK / kVec;

  const int8_t *weights = body;
  const int32_t *bias = reinterpret_cast<const int32_t *>(body + kNTile * kK);
  const float *w_scales =
      reinterpret_cast<const float *>(body + kNTile * kK + kNTile * 4);

  ::aie::set_saturation(aie::saturation_mode::saturate);
  ::aie::set_rounding(aie::rounding_mode::conv_even);

  for (int n = 0; n < kNTile; n++) {
    aie::accum<acc32, kVec> acc;
    acc.from_vector(aie::zeros<int32, kVec>());
    const int8_t *__restrict w_row = weights + n * kK;
    for (int g = 0; g < kGroups; g++) {
      aie::vector<int8, kVec> w_v = aie::load_v<kVec>(w_row + g * kVec);
      aie::vector<int8, kVec> x_v = aie::load_v<kVec>(act + g * kVec);
      acc = aie::mac(acc, w_v, x_v);
    }
    int32_t sum_i32 =
        aie::reduce_add(acc.template to_vector<int32>()) + bias[n];
    out_tile[n] = (float)sum_i32 * act_scale * w_scales[n];
  }
}

extern "C" {

// k_proj / v_proj: K=D=2048 inner; kOutDim=KV_DIM=512 outer (128 tiles).
// fp32 output, act_scale from the h1 activation tail. Same symbol for both
// (they differ only by the weight slot the host streams in).
void llama_gemm_tiled_layer_K2048_N4_perchan_fp32out_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_acttail_impl<2048, 4, 64>(
      act, w_tile, out_full + tile_idx * 4);
  event1();
}

// Multi-head o_proj, fp32 output. K=QD=2048 inner; kOutDim=D=2048 outer.
// Writes the true fp32 o_proj output for the rescale-add to consume.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh_fp32out(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_impl<2048, 4, 64>(act, w_tile,
                                                 out_full + tile_idx * 4);
  event1();
}

// o_proj-mh, fp32 out, act_scale from the af ACTIVATION tail (= the
// self-calibrated o_act_scale at af[QD]). Distinct C symbol from the k/v-proj
// fp32out_acttail (IRON requires one Kernel object per unique symbol, and the
// two have different output shapes: KV_DIM vs D).
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh_fp32out_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, float *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_fp32out_acttail_impl<2048, 4, 64>(
      act, w_tile, out_full + tile_idx * 4);
  event1();
}

// Multi-head q_proj, activation-tail act_scale variant. Same contract as
// ..._v2_up_q_mh but act is int8[2048+8] with the per-token scale in the tail.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q_mh_acttail(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_qmh_acttail_impl<2048, 4, 64, 32>(
      act, w_tile, out_full + tile_idx * 4, out_full, tile_idx);
  event1();
}

// Multi-head q_proj. tile_idx in [0, 512); each tile writes 4 i8 to
// out_full[tile_idx*4 .. tile_idx*4+4]. tile_idx==0 mirrors the 256 B
// scale tail into out_full[QD..QD+256] for rope_mh / q_split.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_up_q_mh(
    int8_t *restrict act, int8_t *restrict w_tile, int8_t *restrict out_full,
    int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_qmh_impl<2048, 4, 64, 32>(
      act, w_tile, out_full + tile_idx * 4, out_full, tile_idx);
  event1();
}

// Multi-head o_proj. K=QD=2048 inner; kOutDim=D=2048 outer.
void llama_gemm_tiled_layer_K2048_N4_perchan_v2_o_mh(int8_t *restrict act,
                                                     int8_t *restrict w_tile,
                                                     int8_t *restrict out_full,
                                                     int32_t tile_idx) {
  event0();
  gemm_tile_perchan_v2_impl<2048, 4, 64>(act, w_tile, out_full + tile_idx * 4);
  event1();
}

} // extern "C"
