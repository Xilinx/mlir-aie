//===- llama_gqa_glue.cc ------------------------------------*- C++ -*-===//
// Phase 7a glue kernels for the multi-head single-layer design.
//
// llama_q_split:
//   Reorder rope_mh's flat output (32 heads body || 32-head scale tail)
//   into 8 KV-group chunks of [4 Q heads body || 4 scale pairs].
//   Input  layout (QD + Tail = 2048 + 256 = 2304 B):
//     [h0..h31 body 2048 B] [s0..s31 tail 256 B]
//     where s_h = [q_scale_h fp32 | sv_inv_out_scale_h fp32]
//   Output layout (8 * 288 = 2304 B):
//     for g in 0..8:
//       out[g*288 .. g*288+256]   = body[g*256 .. g*256+256]      (4 Q heads)
//       out[g*288+256 .. g*288+288] = tail[g*32 .. g*32+32]       (4 scale
//       pairs)
//
// llama_af_concat:
//   Take the concatenated 8x256 = 2048 B int8 sv buffer (produced by
//   memtile concat of 8 attn workers' outputs), per-Q-head dequant +
//   global requant to o_act_scale.  Mirrors the numpy_layer_mh "af-concat"
//   step:
//     af_fp[h*64+j] = sv_in[h*64+j] * sv_out_scale[h]
//     af_out[h*64+j] = round(af_fp[h*64+j] * o_inv_act_scale)
//   scales buffer layout (192 B, host-packed):
//     scales[0..128]  = 32 sv_out_scales fp32
//     scales[128..132] = o_inv_act_scale fp32
//     scales[132..192] = padding
//===---------------------------------------------------------------------===//

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <string.h>

#ifndef LLAMA_GQA_HEAD_DIM
#define LLAMA_GQA_HEAD_DIM 64
#endif
#ifndef LLAMA_GQA_N_HEADS_Q
#define LLAMA_GQA_N_HEADS_Q 32
#endif
#ifndef LLAMA_GQA_N_HEADS_KV
#define LLAMA_GQA_N_HEADS_KV 8
#endif

static constexpr int kHD = LLAMA_GQA_HEAD_DIM;
static constexpr int kNHeadsQ = LLAMA_GQA_N_HEADS_Q;
static constexpr int kNHeadsKV = LLAMA_GQA_N_HEADS_KV;
static constexpr int kREP = kNHeadsQ / kNHeadsKV;
static constexpr int kQD = kNHeadsQ * kHD;             // 2048
static constexpr int kBodyChunk = kREP * kHD;          // 256
static constexpr int kTailChunk = kREP * 8;            // 32
static constexpr int kChunk = kBodyChunk + kTailChunk; // 288

static constexpr int32_t I8_MAX = 127;
static constexpr int32_t I8_MIN = -128;

static inline int8_t round_to_i8(float v) {
  int32_t r = (int32_t)(v + (v >= 0.0f ? 0.5f : -0.5f));
  if (r > I8_MAX)
    r = I8_MAX;
  if (r < I8_MIN)
    r = I8_MIN;
  return (int8_t)r;
}

// Trivial sv_lo + sv_hi -> sv_full memcpy worker. Needed because af_concat
// can't pull from 3 input fifos on one CT (2-in DMA channel cap), so we
// merge the two halves first.
extern "C" void llama_sv_merge(int8_t *restrict in_lo, int8_t *restrict in_hi,
                               int8_t *restrict out) {
  event0();
  constexpr int kHalf = (kNHeadsQ / 2) * kHD;
  memcpy(out, in_lo, kHalf);
  memcpy(out + kHalf, in_hi, kHalf);
  event1();
}

// Self-calibrating sv_merge: the selfcal flowkv writes each KV group's chunk
// as [REP*kHD body | REP*4 sv_out_scale] (272 B). The memtile join packs 4
// such chunks per half (in_lo/in_hi). This kernel DE-INTERLEAVES them into
//   out[0 .. kQD]                 = all 32 Q-head sv bodies, contiguous
//   out[kQD .. kQD + kNHeadsQ*4]  = all 32 per-Q-head sv_out_scales (fp32)
// so the downstream af_concat reads both bodies and sv_out_scales from this
// single buffer (no host af_scales fifo). Q-head order is preserved:
// KV group g holds Q heads [g*REP .. g*REP+REP).
static constexpr int kSvBody = kREP * kHD;        // 256 (one KV group's body)
static constexpr int kSvTail = kREP * 4;          // 16  (one KV group's scales)
static constexpr int kSvChunk = kSvBody + kSvTail; // 272
static constexpr int kSvScalesOff = kQD;          // scales region start in out
static constexpr int kSvHalfKV = kNHeadsKV / 2;   // 4

extern "C" void llama_sv_merge_selfcal(int8_t *restrict in_lo,
                                       int8_t *restrict in_hi,
                                       int8_t *restrict out) {
  event0();
  int8_t *scales_out = out + kSvScalesOff;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *src = (g < kSvHalfKV) ? in_lo : in_hi;
    int local = (g < kSvHalfKV) ? g : (g - kSvHalfKV);
    int8_t *chunk = src + local * kSvChunk;
    memcpy(out + g * kSvBody, chunk, kSvBody);            // body -> contiguous
    memcpy(scales_out + g * kSvTail, chunk + kSvBody, kSvTail);  // scales
  }
  event1();
}

// 2-output split: lower half (KV groups 0..kHalf) and upper half. Required
// to keep memtile DMA fanout under 4-out per stage; one memtile can't
// emit all 8 streams.
static constexpr int kHalfKV = kNHeadsKV / 2;       // 4
static constexpr int kHalfChunk = kHalfKV * kChunk; // 1152

extern "C" void llama_q_split(int8_t *restrict in_full, int8_t *restrict out_lo,
                              int8_t *restrict out_hi) {
  event0();
  int8_t *body = in_full;
  int8_t *tail = in_full + kQD;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kChunk;
    memcpy(dst, body + g * kBodyChunk, kBodyChunk);
    memcpy(dst + kBodyChunk, tail + g * kTailChunk, kTailChunk);
  }
  event1();
}

// qkv_combine (on-chip KV append): build per-head COMBINED chunks
//   [ q_chunk 288 B | k_fp 256 B | v_fp 256 B | cs 256 B ]   (1056 B/head)
// from rope's q output and a pre-joined kvcs buffer. Split into lo (heads
// 0..3) / hi (heads 4..7) like q_split, to keep memtile fanout <=4.
//
// qr layout (input, 2304 B): [QD body 2048 | 32-head scale tail 256]
//   q_chunk_g = [ qr_body[g*256 .. +256] | qr_tail[g*32 .. +32] ]   (288 B)
// kvcs layout (input, 1280 B, memtile-joined): [ kfp 512 | vfp 512 | cs 256 ]
//   kfp/vfp are 8 heads * 64 fp32 contiguous; head g = bytes g*256 .. +256.
//   cs (cos+sin bf16, 256 B) is shared across all heads (one decode pos).
// Sizes: kfp region = 8 heads * 64 fp32 = 2048 B; same for vfp. So
//   kvcs = [kfp 2048 | vfp 2048 | cs 256] = 4352 B; combined chunk = 1056 B.
static constexpr int kKfpHead = kHD * 4;     // 256 (64 fp32 for one head)
static constexpr int kCsBytes = 2 * kHD * 2; // 256 (cos+sin bf16)
static constexpr int kCombChunk = kChunk + 2 * kKfpHead + kCsBytes; // 1056

extern "C" void llama_qkv_combine(int8_t *restrict qr, int8_t *restrict kvcs,
                                  int8_t *restrict out_lo,
                                  int8_t *restrict out_hi) {
  event0();
  constexpr int kKfpRegion = kNHeadsKV * kKfpHead; // 8*256 = 2048
  int8_t *qr_body = qr;
  int8_t *qr_tail = qr + kQD;
  int8_t *kfp = kvcs;
  int8_t *vfp = kvcs + kKfpRegion;
  int8_t *cs = kvcs + 2 * kKfpRegion;
  for (int g = 0; g < kNHeadsKV; g++) {
    int8_t *base = (g < kHalfKV) ? out_lo : out_hi;
    int local = (g < kHalfKV) ? g : (g - kHalfKV);
    int8_t *dst = base + local * kCombChunk;
    // q_chunk: body (256) + tail (32)
    memcpy(dst, qr_body + g * kBodyChunk, kBodyChunk);
    memcpy(dst + kBodyChunk, qr_tail + g * kTailChunk, kTailChunk);
    // k_fp, v_fp, cs
    memcpy(dst + kChunk, kfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + kKfpHead, vfp + g * kKfpHead, kKfpHead);
    memcpy(dst + kChunk + 2 * kKfpHead, cs, kCsBytes);
  }
  event1();
}

// af_concat consumes the full merged sv buffer (2048 B = 32 heads * 64).
// Per-Q-head dequant (sv_out_scale[h]) + global requant (o_inv_act_scale).
extern "C" void llama_af_concat(int8_t *restrict af_in, int8_t *restrict scales,
                                int8_t *restrict af_out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  const float *sv_out_scales = reinterpret_cast<const float *>(scales);
  float o_inv_act_scale;
  memcpy(&o_inv_act_scale, scales + 128, 4);

  for (int h = 0; h < kNHeadsQ; h++) {
    float combined = sv_out_scales[h] * o_inv_act_scale;
    int8_t *src = af_in + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int j = 0; j < kHD; j++) {
      dst[j] = round_to_i8((float)src[j] * combined);
    }
  }
  event1();
}

// Pure IEEE fp32 reciprocal (Peano `/` is a HW approx, Bug 1). Matches the
// dyn-rmsnorm sw_recip so the numpy reference and device share the function.
static inline float sw_recip(float a) {
  int32_t bits;
  memcpy(&bits, &a, 4);
  bits = (int32_t)0x7EF477D5 - bits;
  float x;
  memcpy(&x, &bits, 4);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  x = x * (2.0f - a * x);
  return x;
}

// Self-calibrating af_concat: computes the global o_act_scale on-chip (absmax
// of af_fp = sv_in[h]*sv_out_scale[h] over the whole Q_DIM vector) instead of
// reading o_inv_act_scale from the host scales buffer. Two-pass; writes
// o_act_scale to af_out[kOutDim..kOutDim+4] (af_out is int8[kOutDim+8]) so the
// downstream o_proj reads its act_scale from the af tail. sv_out_scales still
// come from the host scales buffer (that's 1c). kOutDim = N_HEADS_Q*HEAD_DIM.
extern "C" void llama_af_concat_selfcal(int8_t *restrict af_in,
                                        int8_t *restrict scales,
                                        int8_t *restrict af_out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int kOutDim = kNHeadsQ * kHD;
  const float *sv_out_scales = reinterpret_cast<const float *>(scales);

  // Pass A: absmax of af_fp = src * sv_out_scale[h].
  float absmax = 0.0f;
  for (int h = 0; h < kNHeadsQ; h++) {
    float s = sv_out_scales[h];
    int8_t *src = af_in + h * kHD;
    for (int j = 0; j < kHD; j++) {
      float v = (float)src[j] * s;
      float a = v >= 0.0f ? v : -v;
      if (a > absmax)
        absmax = a;
    }
  }
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float o_act_scale = absmax * (1.0f / 127.0f);
  float o_inv_act_scale = sw_recip(o_act_scale);

  // Pass B: requant with the self-computed global scale.
  for (int h = 0; h < kNHeadsQ; h++) {
    float combined = sv_out_scales[h] * o_inv_act_scale;
    int8_t *src = af_in + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int j = 0; j < kHD; j++) {
      dst[j] = round_to_i8((float)src[j] * combined);
    }
  }

  memcpy(af_out + kOutDim, &o_act_scale, 4);
  int32_t zero = 0;
  memcpy(af_out + kOutDim + 4, &zero, 4);
  event1();
}

// Fully self-calibrating af_concat (1b+1c): sv_out_scales come from the MERGED
// sv buffer tail (written by sv_merge_selfcal), NOT a host scales fifo. Input
// `sv` = [kQD bodies | kNHeadsQ*4 sv_out_scales]. Computes o_act_scale on-chip
// and writes it to af_out[kOutDim..kOutDim+4]. Single input -> af_concat drops
// from 2-in to 1-in (the host af_scales fifo is gone).
extern "C" void llama_af_concat_selfcal2(int8_t *restrict sv,
                                         int8_t *restrict af_out) {
  event0();
  ::aie::set_rounding(aie::rounding_mode::conv_even);
  constexpr int kOutDim = kNHeadsQ * kHD;
  const float *sv_out_scales = reinterpret_cast<const float *>(sv + kOutDim);

  // Pass A: absmax of af_fp = sv[h]*sv_out_scale[h].
  float absmax = 0.0f;
  for (int h = 0; h < kNHeadsQ; h++) {
    float s = sv_out_scales[h];
    int8_t *src = sv + h * kHD;
    for (int j = 0; j < kHD; j++) {
      float v = (float)src[j] * s;
      float a = v >= 0.0f ? v : -v;
      if (a > absmax)
        absmax = a;
    }
  }
  if (absmax < 1e-12f)
    absmax = 1e-12f;
  float o_act_scale = absmax * (1.0f / 127.0f);
  float o_inv_act_scale = sw_recip(o_act_scale);

  // Pass B: requant.
  for (int h = 0; h < kNHeadsQ; h++) {
    float combined = sv_out_scales[h] * o_inv_act_scale;
    int8_t *src = sv + h * kHD;
    int8_t *dst = af_out + h * kHD;
    for (int j = 0; j < kHD; j++) {
      dst[j] = round_to_i8((float)src[j] * combined);
    }
  }

  memcpy(af_out + kOutDim, &o_act_scale, 4);
  int32_t zero = 0;
  memcpy(af_out + kOutDim + 4, &zero, 4);
  event1();
}
