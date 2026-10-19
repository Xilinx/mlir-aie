//===- experts.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// A mixture of experts' feed-forward on one core, a row at a time, from two
// input streams: the row, and every expert's weight tiles in a fixed order.
// Tile rows are bf16 dot products summed in float32, four rows to one tree.

#include <aie_api/aie.hpp>
#include <stdint.h>

#include "../aie_kernel_utils.h"
#include "../common/f32_split.h"
#include "../common/row_sums.h"

#ifndef EXPERTS_HIDDEN
#define EXPERTS_HIDDEN 64
#endif
#ifndef EXPERTS_WIDTH
#define EXPERTS_WIDTH 32
#endif
#ifndef EXPERTS_EXPERTS
#define EXPERTS_EXPERTS 4
#endif
#ifndef EXPERTS_TOP_K
#define EXPERTS_TOP_K 2
#endif
#ifndef EXPERTS_TILE_ROWS
#define EXPERTS_TILE_ROWS 8
#endif
#ifndef EXPERTS_TILE_COLS
#define EXPERTS_TILE_COLS 32
#endif

namespace {
constexpr uint32_t H = EXPERTS_HIDDEN, I = EXPERTS_WIDTH;
constexpr uint32_t T = EXPERTS_TILE_ROWS, C = EXPERTS_TILE_COLS;
static_assert((2 * I) % T == 0 && H % T == 0, "tile rows must divide 2I and H");
static_assert(H % C == 0 && I % C == 0, "tile columns must divide H and I");
static_assert(C % 16 == 0, "tile columns must be a multiple of 16");

constexpr uint32_t gate_up_steps = (2 * I / T) * (H / C);
constexpr uint32_t expert_steps = gate_up_steps + (H / T) * (I / C);
constexpr uint32_t row_steps = EXPERTS_EXPERTS * expert_steps;

// Lanes a tile row is read in, and lanes of a walk over h or y. AIE2 walks h
// 16 wide: at 32 Peano 94b320c0 asserts in instruction selection.
constexpr uint32_t V = C % 32 == 0 ? 32 : 16;
constexpr uint32_t VI = I % 32 == 0 && !AIE_ARCH_AIE2 ? 32 : 16;
constexpr uint32_t VH = H % 32 == 0 ? 32 : 16;

alignas(64) float acc[2 * I]; // gate and up dot products, then h in acc[0, I)
alignas(64) float sum[H];     // the weighted sum over the chosen experts
// weight * h as three bf16 limbs that sum to it exactly.
alignas(64) bfloat16 h_hi[I], h_mid[I], h_lo[I];

// Four tile rows times h's columns [c0, c0 + C), packed by pack_rows4.
__attribute__((always_inline)) inline aie::accum<accfloat, 64>
down_rows4(const bfloat16 *__restrict t, uint32_t c0) {
  const bfloat16 *__restrict t0 = t;
  const bfloat16 *__restrict t1 = t + C;
  const bfloat16 *__restrict t2 = t + 2 * C;
  const bfloat16 *__restrict t3 = t + 3 * C;
  const bfloat16 *__restrict hi = h_hi + c0;
  const bfloat16 *__restrict mid = h_mid + c0;
  const bfloat16 *__restrict lo = h_lo + c0;
  aie::accum<accfloat, V> a0, a1, a2, a3;
  auto chunk = [&](bool first) {
    const aie::vector<bfloat16, V> b0 = aie::load_v<V>(hi);
    const aie::vector<bfloat16, V> b1 = aie::load_v<V>(mid);
    const aie::vector<bfloat16, V> b2 = aie::load_v<V>(lo);
    hi += V;
    mid += V;
    lo += V;
    auto row = [&](aie::accum<accfloat, V> &a, const bfloat16 *__restrict &p) {
      const aie::vector<bfloat16, V> r = aie::load_v<V>(p);
      p += V;
      a = first ? aie::mul(r, b0) : aie::mac(a, r, b0);
      a = aie::mac(a, r, b1);
      a = aie::mac(a, r, b2);
    };
    row(a0, t0);
    row(a1, t1);
    row(a2, t2);
    row(a3, t3);
  };
  chunk(true);
  AIE_LOOP_UNROLL_FULL
  for (uint32_t k = V; k < C; k += V)
    chunk(false);
  return pack_rows4<V>(a0, a1, a2, a3);
}

using AccI = aie::accum<accfloat, VI>;

inline AccI zero_acc() { return aie::zeros<accfloat, VI>(); }

inline Limbs<VI> splat_limbs(float c) {
  return bf16_limbs(AccI(aie::broadcast<float, VI>(c)));
}

// silu(g) * u, each product taken on exact bf16 limbs, so within a few
// float32 ulps: exp(-g) the way scalar_exp takes it (Cody-Waite, then
// Taylor to r^7, times 2^k), and 1 / (1 + e) from aie::inv and one Newton
// step on an exact residual. Past -g = 88 or below -g = -80, sigmoid(g) is
// 0 or 1 to float32, and the clamp keeps 2^k p normal.
inline AccI silu_mul(aie::vector<float, VI> g, aie::vector<float, VI> u) {
  const aie::vector<float, VI> x =
      aie::min(aie::max(aie::sub(aie::zeros<float, VI>(), g),
                        aie::broadcast<float, VI>(-80.0f)),
               aie::broadcast<float, VI>(88.0f));
  const aie::vector<float, VI> magic = aie::broadcast<float, VI>(12582912.0f);
  const aie::vector<float, VI> t =
      aie::add(mac_limbs(zero_acc(), bf16_limbs(AccI(x)),
                         splat_limbs(1.44269504f))
                   .template to_vector<float>(),
               magic);
  const aie::vector<int32_t, VI> k = aie::sub(
      t.template cast_to<int32_t>(), aie::broadcast<int32_t, VI>(0x4b400000));
  // k and the two limbs of ln 2's high part are exact in bf16, and so is
  // each product; x - k ln2_hi cancels exactly (Sterbenz).
  const aie::vector<bfloat16, VI> kb =
      AccI(aie::sub(t, magic)).template to_vector<bfloat16>();
  AccI r = aie::msc(AccI(x), kb, aie::broadcast<bfloat16, VI>(0.69140625f));
  r = aie::msc(r, kb, aie::broadcast<bfloat16, VI>(0.001739501953125f));
  const Limbs<VI> ln2_lo = splat_limbs(1.42860677e-6f);
  r = aie::msc(r, kb, ln2_lo.hi);
  r = aie::msc(r, kb, ln2_lo.mid);
  const Limbs<VI> rl = bf16_limbs(r);
  constexpr float taylor[] = {1.0f / 720, 1.0f / 120, 1.0f / 24, 1.0f / 6,
                              0.5f,       1.0f,       1.0f};
  AccI p = AccI(aie::broadcast<float, VI>(1.0f / 5040));
  for (float c : taylor)
    p = mac_limbs(AccI(aie::broadcast<float, VI>(c)), bf16_limbs(p), rl);
  const aie::vector<bfloat16, VI> scale =
      AccI(aie::upshift(aie::add(k, aie::broadcast<int32_t, VI>(127)), 23)
               .template cast_to<float>())
          .template to_vector<bfloat16>();
  const Limbs<VI> pl = bf16_limbs(p);
  AccI d = aie::mul(pl.hi, scale);
  d = aie::mac(d, pl.mid, scale);
  d = aie::mac(d, pl.lo, scale);
  d = aie::mac(d, aie::broadcast<bfloat16, VI>(1.0f),
               aie::broadcast<bfloat16, VI>(1.0f));
  const aie::vector<float, VI> q0 = aie::inv(d.template to_vector<float>());
  const Limbs<VI> dl = bf16_limbs(d), ql = bf16_limbs(AccI(q0));
  // 1 - d q0 by msc: aie::neg on a float vector crashes Peano (llvm-aie
  // #1326, #1356).
  AccI res = AccI(aie::broadcast<float, VI>(1.0f));
  res = aie::msc(res, dl.hi, ql.hi);
  res = aie::msc(res, dl.hi, ql.mid);
  res = aie::msc(res, dl.mid, ql.hi);
  res = aie::msc(res, dl.hi, ql.lo);
  res = aie::msc(res, dl.mid, ql.mid);
  res = aie::msc(res, dl.lo, ql.hi);
  const AccI q = mac_limbs(AccI(q0), ql, bf16_limbs(res));
  const AccI gu = mac_limbs(zero_acc(), bf16_limbs(AccI(g)), bf16_limbs(AccI(u)));
  return mac_limbs(zero_acc(), bf16_limbs(gu), bf16_limbs(q));
}

// A tile row past the last group of four, summed: row(k) is its products
// over columns [k, k + V).
template <class Row>
inline float tail_row(Row row) {
  aie::accum<accfloat, V> a = row(0);
  AIE_LOOP_UNROLL_FULL
  for (uint32_t k = V; k < C; k += V)
    a = aie::add(a, row(k));
  return aie::reduce_add(a.template to_vector<float>());
}
} // namespace

// row is x (H), then the router's 2 * TOP_K weights (chosen expert j's at
// 2 * j), then its TOP_K int32 ids. Step s brings tile s % row_steps of the
// row's stream: for each expert e in order, the T x C blocks of gate_up
// (2I x H) row block by row block, then those of down (H x I). The tiles of
// an expert the row did not choose are skipped. y is the row's bf16 output
// so far; after the row's last step, the sum over j of weight j times
// down[ids[j]] @ (silu(g) * u).
extern "C" void experts(bfloat16 *restrict row, bfloat16 *restrict tile,
                        bfloat16 *restrict y, int32_t step) {
  event0();
  const uint32_t call = (uint32_t)step % row_steps;
  const uint32_t e = call / expert_steps;
  uint32_t s = call - e * expert_steps;
  if (call == 0)
    for (uint32_t o = 0; o < H; o += VH)
      aie::store_v(sum + o, aie::zeros<float, VH>());
  // Each id is an int32 in two bf16 slots, low half first.
  const bfloat16 *ids = row + H + 2 * EXPERTS_TOP_K;
  float weight = 0.0f;
  bool chosen = false;
  for (int j = 0; j < EXPERTS_TOP_K; j++)
    if ((int32_t)(__builtin_bit_cast(uint16_t, ids[2 * j]) |
                  (uint32_t)__builtin_bit_cast(uint16_t, ids[2 * j + 1])
                      << 16) == (int32_t)e) {
      weight += (float)row[H + 2 * j];
      chosen = true;
    }
  if (chosen && s == 0)
    for (uint32_t r = 0; r < 2 * I; r += VI)
      aie::store_v(acc + r, aie::zeros<float, VI>());
  if (chosen && s < gate_up_steps) {
    const uint32_t r0 = s / (H / C) * T, c0 = s % (H / C) * C;
    const bfloat16 *__restrict x = row + c0;
    const bfloat16 *__restrict t = tile;
    float *__restrict a = acc + r0;
    for (uint32_t g = 0; g < T / 4; g++, t += 4 * C, a += 4) {
      const aie::vector<float, 16> d =
          fold_rows<16>(mac_rows4<V, C>(t, x).template to_vector<float>());
      a[0] += d[0];
      a[1] += d[1];
      a[2] += d[2];
      a[3] += d[3];
    }
    for (uint32_t r = 0; r < T % 4; r++, t += C)
      a[r] += tail_row([&](uint32_t k) {
        return aie::accum<accfloat, V>(
            aie::mul(aie::load_v<V>(t + k), aie::load_v<V>(x + k)));
      });
    if (s == gate_up_steps - 1) {
      const Limbs<VI> w = splat_limbs(weight);
      for (uint32_t i = 0; i < I; i += VI) {
        const Limbs<VI> h = bf16_limbs(mac_limbs(
            zero_acc(), w,
            bf16_limbs(silu_mul(aie::load_v<VI>(acc + i),
                                aie::load_v<VI>(acc + I + i)))));
        aie::store_v(h_hi + i, h.hi);
        aie::store_v(h_mid + i, h.mid);
        aie::store_v(h_lo + i, h.lo);
      }
    }
  } else if (chosen) {
    s -= gate_up_steps;
    const uint32_t r0 = s / (I / C) * T, c0 = s % (I / C) * C;
    const bfloat16 *__restrict t = tile;
    float *__restrict o = sum + r0;
    for (uint32_t g = 0; g < T / 4; g++, t += 4 * C, o += 4) {
      const aie::vector<float, 16> d =
          fold_rows<16>(down_rows4(t, c0).template to_vector<float>());
      o[0] += d[0];
      o[1] += d[1];
      o[2] += d[2];
      o[3] += d[3];
    }
    for (uint32_t r = 0; r < T % 4; r++, t += C)
      o[r] += tail_row([&](uint32_t k) {
        const aie::vector<bfloat16, V> v = aie::load_v<V>(t + k);
        aie::accum<accfloat, V> d =
            aie::mul(v, aie::load_v<V>(h_hi + c0 + k));
        d = aie::mac(d, v, aie::load_v<V>(h_mid + c0 + k));
        return aie::mac(d, v, aie::load_v<V>(h_lo + c0 + k));
      });
  }
  for (uint32_t o = 0; o < H; o += VH)
    aie::store_v(y + o, aie::accum<accfloat, VH>(aie::load_v<VH>(sum + o))
                            .template to_vector<bfloat16>());
  event1();
}
