//===- filter2d.cc ----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define REL_WRITE 0
#define REL_READ 1

#define THRESH_TYPE XF_THRESHOLD_TYPE_BINARY

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

const int32_t SRS_SHIFT = 12;

#define KERNEL_WIDTH 3

constexpr unsigned VecFactor = 32;

constexpr unsigned Lanes = 32; // Parallel vector output lanes
constexpr unsigned Points = 8; // Columns where data in summed togther
constexpr unsigned CoeffStep = 1;
constexpr unsigned DataStepXY = 1;

using mul_ops =
    aie::sliding_mul_xy_ops<Lanes, Points, CoeffStep, DataStepXY, int8, uint8>;

void filter2d_3lines_aie(uint8_t *__restrict lineIn0,
                         uint8_t *__restrict lineIn1,
                         uint8_t *__restrict lineIn2,
                         uint8_t *__restrict output, const int32_t width,
                         int16_t *__restrict kernel) {
  event0();

  set_sat(); // Needed for int16 to saturate properly to uint8

  // One line per kernel row; the vectors below are indexed the same way.
  uint8_t *line[KERNEL_WIDTH] = {lineIn0, lineIn1, lineIn2};
  aie::vector<uint8, 64> data_buf[KERNEL_WIDTH];
  aie::vector<uint8, 64> prev_buf[KERNEL_WIDTH];
  aie::vector<int8, 32> kernel_vec;

  const uint32_t kernel_side = KERNEL_WIDTH / 2;

#if AIE_TUNED_AIE2P
  // Each row's three taps are the high bytes of three int16s, packed into the
  // first 32-bit word of its eight-byte group; the rest of the group is zero.
  aie::vector<int32, 8> packed = aie::zeros<int32, 8>();
  AIE_LOOP_UNROLL_FULL
  for (int j = 0; j < KERNEL_WIDTH; j++) {
    uint32_t k0 = (uint16_t)kernel[0], k1 = (uint16_t)kernel[1],
             k2 = (uint16_t)kernel[2];
    packed[2 * j] = (int32)((k0 >> 8) | (k1 & 0xff00) | ((k2 & 0xff00) << 8));
    kernel += KERNEL_WIDTH;
  }
  kernel_vec = packed.template cast_to<int8>();
#else
  for (int j = 0; j < KERNEL_WIDTH; j++) {
    for (int i = 0; i < KERNEL_WIDTH; i++) {
      kernel_vec[j * Points + i] =
          (int8_t)((*kernel) >> 8); // int16 to int8 shift
      kernel++;
    }
    for (int i2 = 0; i2 < Points - KERNEL_WIDTH; i2++) {
      kernel_vec[j * Points + KERNEL_WIDTH + i2] = 0;
    }
  }
#endif

#if AIE_TUNED_AIE2P
  // AIE2P's 8-bit sliding mul is 64 lanes over a 128-pixel window, so one mac
  // per row covers a 64-pixel block. The window is two whole blocks as
  // loaded, which puts each result one pixel ahead of its block; the store
  // shifts it back, taking the first pixel from the previous result. The
  // 32-lane form shifted each row's input instead, and loading every chunk
  // twice and carrying the previous one held its loop at II18 on moves.
  using mul64_ops =
      aie::sliding_mul_xy_ops<64, Points, CoeffStep, DataStepXY, int8, uint8>;
  aie::vector<uint8, 64> blk[KERNEL_WIDTH], nxt[KERNEL_WIDTH];
  auto conv =
      [&](const aie::vector<uint8, 64> *lo,
          const aie::vector<uint8, 64> *hi) __attribute__((always_inline)) {
        auto a = mul64_ops::mul(kernel_vec, 0, aie::concat(lo[0], hi[0]), 0);
        AIE_LOOP_UNROLL_FULL
        for (int r = 1; r < KERNEL_WIDTH; r++)
          a = mul64_ops::mac(a, kernel_vec, r * Points,
                             aie::concat(lo[r], hi[r]), 0);
        return a.template to_vector<uint8>(SRS_SHIFT - 8);
      };
  auto load_block = [&](int r) __attribute__((always_inline)) {
    aie::vector<uint8, 64> b = aie::concat(
        aie::load_v<32>(line[r]), aie::load_v<32>(line[r] + VecFactor));
    line[r] += 2 * VecFactor;
    return b;
  };

  // left of line, border extension by mirroring: the last lane of this
  // result is the first pixel's
  AIE_LOOP_UNROLL_FULL
  for (int r = 0; r < KERNEL_WIDTH; r++) {
    nxt[r] = load_block(r);
    blk[r] = ::aie::shuffle_up_replicate(nxt[r], 2 * VecFactor - 1);
  }
  aie::vector<uint8, 64> prev = conv(blk, nxt);

  auto step = [&]() __attribute__((always_inline)) {
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < KERNEL_WIDTH; r++) {
      blk[r] = aie::concat(aie::load_v<32>(line[r] - 2 * VecFactor),
                           aie::load_v<32>(line[r] - VecFactor));
      nxt[r] = load_block(r);
    }
    aie::vector<uint8, 64> res = conv(blk, nxt);
    ::aie::store_v(output, ::aie::shuffle_up_fill(res, prev, kernel_side));
    output += 2 * VecFactor;
    prev = res;
  };
  const int blocks = width / (2 * VecFactor);
  if (blocks - 1 >= 4) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int j = 1; j < blocks; j++)
      step();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int j = 1; j < blocks; j++)
      step();
  }

  // right of line, border extension by mirroring
  AIE_LOOP_UNROLL_FULL
  for (int r = 0; r < KERNEL_WIDTH; r++)
    blk[r] = aie::concat(aie::load_v<32>(line[r] - 2 * VecFactor),
                         aie::load_v<32>(line[r] - VecFactor));
  if (width % (2 * VecFactor) == 0) {
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < KERNEL_WIDTH; r++)
      nxt[r] =
          ::aie::shuffle_down_replicate(blk[r], 2 * VecFactor - kernel_side);
    ::aie::store_v(output,
                   ::aie::shuffle_up_fill(conv(blk, nxt), prev, kernel_side));
  } else {
    // A last 32 pixels, replicated past the end, follow the last block and
    // then take a conv of their own.
    AIE_LOOP_UNROLL_FULL
    for (int r = 0; r < KERNEL_WIDTH; r++) {
      aie::vector<uint8, 32> c = aie::load_v<32>(line[r]);
      nxt[r] = ::aie::shuffle_down_replicate(aie::concat(c, c), VecFactor);
    }
    aie::vector<uint8, 64> res = conv(blk, nxt);
    ::aie::store_v(output, ::aie::shuffle_up_fill(res, prev, kernel_side));
    output += 2 * VecFactor;
    ::aie::store_v(output,
                   ::aie::shuffle_up_fill(conv(nxt, nxt), res, kernel_side)
                       .template extract<32>(0));
  }
#else
  // left of line, border extension by mirroring
  for (int r = 0; r < KERNEL_WIDTH; r++) {
    data_buf[r].insert(0, aie::load_v<32>(line[r]));
    line[r] += VecFactor;
    data_buf[r].insert(1, aie::load_v<32>(line[r]));
    prev_buf[r].insert(1, data_buf[r].template extract<32>(0));
    data_buf[r] = ::aie::shuffle_up_replicate(data_buf[r], kernel_side);
  }
  auto acc = mul_ops::mul(kernel_vec, 0, data_buf[0], 0);
  for (int r = 1; r < KERNEL_WIDTH; r++) {
    acc = mul_ops::mac(acc, kernel_vec, r * Points, data_buf[r], 0);
  }
  ::aie::store_v(output, acc.to_vector<uint8>(SRS_SHIFT - 8));
  output += VecFactor;

  // middle of line, no border extension needed
  auto body = [&]() __attribute__((always_inline)) {
    for (int r = 0; r < KERNEL_WIDTH; r++) {
      data_buf[r].insert(0, aie::load_v<32>(line[r]));
      line[r] += VecFactor;
      data_buf[r].insert(1, aie::load_v<32>(line[r]));
      // The pixel carried to the next iteration is this vector's own last
      // one, so it has to be taken before the shuffle. Reading it back out
      // afterwards yields the already-shifted vector, whose last element is
      // the second-to-last pixel, and every 32-pixel boundary from the third
      // vector on then convolves against the wrong left neighbor. The store
      // stays after the shuffle, which still needs the previous iteration's
      // value.
      auto carry = data_buf[r].template extract<32>(0);
      data_buf[r] =
          ::aie::shuffle_up_fill(data_buf[r], prev_buf[r], kernel_side);
      prev_buf[r].insert(1, carry);
    }
    acc = mul_ops::mul(kernel_vec, 0, data_buf[0], 0);
    for (int r = 1; r < KERNEL_WIDTH; r++) {
      acc = mul_ops::mac(acc, kernel_vec, r * Points, data_buf[r], 0);
    }
    ::aie::store_v(output, acc.to_vector<uint8>(SRS_SHIFT - 8));
    output += VecFactor;
  };
#if AIE_TUNED_AIE2
  // Pipelined when known to run at least four times (see rgba2gray.cc), that
  // is when i = 5 * VecFactor still passes the test.
  if (5 * (int)VecFactor < width - 1) {
    AIE_LOOP_NO_UNROLL
    AIE_LOOP_MIN_ITERATION_COUNT(4)
    for (int i = 2 * VecFactor; i < width - 1; i += VecFactor)
      body();
  } else {
    AIE_LOOP_NO_UNROLL
    for (int i = 2 * VecFactor; i < width - 1; i += VecFactor)
      body();
  }
#else
  for (int i = 2 * VecFactor; i < width - 1; i += VecFactor)
    body();
#endif

  // right of line, border extension by mirroring
  for (int r = 0; r < KERNEL_WIDTH; r++) {
    data_buf[r].insert(1, aie::load_v<32>(line[r]));
    data_buf[r] = ::aie::shuffle_down_replicate(data_buf[r], 32);
    data_buf[r] = ::aie::shuffle_up_fill(data_buf[r], prev_buf[r], kernel_side);
  }
  acc = mul_ops::mul(kernel_vec, 0, data_buf[0], 0);
  for (int r = 1; r < KERNEL_WIDTH; r++) {
    acc = mul_ops::mac(acc, kernel_vec, r * Points, data_buf[r], 0);
  }
  ::aie::store_v(output, acc.to_vector<uint8>(SRS_SHIFT - 8));
  output += VecFactor;
#endif

  event1();
}

extern "C" {

void filter2dLine(uint8_t *lineIn0, uint8_t *lineIn1, uint8_t *lineIn2,
                  uint8_t *out, int32_t lineWidth, int16_t *filterKernel) {
  filter2d_3lines_aie(lineIn0, lineIn1, lineIn2, out, lineWidth, filterKernel);
}

} // extern "C"
