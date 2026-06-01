//===- yolo_kernel_common.h -----------------------------------*- C++ -*-===//
//
// Shared inline helpers used by multiple yolo26n AIE2P kernels.
// Header-only: every helper is `static __attribute__((always_inline)) inline`
// so each translation unit gets its own copy and there's no ODR concern.
//
//===----------------------------------------------------------------------===//

#ifndef _YOLO_KERNEL_COMMON_H_
#define _YOLO_KERNEL_COMMON_H_

#include <stdint.h>

#include <aie_api/aie.hpp>

// 3x3 conv mmul<*,8,*> A-load with kx slide for inputs in
// (ic_t, x_block, p*8+chan) mmul-packed layout.
//
// kx=1 (center)   : single aie::load_v<64> at the natural x_tile offset.
// kx=0 (left tap) : load (x_tile-1, x_tile) and shuffle_down by 56 bytes to
//                   land the desired 64-byte stripe.
// kx=2 (right tap): load (x_tile, x_tile+1) and shuffle_down by 8 bytes.
// Out-of-bounds adjacent blocks are zero-filled (border behavior).
//
// Used by yolo_c3k2_heavy_inner_pair_cv1[_streamed]_vec.cc and
//        yolo_c3k2_heavy_inner_pair_cv2_skip[_streamed]_vec.cc.
static __attribute__((always_inline)) inline aie::vector<int8, 64>
load_a_mmul_kx(int8_t *line_ptr, int ic_t, int x_tile, int kx, int stride,
               int kXTiles8) {
  int8_t *base = line_ptr + ic_t * stride;
  if (kx == 1) {
    return aie::load_v<64>(base + x_tile * 64);
  }
  int blk_lo = (kx == 0) ? x_tile - 1 : x_tile;
  int blk_hi = blk_lo + 1;
  aie::vector<int8, 64> lo = (blk_lo >= 0 && blk_lo < kXTiles8)
                                 ? aie::load_v<64>(base + blk_lo * 64)
                                 : aie::zeros<int8, 64>();
  aie::vector<int8, 64> hi = (blk_hi >= 0 && blk_hi < kXTiles8)
                                 ? aie::load_v<64>(base + blk_hi * 64)
                                 : aie::zeros<int8, 64>();
  aie::vector<int8, 128> combined = aie::concat(lo, hi);
  const unsigned shift = (kx == 0) ? 56u : 8u;
  return aie::shuffle_down(combined, shift).template extract<64>(0);
}

#endif // _YOLO_KERNEL_COMMON_H_
