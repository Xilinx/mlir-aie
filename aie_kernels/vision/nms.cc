//===- nms.cc ---------------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A detector's score threshold and class-aware NMS, bit for bit
// torchvision's CPU nms under batched_nms's own choice of path.
// nms_setup, nms_score, nms_finish.
//
//===----------------------------------------------------------------------===//

#include <stdint.h>

#include "../aie_kernel_utils.h"
#include <aie_api/aie.hpp>

// AIE2P's scalar fadd/fsub aren't correctly rounded; use the soft-float calls.
extern "C" float __addsf3(float, float);
extern "C" float __subsf3(float, float);

#ifndef NMS_WIDTH
#error                                                                         \
    "NMS_WIDTH (the widest pyramid level's cells, a row's compile size) must be defined"
#endif
#ifndef NMS_CLASSES
#error "NMS_CLASSES (class scores per cell) must be defined"
#endif
#ifndef NMS_CAP
#error "NMS_CAP (candidates one core holds) must be defined"
#endif

namespace {
constexpr int WIDTH = NMS_WIDTH;
constexpr int CLASSES = NMS_CLASSES;
constexpr int CAP = NMS_CAP;
constexpr int OUT = 5 + CLASSES;
static_assert(CAP <= 1000, "cap is above 1000");

int32_t n_cand, seen;
// torchvision's boxes.max(), over every survivor, kept by CAP or not.
float conf_thre_stored, max_coord;
float cand_x1[CAP], cand_y1[CAP], cand_x2[CAP], cand_y2[CAP], cand_score[CAP];
// cand_seq: arrival order, the tie-break once a slot no longer tracks it.
int32_t cand_class[CAP], cand_seq[CAP];

float work_x1[CAP], work_y1[CAP], work_x2[CAP], work_y2[CAP], work_area[CAP];
int32_t sort_order[CAP];
bool sort_used[CAP];
bool suppressed[CAP];

} // namespace

extern "C" {

void nms_setup(float conf_thre) {
  n_cand = 0;
  seen = 0;
  conf_thre_stored = conf_thre;
}

void nms_score(const float *__restrict row, int32_t width) {
  event0();
  for (int32_t i = 0; i < width; i++) {
    const float *cell = row + i * OUT;
    float best = cell[5];
    int32_t best_c = 0;
    for (int32_t c = 1; c < CLASSES; c++) {
      const float v = cell[5 + c];
      if (v > best) {
        best = v;
        best_c = c;
      }
    }
    const float obj = cell[4];
    const float score = obj * best;
    if (score < conf_thre_stored)
      continue;
    const float cx = cell[0], cy = cell[1], w = cell[2], h = cell[3];
    const float half_w = w * 0.5f, half_h = h * 0.5f;
    const float x1 = __subsf3(cx, half_w), y1 = __subsf3(cy, half_h);
    const float x2 = __addsf3(cx, half_w), y2 = __addsf3(cy, half_h);

    const int32_t seq = seen;
    if (seq == 0)
      max_coord = x1;
    seen = seq + 1;
    if (x1 > max_coord)
      max_coord = x1;
    if (y1 > max_coord)
      max_coord = y1;
    if (x2 > max_coord)
      max_coord = x2;
    if (y2 > max_coord)
      max_coord = y2;

    int32_t slot = n_cand;
    if (n_cand == CAP) {
      slot = 0;
      for (int32_t k = 1; k < CAP; k++)
        if (cand_score[k] < cand_score[slot] ||
            (cand_score[k] == cand_score[slot] && cand_seq[k] > cand_seq[slot]))
          slot = k;
      if (score <= cand_score[slot])
        continue;
    } else {
      n_cand++;
    }
    cand_x1[slot] = x1;
    cand_y1[slot] = y1;
    cand_x2[slot] = x2;
    cand_y2[slot] = y2;
    cand_score[slot] = score;
    cand_class[slot] = best_c;
    cand_seq[slot] = seq;
  }
  event1();
}

void nms_finish(float *__restrict out, int32_t *__restrict count,
                float nms_thre, int32_t class_agnostic) {
  event0();
  const int32_t n = n_cand;
  const int32_t seen_local = seen;
  const float max_coord_local = max_coord;
  count[1] = seen_local;
  if (n == 0) {
    for (int32_t k = 0; k < CAP * 6; k++)
      out[k] = 0.0f;
    count[0] = 0;
    event1();
    return;
  }

  const bool vanilla = !class_agnostic && seen_local > 1000;
  const bool offset_trick = !class_agnostic && !vanilla;
  // Duplicated: a runtime branch in this loop makes Peano's pipeliner 35x
  // slower.
  if (offset_trick) {
    for (int32_t i = 0; i < n; i++) {
      const float offset =
          (float)cand_class[i] * __addsf3(max_coord_local, 1.0f);
      work_x1[i] = __addsf3(cand_x1[i], offset);
      work_y1[i] = __addsf3(cand_y1[i], offset);
      work_x2[i] = __addsf3(cand_x2[i], offset);
      work_y2[i] = __addsf3(cand_y2[i], offset);
      work_area[i] =
          __subsf3(work_x2[i], work_x1[i]) * __subsf3(work_y2[i], work_y1[i]);
    }
  } else {
    for (int32_t i = 0; i < n; i++) {
      work_x1[i] = cand_x1[i];
      work_y1[i] = cand_y1[i];
      work_x2[i] = cand_x2[i];
      work_y2[i] = cand_y2[i];
      work_area[i] =
          __subsf3(work_x2[i], work_x1[i]) * __subsf3(work_y2[i], work_y1[i]);
    }
  }

  for (int32_t i = 0; i < n; i++)
    sort_used[i] = false;
  for (int32_t k = 0; k < n; k++) {
    int32_t best = -1;
    for (int32_t i = 0; i < n; i++) {
      if (sort_used[i])
        continue;
      if (best == -1 || cand_score[i] > cand_score[best] ||
          (cand_score[i] == cand_score[best] && cand_seq[i] < cand_seq[best]))
        best = i;
    }
    sort_used[best] = true;
    sort_order[k] = best;
  }

  for (int32_t i = 0; i < n; i++)
    suppressed[i] = false;
  int32_t kept = 0;
  if (vanilla) {
    for (int32_t _i = 0; _i < n; _i++) {
      const int32_t i = sort_order[_i];
      if (suppressed[i])
        continue;
      out[kept * 6 + 0] = cand_x1[i];
      out[kept * 6 + 1] = cand_y1[i];
      out[kept * 6 + 2] = cand_x2[i];
      out[kept * 6 + 3] = cand_y2[i];
      out[kept * 6 + 4] = cand_score[i];
      out[kept * 6 + 5] = (float)cand_class[i];
      kept++;
      const float ix1 = work_x1[i], iy1 = work_y1[i], ix2 = work_x2[i],
                  iy2 = work_y2[i], iarea = work_area[i];
      for (int32_t _j = _i + 1; _j < n; _j++) {
        const int32_t j = sort_order[_j];
        if (suppressed[j])
          continue;
        if (cand_class[j] != cand_class[i])
          continue;
        const float xx1 = ix1 > work_x1[j] ? ix1 : work_x1[j];
        const float yy1 = iy1 > work_y1[j] ? iy1 : work_y1[j];
        const float xx2 = ix2 < work_x2[j] ? ix2 : work_x2[j];
        const float yy2 = iy2 < work_y2[j] ? iy2 : work_y2[j];
        const float dx = __subsf3(xx2, xx1), dy = __subsf3(yy2, yy1);
        const float w = dx > 0.0f ? dx : 0.0f;
        const float h = dy > 0.0f ? dy : 0.0f;
        const float inter = w * h;
        const float ovr =
            inter / __subsf3(__addsf3(iarea, work_area[j]), inter);
        if (ovr > nms_thre)
          suppressed[j] = true;
      }
    }
  } else {
    for (int32_t _i = 0; _i < n; _i++) {
      const int32_t i = sort_order[_i];
      if (suppressed[i])
        continue;
      out[kept * 6 + 0] = cand_x1[i];
      out[kept * 6 + 1] = cand_y1[i];
      out[kept * 6 + 2] = cand_x2[i];
      out[kept * 6 + 3] = cand_y2[i];
      out[kept * 6 + 4] = cand_score[i];
      out[kept * 6 + 5] = (float)cand_class[i];
      kept++;
      const float ix1 = work_x1[i], iy1 = work_y1[i], ix2 = work_x2[i],
                  iy2 = work_y2[i], iarea = work_area[i];
      for (int32_t _j = _i + 1; _j < n; _j++) {
        const int32_t j = sort_order[_j];
        if (suppressed[j])
          continue;
        const float xx1 = ix1 > work_x1[j] ? ix1 : work_x1[j];
        const float yy1 = iy1 > work_y1[j] ? iy1 : work_y1[j];
        const float xx2 = ix2 < work_x2[j] ? ix2 : work_x2[j];
        const float yy2 = iy2 < work_y2[j] ? iy2 : work_y2[j];
        const float dx = __subsf3(xx2, xx1), dy = __subsf3(yy2, yy1);
        const float w = dx > 0.0f ? dx : 0.0f;
        const float h = dy > 0.0f ? dy : 0.0f;
        const float inter = w * h;
        const float ovr =
            inter / __subsf3(__addsf3(iarea, work_area[j]), inter);
        if (ovr > nms_thre)
          suppressed[j] = true;
      }
    }
  }
  for (int32_t k = kept; k < CAP; k++)
    for (int32_t c = 0; c < 6; c++)
      out[k * 6 + c] = 0.0f;
  count[0] = kept;
  event1();
}

} // extern "C"
