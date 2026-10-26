//===- decode_layout.h ------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIE_KERNELS_FLM_GEMMA4_DECODE_LAYOUT_H
#define AIE_KERNELS_FLM_GEMMA4_DECODE_LAYOUT_H

#include "decode_geometry.h"

// One round of the MVM cores computes M_PER_ROUND output rows. Each *_REPEATS
// constant is the number of rounds of one projection, so the projection reads
// its input that many times.
constexpr int MVM_CORES = 16;
constexpr int QKV_REPEATS = (DQ + DK + DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int Q_REPEATS = DQ / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int SWA_QKV_REPEATS =
    (SWA_DQ + SWA_DK + SWA_DV) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int SWA_Q_REPEATS = SWA_DQ / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int O_DOWN_REPEATS = MODEL_DIM / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;
constexpr int UP_GATE_REPEATS =
    (2 * INTERMEDIATE_SIZE) / MVM_CORES / Q4NX_ROW_BLOCK_SIZE;

constexpr int M_PER_ROUND = Q4NX_ROW_BLOCK_SIZE * MVM_CORES;

constexpr int pkt_id_rms_to_proj = 0;
constexpr int pkt_id_rms_to_it = 1;

constexpr bfloat16 exp_scale = 1.44269504089;

#endif // AIE_KERNELS_FLM_GEMMA4_DECODE_LAYOUT_H