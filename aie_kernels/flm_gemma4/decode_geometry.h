//===- decode_geometry.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The model geometry every decode_*.cc kernel builds for, from -D flags. The
// flags are required, so a build that forgets one fails here rather than
// running with another model's shape.
#ifndef AIE_KERNELS_FLM_GEMMA4_DECODE_GEOMETRY_H
#define AIE_KERNELS_FLM_GEMMA4_DECODE_GEOMETRY_H

#include <aie_api/aie.hpp>

using bf16 = bfloat16;

#if !defined(FLM_GEMMA4_DECODE_MODEL_DIM) ||                                   \
    !defined(FLM_GEMMA4_DECODE_NUM_ATTN_HEADS) ||                              \
    !defined(FLM_GEMMA4_DECODE_NUM_KV_HEADS) ||                                \
    !defined(FLM_GEMMA4_DECODE_INTERMEDIATE_SIZE) ||                           \
    !defined(FLM_GEMMA4_DECODE_GLU_SLICE) ||                                   \
    !defined(FLM_GEMMA4_DECODE_PLI_D) || !defined(FLM_GEMMA4_DECODE_DH) ||     \
    !defined(FLM_GEMMA4_DECODE_SWA_DH) ||                                      \
    !defined(FLM_GEMMA4_DECODE_ATTN_SCALE) ||                                  \
    !defined(FLM_GEMMA4_DECODE_SWA_ATTN_SCALE) ||                              \
    !defined(FLM_GEMMA4_DECODE_PLI_PROJECTION_SCALE) ||                        \
    !defined(FLM_GEMMA4_DECODE_PLI_INPUT_SCALE)
#error "decode_geometry.h needs the model geometry -D defined"
#endif

// 1 selects GELU, 0 SiLU.
#ifndef FLM_GEMMA4_DECODE_GELU
#define FLM_GEMMA4_DECODE_GELU 1
#endif
#ifndef FLM_GEMMA4_DECODE_QK_NORM
#define FLM_GEMMA4_DECODE_QK_NORM 1
#endif
// Doubles the MLP of the layers that share another layer's KV cache.
#ifndef FLM_GEMMA4_DECODE_DOUBLE_WIDE_MLP
#define FLM_GEMMA4_DECODE_DOUBLE_WIDE_MLP 0
#endif

#define ATTN_IMPL_2x4x1 0
#define ATTN_IMPL_1x8x1 1
#if FLM_GEMMA4_DECODE_NUM_KV_HEADS == 1
#define ATTN_IMPL ATTN_IMPL_1x8x1
#elif FLM_GEMMA4_DECODE_NUM_KV_HEADS == 2
#define ATTN_IMPL ATTN_IMPL_2x4x1
#else
#error "the decode attention kernels serve one or two KV heads"
#endif

#define A_SILU 0
#define A_GELU 1
#define A_FUNC (FLM_GEMMA4_DECODE_GELU ? A_GELU : A_SILU)
#if FLM_GEMMA4_DECODE_QK_NORM
#define HAS_QK_NORM
#endif
#if FLM_GEMMA4_DECODE_DOUBLE_WIDE_MLP
#define DOUBLE_WIDE_MLP
#endif

constexpr int MODEL_DIM = FLM_GEMMA4_DECODE_MODEL_DIM;
constexpr int NUM_ATTN_HEADS = FLM_GEMMA4_DECODE_NUM_ATTN_HEADS;
constexpr int NUM_KV_HEADS = FLM_GEMMA4_DECODE_NUM_KV_HEADS;
constexpr int INTERMEDIATE_SIZE = FLM_GEMMA4_DECODE_INTERMEDIATE_SIZE;
constexpr int GLU_SLICE = FLM_GEMMA4_DECODE_GLU_SLICE;
constexpr int PLI_D = FLM_GEMMA4_DECODE_PLI_D; // per-layer input width
constexpr int SWA_DH = FLM_GEMMA4_DECODE_SWA_DH;
constexpr int DH = FLM_GEMMA4_DECODE_DH;
constexpr float SWA_ATTN_SCALE = FLM_GEMMA4_DECODE_SWA_ATTN_SCALE;
constexpr float ATTN_SCALE = FLM_GEMMA4_DECODE_ATTN_SCALE;
constexpr float PER_LAYER_MODEL_PROJECTION_SCALE =
    FLM_GEMMA4_DECODE_PLI_PROJECTION_SCALE;
constexpr float PER_LAYER_INPUT_SCALE = FLM_GEMMA4_DECODE_PLI_INPUT_SCALE;

// The shape of the projections' q4nx weight block; see q4nx.h.
constexpr int Q4NX_ROW_STRIDE = 16;
constexpr int Q4NX_ROW_STRIDE_SIZE = Q4NX_ROW_STRIDE;
constexpr int Q4NX_ROW_BLOCK_SIZE = 32;
constexpr int Q4NX_COL_BLOCK_SIZE = 256;
constexpr int Q4NX_GROUP_SIZE = 32;
constexpr int BF16_PROJ_M_BLOCK = 32;
constexpr int BF16_PROJ_K_BLOCK = 256;

constexpr int Q_HEADS_PER_GROUP = NUM_ATTN_HEADS / NUM_KV_HEADS;

constexpr int DQ = NUM_ATTN_HEADS * DH;
constexpr int DK = NUM_KV_HEADS * DH;
constexpr int DV = DK;

constexpr int SWA_DQ = NUM_ATTN_HEADS * SWA_DH;
constexpr int SWA_DK = NUM_KV_HEADS * SWA_DH;
constexpr int SWA_DV = SWA_DK;

#if ATTN_IMPL == ATTN_IMPL_2x4x1
const int GQA_R = 8;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 4; // 8x8x8 or 4x8x8, must be multiple of 4
constexpr int KV_HEADS_PER_CU = 2;
constexpr int Q_HEADS_PER_CU = Q_HEADS_PER_GROUP * 2;
constexpr int ATTN_GROUPS_PADDING =
    ((Q_HEADS_PER_GROUP + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) *
        GQA_SEGMENT_SIZE -
    Q_HEADS_PER_GROUP;
#elif ATTN_IMPL == ATTN_IMPL_1x8x1
const int GQA_R = 8;
const int GQA_S = 8;
const int GQA_T = 8;
constexpr int GQA_SEGMENT_SIZE = 8; // 8x8x8, must be multiple of 8
constexpr int KV_HEADS_PER_CU = 1;
constexpr int Q_HEADS_PER_CU = Q_HEADS_PER_GROUP * 1;
constexpr int ATTN_GROUPS_PADDING =
    ((Q_HEADS_PER_CU + GQA_SEGMENT_SIZE - 1) / GQA_SEGMENT_SIZE) *
        GQA_SEGMENT_SIZE -
    Q_HEADS_PER_CU;
#endif

constexpr int Q_HEADS_PER_GROUP_PADDED =
    Q_HEADS_PER_GROUP + ATTN_GROUPS_PADDING;

constexpr int Q_HEADS_PADDED_PER_CU =
    KV_HEADS_PER_CU * Q_HEADS_PER_GROUP_PADDED;

#endif // AIE_KERNELS_FLM_GEMMA4_DECODE_GEOMETRY_H
