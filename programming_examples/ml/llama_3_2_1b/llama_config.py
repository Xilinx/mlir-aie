#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B's layer shape and the decode's fixed scales and record sizes."""

D = 2048
HD = 8192
HEAD_D = 64
N_HEADS_Q = 32
N_HEADS_KV = 8
REP = N_HEADS_Q // N_HEADS_KV
N_TILE = 4

# A KV group's Q chunk: [REP * HEAD_D int8 | REP * 8 B of scales].
QCHUNK_BYTES = REP * HEAD_D + REP * 8
# A KV group's attention output: [REP * HEAD_D int8 | REP fp32 sv_out_scales].
SVCHUNK_BYTES = REP * HEAD_D + REP * 4
# Every Q head's attention output: [N_HEADS_Q * HEAD_D int8 | N_HEADS_Q fp32].
SVFULL_BYTES = N_HEADS_Q * HEAD_D + N_HEADS_Q * 4
CS_BYTES = HEAD_D * 2 * 2

ACT_SCALE = 0.05
INV_ACT_SCALE = 1.0 / ACT_SCALE
SILU_GATE_SCALE = 0.05
GATE_INV_OUT_SCALE = 1.0 / SILU_GATE_SCALE

# The softmax and sampler exp LUT's resolution.
EXP_QUANT_SCALE = 0.05
