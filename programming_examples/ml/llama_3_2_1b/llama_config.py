#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B's layer shape and the decode's record sizes."""

D = 2048
HD = 8192
HEAD_D = 64
N_HEADS_Q = 32
N_HEADS_KV = 8
REP = N_HEADS_Q // N_HEADS_KV
Q_DIM = N_HEADS_Q * HEAD_D
KV_DIM = N_HEADS_KV * HEAD_D
N_LAYERS = 16
VOCAB = 128_256
N_TILE = 4

# Each projection's (rows, K), in the order a layer runs them.
PROJECTIONS = {
    "q": (Q_DIM, D),
    "k": (KV_DIM, D),
    "v": (KV_DIM, D),
    "o": (D, Q_DIM),
    "g": (HD, D),
    "u": (HD, D),
    "d": (D, HD),
}

# A KV group's Q chunk: [REP * HEAD_D int8 | REP * 8 B of scales].
QCHUNK_BYTES = REP * HEAD_D + REP * 8
# A KV group's attention output: its REP Q heads' sv, REP * HEAD_D fp32.
SVCHUNK_BYTES = REP * HEAD_D * 4
# Every Q head's attention output, N_HEADS_Q * HEAD_D fp32.
SVFULL_BYTES = N_HEADS_Q * HEAD_D * 4
CS_BYTES = HEAD_D * 2 * 2
# KV cache rows a block holds: attention reads and folds the cache a block
# at a time.
KV_TB = 64
