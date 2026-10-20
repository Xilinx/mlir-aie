#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Block-wise attention (aie2_attn_block.py) on NPU2 vs its numpy mirror.

Every live length and core count must match numpy_attn_block byte for byte;
at T <= TB with one core the mirror itself must equal today's flowkv_mh
reference (numpy_attention_perslot_flowkv + per-head self-calibration).
"""

import argparse
import sys

import aie.iron as iron
import aie_fp
import numpy as np
from aie.utils import DefaultNPURuntime
from aie2_attn_block import (
    BLOCK_WORDS,
    HEAD_DIM,
    OUT_BYTES,
    REP,
    ROW_BYTES,
    TB,
    build,
    kv_words,
)
from llama_npu import add_artifact_args, design_kernel
from numpy_attn_block import attention
from numpy_layer_mh import numpy_attention_perslot_flowkv

N_LAYERS, N_HEADS, LAYER, HALF, HEAD = 2, 4, 1, 1, 2


def cache(rng, t):
    k = rng.integers(-128, 128, (t, HEAD_DIM), dtype=np.int8)
    v = rng.integers(-128, 128, (t, HEAD_DIM), dtype=np.int8)
    k_slot = rng.uniform(0.002, 0.05, t).astype(np.float32)
    v_slot = rng.uniform(0.002, 0.05, t).astype(np.float32)
    n_blocks = -(-t // TB)
    blob = rng.integers(
        -(2**31), 2**31 - 1, kv_words(t, N_LAYERS, N_HEADS), dtype=np.int64
    ).astype(np.int32)
    rows = np.zeros((n_blocks * TB, ROW_BYTES), np.int8)
    rows[:t, :HEAD_DIM] = k
    rows[:t, HEAD_DIM : 2 * HEAD_DIM] = v
    rows[:t, 2 * HEAD_DIM : 2 * HEAD_DIM + 4] = k_slot.view(np.int8).reshape(t, 4)
    rows[:t, 2 * HEAD_DIM + 4 :] = v_slot.view(np.int8).reshape(t, 4)
    blocks = blob.reshape(n_blocks, N_LAYERS, 2, N_HEADS, BLOCK_WORDS)
    blocks[:, LAYER, HALF, HEAD] = rows.view(np.int32).reshape(n_blocks, BLOCK_WORDS)
    return k, v, k_slot, v_slot, blob


def flowkv_reference(q, q_scales, k, v, k_slot, v_slot):
    out = np.zeros((REP, HEAD_DIM), np.int8)
    for h in range(REP):
        _, sv = numpy_attention_perslot_flowkv(
            q[h], k, v, HEAD_DIM, len(k), q_scales[h], k_slot, v_slot, 1.0
        )
        absmax = np.float32(max(float(np.abs(sv).max()), 1e-12))
        out[h] = aie_fp.round_to_i8(
            sv * aie_fp.sw_recip(np.float32(absmax * np.float32(1.0 / 127.0)))
        )
    return out


def main():
    p = argparse.ArgumentParser()
    add_artifact_args(p)
    p.add_argument(
        "-T",
        type=int,
        nargs="+",
        default=[1, 63, 64, 65, 127, 128, 129, 1023, 1024, 4096],
    )
    p.add_argument("-C", type=int, nargs="+", default=[1, 2, 4], choices=(1, 2, 4))
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    fails = 0
    for t in opts.T:
        rng = np.random.default_rng(opts.seed + t)
        q = rng.integers(-128, 128, (REP, HEAD_DIM), dtype=np.int8)
        q_scales = rng.uniform(0.01, 0.2, REP).astype(np.float32)
        k, v, k_slot, v_slot, blob = cache(rng, t)
        q_buf = np.concatenate(
            [
                q.reshape(-1),
                np.stack([q_scales, np.zeros(REP, np.float32)], 1)
                .reshape(-1)
                .view(np.int8),
            ]
        )
        if t <= TB:
            ref1, _, _ = attention(q, q_scales, k, v, k_slot, v_slot, t, TB, 1)
            same = np.array_equal(
                ref1, flowkv_reference(q, q_scales, k, v, k_slot, v_slot)
            )
            print(
                f"T={t}: one-block mirror vs flowkv_mh reference: {'EQUAL' if same else 'DIFFERENT'}"
            )
            fails += not same
        for c in opts.C:
            ref, scales, _ = attention(q, q_scales, k, v, k_slot, v_slot, t, TB, c)
            expect = np.concatenate([ref.reshape(-1), scales.view(np.int8)])
            out = iron.zeros((OUT_BYTES,), dtype=np.int8)
            npu = design_kernel(build, opts, T=t, C=c)
            rc = DefaultNPURuntime.run_test(
                npu,
                [iron.tensor(q_buf), iron.tensor(blob), out],
                {},
                verify=False,
                verbosity=opts.verbosity,
            )
            if rc != 0:
                return rc
            out.to("cpu")
            got = out.numpy()
            n = int(np.count_nonzero(got != expect))
            print(
                f"T={t} C={c}: {'BYTE-EXACT' if n == 0 else f'{n}/{OUT_BYTES} bytes differ'}"
            )
            fails += n != 0
    print("PASS" if fails == 0 else f"FAIL ({fails})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
