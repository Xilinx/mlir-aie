#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The kernels' rare paths on the NPU, from crafted inputs, against the oracle.

- llama_mma_f32's per-element epilogue, which half_epilogue takes when a sum
  leaves +-2^23: weight rows near +-127 against activations near +-127,
  through mma_rows as the decode runs a projection's row group, against
  llama_oracle.Layer.project; and small sums, through the vector epilogue.
- llama_attn_block's fixed-point sv (block_sv) at its edges: a block where
  live rows' pv sit more than 30 binades below its largest, so their P is 0
  (one dominant dot leaves the other rows' e near 2^-115), and one whose
  largest pv is at or above 2^112 (a v_scale near 2^119), scaled near the
  top of the fp32 range. One core attends two blocks, the second the
  query's own, as the decode's pair of cores for a KV head does, against
  llama_oracle.attention_blocks.

Each case first checks that its inputs take the path it names.

Run:
  python test_kernel_edges.py
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.iron import (
    Buffer,
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    jit,
    tensor,
    zeros,
)
from aie.iron.device import NPU2
from aie2_decode import (
    ACT_BYTES,
    ATTN_STATE_BYTES,
    KV_HEAD_BLOCK_WORDS,
    KV_ROW_WORDS,
    M_MAX,
    ROW_CHUNK,
    ROW_CHUNK_STRIDE,
)
from aie2_gemv_array import MMA_SLOT, mma_kernels, mma_rows
from llama_config import (
    HEAD_D,
    KV_DIM,
    KV_TB,
    N_HEADS_KV,
    N_HEADS_Q,
    QCHUNK_BYTES,
    REP,
    SVCHUNK_BYTES,
    D,
)
from llama_decode import pack_lane_mma
from llama_npu import DispatchSession
from llama_oracle import Layer, attention_blocks, attention_weights

# One row group's outputs, M_MAX activations by 8 rows in fp32: projection
# k's first two slots on core 0.
MMA_OUT_BYTES = M_MAX * 8 * 4
# The own block's live rows end at OWN_ROW, the query's own.
OWN_ROW = 37


@jit
def mma_group():
    t_act = np.ndarray[(M_MAX * ACT_BYTES,), np.dtype[np.int8]]
    t_slot = np.ndarray[(MMA_SLOT,), np.dtype[np.int8]]
    t_out = np.ndarray[(MMA_OUT_BYTES,), np.dtype[np.int8]]
    k_spread, k_mma, _, t_tile, t_part = mma_kernels(t_act, t_slot, t_out, t_out, M_MAX)
    of_act = ObjectFifo(t_act, depth=1, name="act")
    of_w = ObjectFifo(t_slot, depth=2, name="w")
    of_out = ObjectFifo(t_out, depth=1, name="out")

    def body(c_act, c_w, c_out, tile, part, spread, k_acc, k_f):
        a = c_act.acquire(1)
        spread(a, tile, M_MAX, D, 0, M_MAX * D, 16)
        c_act.release(1)
        o = c_out.acquire(1)
        mma_rows("k", tile, c_w, part, o, (k_acc, k_f), 0, M_MAX, 8, 1)
        c_out.release(1)

    worker = Worker(
        body,
        [
            of_act.cons(),
            of_w.cons(),
            of_out.prod(),
            Buffer(t_tile, name="tile"),
            Buffer(t_part, name="part"),
            k_spread,
            *k_mma,
        ],
    )

    def sequence(act, w, out, p_act, p_w, c_out):
        p_act.fill(act)
        p_w.fill(w)
        c_out.drain(out, wait=True)

    rt = Runtime(
        sequence,
        [
            t_act,
            np.ndarray[(2 * MMA_SLOT,), np.dtype[np.int8]],
            t_out,
            of_act.prod(),
            of_w.prod(),
            of_out.cons(),
        ],
    )
    return Program(NPU2(), rt, workers=[worker]).resolve_program()


@jit
def attn_pair(*, ROW0: CompileTime[int] = OWN_ROW):
    t_rc = np.ndarray[(ROW_CHUNK,), np.dtype[np.int8]]
    t_kvhb = np.ndarray[(KV_HEAD_BLOCK_WORDS,), np.dtype[np.int32]]
    t_chunks = np.ndarray[(M_MAX * ROW_CHUNK_STRIDE,), np.dtype[np.int8]]
    t_states = np.ndarray[(M_MAX * ATTN_STATE_BYTES,), np.dtype[np.int8]]
    t_state = np.ndarray[(ATTN_STATE_BYTES,), np.dtype[np.int8]]
    t_sv = np.ndarray[(SVCHUNK_BYTES,), np.dtype[np.int8]]
    KT = llama_kernels.attn_block(tb=KV_TB)
    k_take = KT.bind("llama_attn_take_q", [t_rc, t_chunks, t_states, np.int32])
    k_blk = KT.bind("llama_attn_block_q", [t_chunks, t_kvhb, t_states] + [np.int32] * 2)
    k_put = KT.bind("llama_attn_put_q", [t_states, t_state, np.int32])
    k_pair = KT.bind("llama_attn_merge_pair_q", [t_state, t_states, t_sv, np.int32])
    of_rc = ObjectFifo(t_rc, depth=1, name="rc")
    of_kv = ObjectFifo(t_kvhb, depth=2, name="kv")
    of_sv = ObjectFifo(t_sv, depth=1, name="sv")

    # Block 0 into the first core's state, the own block 1 into the second's,
    # then the second merges the first's, as w_attn's pair does.
    def body(c_rc, c_kv, c_sv, chunks, st_a, st_b, part, take, blk, put, pair):
        rc = c_rc.acquire(1)
        take(rc, chunks, st_a, 0)
        take(rc, chunks, st_b, 0)
        c_rc.release(1)
        for st, row0 in ((st_a, -1), (st_b, ROW0)):
            kb = c_kv.acquire(1)
            blk(chunks, kb, st, row0, 0)
            c_kv.release(1)
        put(st_a, part, 0)
        sv = c_sv.acquire(1)
        pair(part, st_b, sv, 0)
        c_sv.release(1)

    worker = Worker(
        body,
        [
            of_rc.cons(),
            of_kv.cons(),
            of_sv.prod(),
            Buffer(t_chunks, name="chunks"),
            Buffer(t_states, name="st_a"),
            Buffer(t_states, name="st_b"),
            Buffer(t_state, name="part"),
            k_take,
            k_blk,
            k_put,
            k_pair,
        ],
    )

    def sequence(rc, kv, sv, p_rc, p_kv, c_sv):
        p_rc.fill(rc)
        p_kv.fill(kv)
        c_sv.drain(sv, wait=True)

    rt = Runtime(
        sequence,
        [
            t_rc,
            np.ndarray[(2 * KV_HEAD_BLOCK_WORDS,), np.dtype[np.int32]],
            t_sv,
            of_rc.prod(),
            of_kv.prod(),
            of_sv.cons(),
        ],
    )
    return Program(NPU2(), rt, workers=[worker]).resolve_program()


class MmaCase:
    """Eight weight rows against M_MAX activations, and the oracle's rows.

    Args:
        `name`: The case's name.
        `w`: int8 (8, D), the row group.
        `act`: int8 (M_MAX, D), the activations.
        `rng`: The numpy Generator for the scales.
    """

    def __init__(self, name, w, act, rng):
        self.name = name
        self.act = act
        self.act_scales = rng.uniform(0.001, 0.1, M_MAX).astype(np.float32)
        w_i8 = np.zeros((KV_DIM, D), np.int8)
        w_i8[:8] = w
        w_sc = rng.uniform(0.001, 0.05, KV_DIM).astype(np.float32)
        self.layer = Layer({"k": (w_i8, w_sc)}, None, None, 1)
        self.sums = w.astype(np.int64) @ act.astype(np.int64).T

    def per_element(self):
        """Whether half_epilogue takes its per-element path: a sum past +-2^23."""
        return bool(np.any((self.sums <= -(1 << 23)) | (self.sums >= 1 << 23)))

    def run(self, session):
        """Run the case; return the words that differ from the oracle's."""
        act = np.zeros((M_MAX * ACT_BYTES,), np.int8)
        act[: M_MAX * D] = self.act.ravel()
        tail = np.zeros((M_MAX, 4), np.float32)
        tail[:, 0] = self.act_scales
        act[M_MAX * D :] = tail.view(np.int8).ravel()
        w = pack_lane_mma(self.layer.weights, "k", 0)[: 2 * MMA_SLOT]
        out_t = zeros(MMA_OUT_BYTES, dtype=np.int8)
        ms = session([tensor(act, dtype=np.int8), tensor(w, dtype=np.int8), out_t])
        got = out_t.numpy().view(np.float32).reshape(M_MAX, 8)
        want = np.stack(
            [
                self.layer.project("k", self.act[m], self.act_scales[m])[:8]
                for m in range(M_MAX)
            ]
        )
        return int(np.count_nonzero(got.view(np.int32) != want.view(np.int32))), ms


class AttnCase:
    """A query's KV head 0 over two blocks, the second its own.

    Args:
        `name`: The case's name.
        `q`: int8 (N_HEADS_Q, HEAD_D).
        `q_scales`: fp32 (N_HEADS_Q,).
        `k`, `v`: int8 (N_HEADS_KV, 2 * KV_TB, HEAD_D).
        `ks`, `vs`: fp32 (N_HEADS_KV, 2 * KV_TB), the rows' scales.
        `rng`: The numpy Generator for the own row's stale bytes.
    """

    def __init__(self, name, q, q_scales, k, v, ks, vs, rng):
        self.name = name
        self.q = q
        self.q_scales = q_scales
        self.k = k
        self.v = v
        self.ks = ks
        self.vs = vs
        self.t = KV_TB + OWN_ROW + 1
        self.rng = rng

    def paths(self):
        """Which of block_sv's edges each block reaches, from the oracle's pv.

        Returns:
            Per block, "drop" if a live nonzero pv's exponent field is more
            than 30 below the block's largest, else "top" if the largest is
            at least 230, else "plain".
        """
        pv = attention_weights(
            self.q, self.q_scales, self.k, self.ks, self.vs, self.t, KV_TB
        )[2][:REP]
        live = (np.arange(2 * KV_TB) < self.t).reshape(2, KV_TB)
        field = (pv.view(np.int32) >> 23) & 0xFF
        out = []
        for b in range(2):
            f = np.where(live[b], field[:, b], 0)
            top = f.max(axis=1, keepdims=True)
            if np.any((f > 0) & (top - f > 30)):
                out.append("drop")
            elif np.any(top >= 230):
                out.append("top")
            else:
                out.append("plain")
        return out

    def run(self, session):
        """Run the case; return the words that differ from the oracle's."""
        rows = 2 * KV_TB
        blk = np.zeros((rows, KV_ROW_WORDS * 4), np.uint8)
        blk[:, :HEAD_D] = self.k[0].view(np.uint8)
        blk[:, HEAD_D : 2 * HEAD_D] = self.v[0].view(np.uint8)
        blk[:, 2 * HEAD_D : 2 * HEAD_D + 4] = self.ks[0].view(np.uint8).reshape(rows, 4)
        blk[:, 2 * HEAD_D + 4 :] = self.vs[0].view(np.uint8).reshape(rows, 4)
        # The own row reaches the core in the query's chunk; the cache still
        # holds stale bytes there.
        own = blk[self.t - 1].copy()
        blk[self.t - 1] = self.rng.integers(0, 256, KV_ROW_WORDS * 4)
        rc = np.zeros(ROW_CHUNK, np.uint8)
        rc[: REP * HEAD_D] = self.q[:REP].view(np.uint8).ravel()
        tail = np.zeros((REP, 2), np.float32)
        tail[:, 0] = self.q_scales[:REP]
        rc[REP * HEAD_D : QCHUNK_BYTES] = tail.view(np.uint8).ravel()
        rc[QCHUNK_BYTES:] = own
        sv_t = zeros(SVCHUNK_BYTES, dtype=np.int8)
        ms = session(
            [
                tensor(rc.view(np.int8), dtype=np.int8),
                tensor(blk.view(np.int32).ravel(), dtype=np.int32),
                sv_t,
            ]
        )
        got = sv_t.numpy().view(np.float32).reshape(REP, HEAD_D)
        want = attention_blocks(
            self.q, self.q_scales, self.k, self.v, self.ks, self.vs, self.t, KV_TB
        )[:REP]
        return int(np.count_nonzero(got.view(np.int32) != want.view(np.int32))), ms


def mma_cases(rng):
    """The fallback case, rows near +-127 with random heads, and a vector one."""
    big_w = np.repeat(np.array([127, -128, 127, -127, 127, -128, 1, -1]), D)
    big_w = big_w.reshape(8, D).astype(np.int8)
    big_w[:, :16] = rng.integers(-128, 128, (8, 16))
    big_act = np.repeat(np.array([127, -127, -128, 127]), D).reshape(M_MAX, D)
    big_act = big_act.astype(np.int8)
    big_act[:, :16] = rng.integers(-128, 128, (M_MAX, 16))
    return [
        MmaCase("per-element epilogue", big_w, big_act, rng),
        MmaCase(
            "vector epilogue",
            rng.integers(-8, 9, (8, D)).astype(np.int8),
            rng.integers(-8, 9, (M_MAX, D)).astype(np.int8),
            rng,
        ),
    ]


def attn_cases(rng):
    """A plain pair of blocks, then block 0 at block_sv's two edges."""
    rows = 2 * KV_TB

    def base():
        return (
            rng.integers(-127, 128, (N_HEADS_Q, HEAD_D)).astype(np.int8),
            rng.uniform(0.004, 0.02, N_HEADS_Q).astype(np.float32),
            rng.integers(-127, 128, (N_HEADS_KV, rows, HEAD_D)).astype(np.int8),
            rng.integers(-128, 128, (N_HEADS_KV, rows, HEAD_D)).astype(np.int8),
            rng.uniform(0.004, 0.02, (N_HEADS_KV, rows)).astype(np.float32),
            rng.uniform(0.002, 0.05, (N_HEADS_KV, rows)).astype(np.float32),
        )

    cases = [AttnCase("plain", *base(), rng)]

    # Every Q head of KV head 0 aligned with block 0's row 5 alone: its dot is
    # 64 * 127^2, the others' near 0, so their scores trail by about 80.
    q, q_scales, k, v, ks, vs = base()
    sign = np.where(rng.integers(0, 2, HEAD_D) == 0, -1, 1)
    q[:REP] = 127 * sign
    q_scales[:REP] = 0.02
    k[0, :KV_TB] = rng.integers(-8, 9, (KV_TB, HEAD_D))
    k[0, 5] = 127 * sign
    ks[0, :KV_TB] = 0.031
    cases.append(AttnCase("dropped rows", q, q_scales, k, v, ks, vs, rng))

    # Block 0's v_scale near 2^119: its largest pv is at least 2^113, and its
    # sv, at most 127 * 2^119, stays finite.
    q, q_scales, k, v, ks, vs = base()
    vs[0, :KV_TB] = np.ldexp(rng.uniform(1.0, 1.01, KV_TB), 119).astype(np.float32)
    cases.append(AttnCase("top of range", q, q_scales, k, v, ks, vs, rng))
    return cases


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    opts = p.parse_args()
    rng = np.random.default_rng(opts.seed)
    fails = 0

    session = DispatchSession(mma_group)
    for case in mma_cases(rng):
        took = case.per_element()
        want = case.name == "per-element epilogue"
        bad, ms = case.run(session)
        ok = took == want and bad == 0
        fails += not ok
        print(
            f"mma {case.name}: {bad} words differ, sums "
            f"{'past' if took else 'within'} 2^23, {ms:.2f} ms: "
            f"{'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    del session

    session = DispatchSession(attn_pair, ROW0=OWN_ROW)
    for case, want in zip(
        attn_cases(rng), (["plain", "plain"], ["drop", "plain"], ["top", "plain"])
    ):
        paths = case.paths()
        bad, ms = case.run(session)
        ok = paths == want and bad == 0
        fails += not ok
        print(
            f"attention {case.name}: {bad} words differ, blocks take {paths}, "
            f"{ms:.2f} ms: {'PASS' if ok else 'FAIL'}",
            flush=True,
        )
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
