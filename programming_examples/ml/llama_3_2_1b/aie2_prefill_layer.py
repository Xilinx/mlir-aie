#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""One Llama 3.2 1B layer's prefill of M=64 prompt rows on the NPU.

The 64 rows are the prompt's positions p0..p0 + 63, one KV block (block j,
p0 = 64 j); the KV cache holds the rows before them. The layer runs in
phases through a DDR `stage` buffer (PrefillStage), on all 32 cores:

- the GEMM array (aie2_prefill.GemmArray, row 2) runs qkv, o, g, u, d;
- 4 glue cores (GR, row 5) run each row's norms (llama_rmsnorm_f32), its
  fp32 residual adds and its attention output's quantization
  (llama_af_quant), 16 rows each;
- 2 (GQ) rope each row's Q and K and quantize Q per head and K and V into
  the row's KV cache row (llama_prefill_qkv_row), 32 rows each, writing the
  attention's per-KV-head chunks and the cache rows of block j;
- 2 (GS) run SiLU(gate) * up and its quantization (llama_silu_*), 32 rows
  each;
- 16 attention cores (rows 3 and 4) run the decode's block attention, two
  per KV head as the decode pairs them: the queries in 4 sub-batches of
  16, each over blocks 0..j, the diagonal block j up to each query's own
  row (llama_attn_block_q), so every query's fold is attention_blocks'.

Every kernel is the decode's arithmetic, so each row matches
llama_oracle.Layer.forward bit for bit. Per dispatch, as ScratchpadParameters:
the blocks attention reads (j + 1), the first row's RoPE row and block j's
offset in the cache.
"""

import argparse
import sys

import llama_kernels
import numpy as np
from aie.dialects.arith import (  # pyright: ignore[reportMissingImports]
    CmpIPredicate,
    addi,
    cmpi,
    constant,
    index_cast,
    muli,
    remsi,
    select,
)
from aie.extras.types import i32, index  # pyright: ignore[reportMissingImports]
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer,
    CompileTime,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import if_, range_
from aie.iron.device import NPU2, Tile
from aie.utils.hostruntime.argparse import add_compile_args
from aie.utils.hostruntime.cli import run_design_cli
from aie2_decode import (
    ACT_BYTES,
    ATTN_STATE_BYTES,
    HACT_BYTES,
    KV_BLOCK_WORDS,
    KV_HEAD_BLOCK_WORDS,
    KV_ROW_WORDS,
    NH,
    kv_blocks,
    kv_words,
    wblob_layout,
)
from aie2_gemv_array import N_CORES
from aie2_prefill import PREFILL_PROJS, TABLE_LAYER, GemmArray, M
from llama_config import CS_BYTES, HD, HEAD_D, N_HEADS_KV, D

GR = 4
GQ = 2
GS = 2
SUB = 16
CHUNK_STRIDE = 448
SILU_CHUNK = 512
QKV_FP = N_CORES * (4 + 2) * HEAD_D
GR_ROWS, GQ_ROWS, GS_ROWS = M // GR, M // GQ, M // GS


class PrefillStage:
    """Byte offsets of a layer's regions in the `stage` buffer.

    Attributes:
        at: Each region's offset by name: the residual rows entering the
            layer (x_in), after attention (x1) and leaving it (x_out),
            [M][D] fp32; the norms' gammas (bf16, padded to a row); the
            projections' activation rows (a_qkv, a_o, a_gu, a_d, each row
            [K int8 | fp32 scale | pad]); their outputs, row-major (out_qkv
            [M][core][q 256 | k 64 | v 64], out_o, out_gu [M][g | u],
            out_d); the attention's chunks [KV head][row][CHUNK_STRIDE] and
            its outputs, sv [M][D] fp32.
        total: Bytes.
    """

    def __init__(self):
        sizes = {
            "x_in": M * D * 4,
            "x1": M * D * 4,
            "x_out": M * D * 4,
            "gamma_in": D * 4,
            "gamma_post": D * 4,
            "a_qkv": M * ACT_BYTES + 128,
            "a_o": M * ACT_BYTES + 128,
            "a_gu": M * ACT_BYTES + 128,
            "a_d": M * HACT_BYTES + 128,
            "out_qkv": M * QKV_FP * 4,
            "out_o": M * D * 4,
            "out_gu": M * 2 * HD * 4,
            "out_d": M * D * 4,
            # A sub-batch's chunk reads run KV_HEAD_BLOCK_WORDS past its start.
            "chunks": N_HEADS_KV * M * CHUNK_STRIDE + KV_HEAD_BLOCK_WORDS * 4,
            "sv": M * D * 4,
        }
        self.at = {}
        at = 0
        for name, n in sizes.items():
            self.at[name] = at
            at += -(-n // 64) * 64
        self.total = at

    def words(self, name):
        """A region's offset in int32 words."""
        return self.at[name] // 4


def tap(words, offset, sizes, strides):
    """A TensorAccessPattern on a 1-D tensor of int32 words."""
    return TensorAccessPattern(
        tensor_dims=[words], offset=offset, sizes=sizes, strides=strides
    )


@jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1"])
def build_layer(
    *,
    LAYER: CompileTime[int] = 0,
    W_LAYERS: CompileTime[int] = 1,
    T: CompileTime[int] = 128,
):
    """Layer LAYER's prefill of M rows at block j.

    Args:
        LAYER: The layer, in the weight blob, trailer table and KV cache.
        W_LAYERS: Layers of the weight blob and KV cache (aie2_decode's
            N_LAYERS).
        T: The KV cache's rows and the blob's RoPE table rows.
    """
    stage = PrefillStage()
    stage_words = stage.total // 4
    off_rope, w_total = wblob_layout(W_LAYERS, T)
    kv_total = kv_words(T, W_LAYERS)
    kv_region = kv_blocks(T) * KV_BLOCK_WORDS

    p_nb = ScratchpadParameter("nb", np.int32)
    p_cs = ScratchpadParameter("cs_off", np.int32)
    p_kv = ScratchpadParameter("kv_row", np.int32)

    array = GemmArray(PREFILL_PROJS, 1)
    workers = list(array.workers)

    t_row = np.ndarray[(D,), np.dtype[np.int32]]
    t_act = np.ndarray[(ACT_BYTES // 4,), np.dtype[np.int32]]
    t_hact = np.ndarray[(HACT_BYTES // 4,), np.dtype[np.int32]]

    # GR: rows in through two splits (a row, then a second operand: a
    # gamma or a projection's row), out through two joins.
    of_gr = [
        ObjectFifo(np.ndarray[(GR * D,), np.dtype[np.int32]], name=f"gr{i}")
        for i in range(2)
    ]
    gr_in = [
        f.cons().split(
            [k * D for k in range(GR)],
            obj_types=[t_row] * GR,
            names=[f"gr{i}_{k}" for k in range(GR)],
            tile=Tile((0, 4)[i], 1),
        )
        for i, f in enumerate(of_gr)
    ]
    of_grh = ObjectFifo(
        np.ndarray[(GR * ACT_BYTES // 4,), np.dtype[np.int32]], name="grh"
    )
    grh = of_grh.prod().join(
        [k * ACT_BYTES // 4 for k in range(GR)],
        obj_types=[t_act] * GR,
        names=[f"grh_{k}" for k in range(GR)],
        tile=Tile(0, 1),
    )
    of_grx = ObjectFifo(np.ndarray[(GR * D,), np.dtype[np.int32]], name="grx")
    grx = of_grx.prod().join(
        [k * D for k in range(GR)],
        obj_types=[t_row] * GR,
        names=[f"grx_{k}" for k in range(GR)],
        tile=Tile(4, 1),
    )
    k_norm = llama_kernels.rmsnorm().bind("llama_rmsnorm_f32", [t_row, t_row, t_act])
    k_af = llama_kernels.gqa_glue().bind("llama_af_quant", [t_row, t_act])
    kp = llama_kernels.prefill_glue()
    k_add = kp.bind("llama_prefill_add", [t_row, t_row, t_row])

    def gr_body(c_x, c_y, c_h, c_o, kn, ka, kadd):
        def norm():
            g = c_y.acquire(1)
            for _ in range_(GR_ROWS):
                x = c_x.acquire(1)
                h = c_h.acquire(1)
                kn(x, g, h)
                h[ACT_BYTES // 4 - 2] = 0
                h[ACT_BYTES // 4 - 1] = 0
                c_h.release(1)
                c_x.release(1)
            c_y.release(1)

        def add():
            for _ in range_(GR_ROWS):
                x = c_x.acquire(1)
                y = c_y.acquire(1)
                o = c_o.acquire(1)
                kadd(x, y, o)
                c_o.release(1)
                c_y.release(1)
                c_x.release(1)

        norm()
        for _ in range_(GR_ROWS):
            sv = c_x.acquire(1)
            h = c_h.acquire(1)
            ka(sv, h)
            h[ACT_BYTES // 4 - 2] = 0
            h[ACT_BYTES // 4 - 1] = 0
            c_h.release(1)
            c_x.release(1)
        add()
        norm()
        add()

    workers += [
        Worker(
            gr_body,
            [
                gr_in[0][k].cons(),
                gr_in[1][k].cons(),
                grh[k].prod(),
                grx[k].prod(),
                k_norm,
                k_af,
                k_add,
            ],
            tile=Tile(k, 5),
        )
        for k in range(GR)
    ]

    # GQ: a row's qkv and its RoPE row in through splits; its chunks and
    # cache rows out.
    t_fp = np.ndarray[(QKV_FP,), np.dtype[np.int32]]
    t_cs = np.ndarray[(CS_BYTES // 4,), np.dtype[np.int32]]
    t_ch = np.ndarray[(N_HEADS_KV * CHUNK_STRIDE // 4,), np.dtype[np.int32]]
    t_rw = np.ndarray[(N_HEADS_KV * KV_ROW_WORDS,), np.dtype[np.int32]]
    of_gq = ObjectFifo(np.ndarray[(GQ * QKV_FP,), np.dtype[np.int32]], name="gq")
    gq_in = of_gq.cons().split(
        [k * QKV_FP for k in range(GQ)],
        obj_types=[t_fp] * GQ,
        names=[f"gq_{k}" for k in range(GQ)],
        tile=Tile(3, 1),
    )
    of_gqcs = ObjectFifo(
        np.ndarray[(GQ * CS_BYTES // 4,), np.dtype[np.int32]], name="gqcs"
    )
    gq_cs = of_gqcs.cons().split(
        [k * CS_BYTES // 4 for k in range(GQ)],
        obj_types=[t_cs] * GQ,
        names=[f"gqcs_{k}" for k in range(GQ)],
        tile=Tile(3, 1),
    )
    of_gqch = [ObjectFifo(t_ch, name=f"gqch{k}") for k in range(GQ)]
    of_gqrw = [ObjectFifo(t_rw, name=f"gqrw{k}") for k in range(GQ)]
    k_qkv = kp.bind("llama_prefill_qkv_row", [t_fp, t_cs, t_ch, t_rw])

    def gq_body(c_fp, c_cs, c_ch, c_rw, k):
        for _ in range_(GQ_ROWS):
            fp = c_fp.acquire(1)
            cs = c_cs.acquire(1)
            ch = c_ch.acquire(1)
            rw = c_rw.acquire(1)
            k(fp, cs, ch, rw)
            c_rw.release(1)
            c_ch.release(1)
            c_cs.release(1)
            c_fp.release(1)

    workers += [
        Worker(
            gq_body,
            [
                gq_in[k].cons(),
                gq_cs[k].cons(),
                of_gqch[k].prod(),
                of_gqrw[k].prod(),
                k_qkv,
            ],
            tile=Tile(4 + k, 5),
        )
        for k in range(GQ)
    ]

    # GS: a row's gate chunks, then its up chunks, in through a split; its
    # quantized SiLU(gate) * up out through a join.
    t_chunk = np.ndarray[(SILU_CHUNK,), np.dtype[np.int32]]
    t_h = np.ndarray[(HD,), np.dtype[np.float32]]
    of_gs = ObjectFifo(np.ndarray[(GS * SILU_CHUNK,), np.dtype[np.int32]], name="gs")
    gs_in = of_gs.cons().split(
        [k * SILU_CHUNK for k in range(GS)],
        obj_types=[t_chunk] * GS,
        names=[f"gs_{k}" for k in range(GS)],
        tile=Tile(7, 1),
    )
    of_gso = ObjectFifo(
        np.ndarray[(GS * HACT_BYTES // 4,), np.dtype[np.int32]], name="gso"
    )
    gs_out = of_gso.prod().join(
        [k * HACT_BYTES // 4 for k in range(GS)],
        obj_types=[t_hact] * GS,
        names=[f"gso_{k}" for k in range(GS)],
        tile=Tile(7, 1),
    )
    ks = llama_kernels.silu_mul(chunk=SILU_CHUNK)
    k_gate = ks.bind("llama_silu_gate", [t_chunk, t_h, np.int32])
    k_up = ks.bind("llama_silu_up", [t_chunk, t_h, np.int32])
    k_quant = ks.bind("llama_silu_quant", [t_h, t_hact])

    def gs_body(c_in, c_out, h, kg, ku, kq):
        for _ in range_(GS_ROWS):
            for kernel in (kg, ku):
                for i in range_(HD // SILU_CHUNK):
                    x = c_in.acquire(1)
                    kernel(x, h, index_cast(i32(), i))
                    c_in.release(1)
            o = c_out.acquire(1)
            kq(h, o)
            o[HACT_BYTES // 4 - 2] = 0
            o[HACT_BYTES // 4 - 1] = 0
            c_out.release(1)

    workers += [
        Worker(
            gs_body,
            [
                gs_in[k].cons(),
                gs_out[k].prod(),
                Buffer(t_h, name=f"gs_h{k}"),
                k_gate,
                k_up,
                k_quant,
            ],
            tile=Tile(6 + k, 5),
        )
        for k in range(GS)
    ]

    # Attention: per half of the KV heads, a sub-batch's chunks, then the
    # live blocks, through a split to the half's 4 heads, both cores of a
    # head taking each; the second core's sv out through a join.
    t_kvh = np.ndarray[(KV_HEAD_BLOCK_WORDS,), np.dtype[np.int32]]
    t_chunks = np.ndarray[(SUB * CHUNK_STRIDE // 4,), np.dtype[np.int32]]
    t_states = np.ndarray[(SUB * ATTN_STATE_BYTES // 4,), np.dtype[np.int32]]
    t_state = np.ndarray[(ATTN_STATE_BYTES // 4,), np.dtype[np.int32]]
    t_sv = np.ndarray[(4 * HEAD_D,), np.dtype[np.int32]]
    of_kv = [
        ObjectFifo(np.ndarray[(KV_BLOCK_WORDS,), np.dtype[np.int32]], name=f"kv{h}")
        for h in range(2)
    ]
    kv_heads = [
        f
        for half in range(2)
        for f in of_kv[half]
        .cons()
        .split(
            [i * KV_HEAD_BLOCK_WORDS for i in range(NH)],
            obj_types=[t_kvh] * NH,
            names=[f"kvh_{half * NH + i}" for i in range(NH)],
            tile=Tile((2, 5)[half], 1),
        )
    ]
    of_sv = [
        ObjectFifo(np.ndarray[(NH * 4 * HEAD_D,), np.dtype[np.int32]], name=f"sv{h}")
        for h in range(2)
    ]
    sv_heads = [
        f
        for half in range(2)
        for f in of_sv[half]
        .prod()
        .join(
            [i * 4 * HEAD_D for i in range(NH)],
            obj_types=[t_sv] * NH,
            names=[f"svh_{half * NH + i}" for i in range(NH)],
            tile=Tile((2, 5)[half], 1),
        )
    ]
    of_part = [ObjectFifo(t_state, depth=1, name=f"part{h}") for h in range(N_HEADS_KV)]
    kt = llama_kernels.attn_block(tb=64)
    k_take = kt.bind("llama_attn_take_batch", [t_kvh, t_chunks, t_states, np.int32])
    k_blk = kt.bind("llama_attn_block_q", [t_chunks, t_kvh, t_states] + [np.int32] * 2)
    k_put = kt.bind("llama_attn_put_q", [t_states, t_state, np.int32])
    k_pair = kt.bind("llama_attn_merge_pair_q", [t_state, t_states, t_sv, np.int32])

    def attn_body(c_kv, c_part, c_sv, chunks, states, nbp, mine, ktk, kb, kout):
        nb = nbp.read()
        last = addi(nb, constant(i32(), -1))
        for s in range_(M // SUB):
            q = c_kv.acquire(1)
            ktk(q, chunks, states, SUB)
            c_kv.release(1)
            row0 = muli(index_cast(i32(), s), constant(i32(), SUB))
            for b in range_(index_cast(index(), nb)):
                bi = index_cast(i32(), b)
                blk = c_kv.acquire(1)
                with if_(
                    cmpi(
                        CmpIPredicate.eq,
                        remsi(bi, constant(i32(), 2)),
                        constant(i32(), mine),
                    )
                ):
                    own = select(
                        cmpi(CmpIPredicate.eq, bi, last), row0, constant(i32(), -1)
                    )
                    for r in range_(SUB):
                        kb(chunks, blk, states, own, index_cast(i32(), r))
                c_kv.release(1)
            for r in range_(SUB):
                part = c_part.acquire(1)
                if mine == 0:
                    kout(states, part, index_cast(i32(), r))
                else:
                    sv = c_sv.acquire(1)
                    kout(part, states, sv, index_cast(i32(), r))
                    c_sv.release(1)
                c_part.release(1)

    for h in range(N_HEADS_KV):
        for mine in range(2):
            workers.append(
                Worker(
                    attn_body,
                    [
                        kv_heads[h].cons(),
                        of_part[h].prod() if mine == 0 else of_part[h].cons(),
                        sv_heads[h].prod() if mine == 1 else None,
                        Buffer(t_chunks, name=f"attn_chunks_{h}_{mine}"),
                        Buffer(t_states, name=f"attn_states_{h}_{mine}"),
                        p_nb,
                        mine,
                        k_take,
                        k_blk,
                        k_put if mine == 0 else k_pair,
                    ],
                    tile=Tile(h, 3 + mine),
                    wait_for_parameters=True,
                )
            )

    fifo_groups = [
        array.fifos,
        [f.prod() for f in of_gr],
        [of_grh.cons(), of_grx.cons()],
        [of_gq.prod(), of_gqcs.prod()],
        [f.cons() for f in of_gqch],
        [f.cons() for f in of_gqrw],
        [of_gs.prod(), of_gso.cons()],
        [f.prod() for f in of_kv],
        [f.cons() for f in of_sv],
    ]
    sizes = [len(g) for g in fifo_groups]
    projs = {p.name: p for p in PREFILL_PROJS}
    act_of = {"qkv": "a_qkv", "o": "a_o", "g": "a_gu", "u": "a_gu", "d": "a_d"}
    out_of = {
        "qkv": ("out_qkv", 0, QKV_FP),
        "o": ("out_o", 0, D),
        "g": ("out_gu", 0, 2 * HD),
        "u": ("out_gu", HD, 2 * HD),
        "d": ("out_d", 0, D),
    }

    def sequence(wblob, table, kv, st, *fifos):
        it = iter(fifos)
        gemm, gr, gr_out, gq, gq_ch, gq_rw, gs, kv_in, sv_out = (
            [next(it) for _ in range(n)] for n in sizes
        )
        w = (wblob, w_total // 4, LAYER)
        tbl = (table, W_LAYERS * TABLE_LAYER // 4, LAYER)

        def gemms(names):
            run = [projs[n] for n in names]
            array.run(
                gemm,
                run,
                [(w, tbl, (st, stage_words, stage.at[act_of[n]])) for n in names],
                [
                    (
                        st,
                        stage_words,
                        stage.words(out_of[n][0]) + out_of[n][1],
                        out_of[n][2],
                    )
                    for n in names
                ],
            )

        def gr_rows(region):
            return tap(
                stage_words,
                stage.words(region),
                [GR_ROWS, GR, 4, 512],
                [D, GR_ROWS * D, 512, 1],
            )

        def gr_phase(x, y, out, h_out):
            """GR's phase: rows of x, with y's rows (or a gamma, or nothing),
            to out (h rows [ACT_BYTES] or x rows [D fp32])."""
            tg = TaskGroup()
            gr[0].fill(st, tap=gr_rows(x), group=tg)
            if y in ("gamma_in", "gamma_post"):
                gr[1].fill(
                    st,
                    tap=tap(
                        stage_words,
                        stage.words(y),
                        [1, GR, 4, 512],
                        [0, 0, 512, 1],
                    ),
                    group=tg,
                )
            elif y is not None:
                gr[1].fill(st, tap=gr_rows(y), group=tg)
            if h_out:
                act = ACT_BYTES // 4
                gr_out[0].drain(
                    st,
                    tap=tap(
                        stage_words,
                        stage.words(out),
                        [1, GR_ROWS, GR, act],
                        [0, act, GR_ROWS * act, 1],
                    ),
                    group=tg,
                    wait=True,
                )
            else:
                gr_out[1].drain(st, tap=gr_rows(out), group=tg, wait=True)
            tg.finish()

        gr_phase("x_in", "gamma_in", "a_qkv", True)
        gemms(["qkv"])

        tg = TaskGroup()
        gq[0].fill(
            st,
            tap=tap(
                stage_words,
                stage.words("out_qkv"),
                [GQ_ROWS, GQ, QKV_FP // 512, 512],
                [QKV_FP, GQ_ROWS * QKV_FP, 512, 1],
            ),
            group=tg,
        )
        cs = CS_BYTES // 4
        gq[1].fill(
            wblob,
            tap=tap(
                w_total // 4,
                off_rope // 4,
                [1, GQ_ROWS, GQ, cs],
                [0, cs, GQ_ROWS * cs, 1],
            ),
            offset_parameter=p_cs,
            group=tg,
        )
        ch = CHUNK_STRIDE // 4
        for k in range(GQ):
            gq_ch[k].drain(
                st,
                tap=tap(
                    stage_words,
                    stage.words("chunks") + k * GQ_ROWS * ch,
                    [1, GQ_ROWS, N_HEADS_KV, ch],
                    [0, ch, M * ch, 1],
                ),
                group=tg,
                wait=True,
            )
            gq_rw[k].drain(
                kv,
                tap=tap(
                    kv_total,
                    LAYER * 2 * kv_region + k * GQ_ROWS * KV_ROW_WORDS,
                    [GQ_ROWS, 2, NH, KV_ROW_WORDS],
                    [KV_ROW_WORDS, kv_region, KV_HEAD_BLOCK_WORDS, 1],
                ),
                offset_parameter=p_kv,
                group=tg,
                wait=True,
            )
        tg.finish()

        for s in range(M // SUB):
            tg = TaskGroup()
            for half in range(2):
                kv_in[half].fill(
                    st,
                    tap=tap(
                        stage_words,
                        stage.words("chunks") + (half * NH * M + s * SUB) * ch,
                        [1, NH, 4, KV_HEAD_BLOCK_WORDS // 4],
                        [0, M * ch, KV_HEAD_BLOCK_WORDS // 4, 1],
                    ),
                    group=tg,
                )
                kv_in[half].fill(
                    kv,
                    tap=tap(
                        kv_total,
                        (LAYER * 2 + half) * kv_region,
                        [1, 1, 16, KV_BLOCK_WORDS // 16],
                        [0, 0, KV_BLOCK_WORDS // 16, 1],
                    ),
                    length_parameter=p_nb,
                    group=tg,
                )
                sv_out[half].drain(
                    st,
                    tap=tap(
                        stage_words,
                        stage.words("sv") + s * SUB * D + half * NH * 4 * HEAD_D,
                        [1, SUB, 2, 512],
                        [0, D, 512, 1],
                    ),
                    group=tg,
                    wait=True,
                )
            tg.finish()

        gr_phase("sv", None, "a_o", True)
        gemms(["o"])
        gr_phase("x_in", "out_o", "x1", False)
        gr_phase("x1", "gamma_post", "a_gu", True)
        gemms(["g", "u"])

        tg = TaskGroup()
        gs[0].fill(
            st,
            tap=tap(
                stage_words,
                stage.words("out_gu"),
                [GS_ROWS, 2 * HD // SILU_CHUNK, GS, SILU_CHUNK],
                [2 * HD, SILU_CHUNK, GS_ROWS * 2 * HD, 1],
            ),
            group=tg,
        )
        hact = HACT_BYTES // 4
        gs[1].drain(
            st,
            tap=tap(
                stage_words,
                stage.words("a_d"),
                [GS_ROWS, GS, 4, hact // 4],
                [hact, GS_ROWS * hact, hact // 4, 1],
            ),
            group=tg,
            wait=True,
        )
        tg.finish()

        gemms(["d"])
        gr_phase("x1", "out_d", "x_out", False)

    rt = Runtime(
        sequence,
        [
            np.ndarray[(w_total // 4,), np.dtype[np.int32]],
            np.ndarray[(W_LAYERS * TABLE_LAYER // 4,), np.dtype[np.int32]],
            np.ndarray[(kv_total,), np.dtype[np.int32]],
            np.ndarray[(stage_words,), np.dtype[np.int32]],
            *[f for g in fifo_groups for f in g],
        ],
    )
    return Program(NPU2(), rt, workers=workers).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B prefill layer")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--w-layers", type=int, default=1)
    p.add_argument("--t", type=int, default=128)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build_layer,
        opts,
        compile_kwargs=lambda o: dict(LAYER=o.layer, W_LAYERS=o.w_layers, T=o.t),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
