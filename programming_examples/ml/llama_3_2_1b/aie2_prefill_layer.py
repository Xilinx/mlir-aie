#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Llama 3.2 1B's prefill of up to M=64 prompt rows on the NPU, through N layers.

A pass's rows are the prompt's positions p0..p0 + rows - 1 of one KV block
(block j, p0 = 64 j), rows a multiple of ROW_STEP chosen per pass; the KV
cache holds the rows before them. Each row's layer-0 input is its token's
embed seed (llama_embed_seed), made on the device from the token's lm_head
weight slot, read straight from the lm_head blob `lmw`, and from the token's
id in `ids`. Each layer runs in phases through a DDR `stage` buffer
(PrefillStage), on all 32 cores, each glue group's cores taking the rows in
turn (core k of n: rows n i + k):

- the GEMM array (aie2_prefill.GemmArray, row 2) runs qkv, o, g, u, d, its
  mmul on M rows and its epilogue and output on the pass's;
- GR glue cores (row 5) make the rows' embed seeds, and run each row's
  norms (llama_rmsnorm_f32), its fp32 residual adds and its attention
  output's quantization (llama_af_quant);
- GQ rope each row's Q and K and quantize Q per head and K and V into the
  row's KV cache row (llama_prefill_qkv_row), writing the attention's
  per-KV-head chunks and the cache rows of block j;
- GS run SiLU(gate) * up and its quantization (llama_silu_*);
- 16 attention cores (rows 3 and 4) run the decode's block attention, two
  per KV head as the decode pairs them: the queries in sub-batches of SUB,
  each over blocks 0..j, the diagonal block j up to each query's own row
  (llama_attn_block_q), so every query's fold is attention_blocks'.

Every kernel is the decode's arithmetic, so each row matches
llama_oracle.Layer.forward bit for bit. A pass is commands queued back to
back (pass_commands), behind a prefill:configure command that loads the
device's PDI when the array held another device: embed commands
(prefill:embed{r}), each making r of its rows' seeds from their tokens'
slots (slot{k}) and ids (ids_off) into xa from row xa_off / D, then one
command per layer (prefill:layer{i}), so a command's work is at most one
layer's of M rows whatever the prompt's length. A layer command's
ScratchpadParameters: the blocks attention reads (j + 1, nb, and per further
sub-batch nb{s} and sub{s}, 0 past the rows), the rows a GQ core takes
(gq_rows), the SiLU row groups less one (gs_rep), the GEMMs' output parts
of OUT_ROWS less one (gemm_rep), the first row's RoPE row (rope_off) and
block j's offset in the cache (kv_row). Every command syncs the cores'
pass_rows and nb; an embed or configure command leaves nb 0, which the
GEMM, GQ, GS and attention cores skip, and configure's pass_rows 0, which
the GR cores skip.
"""

import argparse
import sys
from functools import partial

import llama_kernels
import numpy as np
from aie.dialects.aiex import npu_load_pdi  # pyright: ignore[reportMissingImports]
from aie.dialects.arith import (  # pyright: ignore[reportMissingImports]
    CmpIPredicate,
    addi,
    andi,
    cmpi,
    constant,
    divsi,
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
    DeviceConfiguration,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    jit,
)
from aie.iron.controlflow import else_, if_, range_
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
    LMW_GAMMA_BYTES,
    NH,
    WLM_SLOT,
    WLM_TOTAL,
    kv_blocks,
    kv_words,
    wblob_layout,
)
from aie2_gemv_array import N_CORES
from aie2_prefill import OUT_ROWS, PREFILL_PROJS, TABLE_LAYER, GemmArray, M
from llama_config import CS_BYTES, HD, HEAD_D, N_HEADS_KV, N_TILE, D

GR = 2
GQ = 2
GS = 4
SUB = 16
CHUNK_STRIDE = 448
SILU_CHUNK = 512
QKV_FP = N_CORES * (4 + 2) * HEAD_D
PREFILL = "prefill"
CONFIGURE = f"{PREFILL}:configure"
SLOT_WORDS = WLM_SLOT // 4
LMW_WORDS = (LMW_GAMMA_BYTES + WLM_TOTAL) // 4
# A pass's rows come in multiples of ROW_STEP, so every glue group's cores
# take as many rows (the SiLU cores, GS, the most of them).
ROW_STEP = GS
SUB_BATCHES = M // SUB
# Rows an embed command makes seeds for, at most: its slot offsets, its first
# row (xa_off, ids_off) and every core's nb and rows fill a command's 32
# ScratchpadParameters.
EMBED_ROWS = 28


def ids_words(t):
    """`ids`' int32 words for a T-row build: T ids, then a row's slack."""
    return t + D


def pass_commands(tokens, block, n_layers, ids_off):
    """A pass's commands and their ScratchpadParameters.

    Args:
        tokens: The pass's rows' token ids, in row order: a multiple of
            ROW_STEP, at most M.
        block: Its KV block, j.
        n_layers: The build's N_LAYERS.
        ids_off: Where `tokens` start in `ids`.

    Returns:
        [(command, {parameter: value})] in order: embed{r} commands of r
        rows each (EMBED_ROWS, then the rest), each row's token's lm_head
        slot a word offset in `lmw`, then a layer command per layer.
    """
    rows = len(tokens)
    if not rows or rows % ROW_STEP or rows > M:
        raise ValueError(f"a pass of {rows} rows, not a multiple of {ROW_STEP} to {M}")
    out = []
    for first in range(0, rows, EMBED_ROWS):
        r = min(EMBED_ROWS, rows - first)
        params = {
            f"slot{k}": (LMW_GAMMA_BYTES + tok // N_TILE * WLM_SLOT) // 4
            for k, tok in enumerate(tokens[first : first + r])
        }
        params.update(pass_rows=r, ids_off=ids_off + first, xa_off=first * D)
        out.append((f"{PREFILL}:embed{r}", params))
    subs = -(-rows // SUB)
    layer = dict(
        nb=block + 1,
        pass_rows=rows,
        gq_rows=rows // GQ,
        gs_rep=rows // GS - 1,
        gemm_rep=-(-rows // OUT_ROWS) - 1,
        rope_off=block * M * CS_BYTES // 4,
        kv_row=block * KV_BLOCK_WORDS,
    )
    for s in range(1, SUB_BATCHES):
        layer[f"nb{s}"] = (block + 1) * (s < subs)
        layer[f"sub{s}"] = int(s < subs)
    out += [(f"{PREFILL}:layer{i}", layer) for i in range(n_layers)]
    return out


class PrefillStage:
    """Byte offsets of a dispatch's regions in the `stage` buffer.

    Args:
        n_layers: Layers a dispatch runs.

    Attributes:
        at: Each region's offset by name: the residual rows between layers
            (xa, xb: layer l of a dispatch reads xa and writes xb for even l,
            the other way for odd) and after attention (x1), [M][D] fp32;
            per layer its norms' gammas (gammas, [layer][in, post][D words],
            bf16 then zeros); the projections' activation rows (a_qkv, a_o,
            a_gu, a_d, each row [K int8 | fp32 scale | pad]); their outputs,
            row-major (out_qkv [M][core][q 256 | k 64 | v 64], out_o, out_gu
            [M][g | u], out_d); the attention's chunks [KV head][row]
            [CHUNK_STRIDE] and its outputs, sv [M][D] fp32.
        total: Bytes.
    """

    def __init__(self, n_layers):
        sizes = {
            "xa": M * D * 4,
            "xb": M * D * 4,
            "x1": M * D * 4,
            "gammas": n_layers * 2 * D * 4,
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
            # A row read into a slot-sized object runs on past its row.
            "tail": WLM_SLOT,
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

    def gamma(self, layer, post):
        """Dispatch layer `layer`'s input (post 0) or post-attention (post 1)
        norm gamma's offset, in words."""
        return self.words("gammas") + (2 * layer + post) * D


def tap(words, offset, sizes, strides):
    """A TensorAccessPattern on a 1-D tensor of int32 words."""
    return TensorAccessPattern(
        tensor_dims=[words], offset=offset, sizes=sizes, strides=strides
    )


def prefill_configuration(
    *, N_LAYERS: int, LAYER: int, W_LAYERS: int, T: int
) -> DeviceConfiguration:
    """Layers LAYER..LAYER + N_LAYERS - 1's prefill of up to M rows at block
    j, as the module docstring's commands, on a DeviceConfiguration named
    PREFILL; its configure command loads the device's PDI, so passes may
    follow another device's commands in one ELF.

    Args:
        N_LAYERS: Layers a pass runs.
        LAYER: The first, in the weight blob, trailer table and KV cache.
        W_LAYERS: Layers of the weight blob and KV cache (aie2_decode's
            N_LAYERS).
        T: The KV cache's rows and the blob's RoPE table rows; `ids` holds
            T rows' ids.

    Returns:
        The DeviceConfiguration.
    """
    stage = PrefillStage(N_LAYERS)
    stage_words = stage.total // 4
    off_rope, w_total = wblob_layout(W_LAYERS, T)
    kv_total = kv_words(T, W_LAYERS)
    kv_region = kv_blocks(T) * KV_BLOCK_WORDS

    p_nb = ScratchpadParameter("nb", np.int32)
    p_rope = ScratchpadParameter("rope_off", np.int32)
    p_kv = ScratchpadParameter("kv_row", np.int32)
    p_ids = ScratchpadParameter("ids_off", np.int32)
    p_rows = ScratchpadParameter("pass_rows", np.int32)
    p_gq_rows = ScratchpadParameter("gq_rows", np.int32)
    p_gs_rep = ScratchpadParameter("gs_rep", np.int32)
    p_gemm_rep = ScratchpadParameter("gemm_rep", np.int32)
    p_xa = ScratchpadParameter("xa_off", np.int32)
    p_slot = [ScratchpadParameter(f"slot{k}", np.int32) for k in range(EMBED_ROWS)]
    p_sub_nb = [ScratchpadParameter(f"nb{s}", np.int32) for s in range(1, SUB_BATCHES)]
    p_sub = [ScratchpadParameter(f"sub{s}", np.int32) for s in range(1, SUB_BATCHES)]

    array = GemmArray(PREFILL_PROJS, N_LAYERS, rows=(p_nb, p_rows, p_gemm_rep))
    workers = list(array.workers)

    t_row = np.ndarray[(D,), np.dtype[np.int32]]
    t_slot = np.ndarray[(SLOT_WORDS,), np.dtype[np.int32]]
    t_act = np.ndarray[(ACT_BYTES // 4,), np.dtype[np.int32]]
    t_hact = np.ndarray[(HACT_BYTES // 4,), np.dtype[np.int32]]

    # GR: rows in through two splits (a row or a token's slot, then a
    # second operand: a gamma, a projection's row or the ids), out through
    # two joins.
    gr_sizes = (SLOT_WORDS, D)
    of_gr = [
        ObjectFifo(np.ndarray[(GR * n,), np.dtype[np.int32]], name=f"gr{i}")
        for i, n in enumerate(gr_sizes)
    ]
    gr_in = [
        of_gr[i]
        .cons()
        .split(
            [k * n for k in range(GR)],
            obj_types=[t] * GR,
            names=[f"gr{i}_{k}" for k in range(GR)],
            tile=Tile((0, 4)[i], 1),
        )
        for i, (n, t) in enumerate(zip(gr_sizes, (t_slot, t_row)))
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
    k_norm = llama_kernels.rmsnorm().bind("llama_rmsnorm_f32", [t_slot, t_row, t_act])
    k_af = llama_kernels.gqa_glue().bind("llama_af_quant", [t_slot, t_act])
    kp = llama_kernels.prefill_glue()
    k_add = kp.bind("llama_prefill_add", [t_slot, t_row, t_row])
    k_embed = llama_kernels.topk_sample().bind(
        "llama_embed_seed", [t_slot, t_row, np.int32]
    )

    def gr_body(c_x, c_y, c_h, c_o, nbp, rowsp, core, kn, ka, kadd, ke):
        # A command's rows, GR at a time: core k takes rows GR i + k.
        rows = rowsp.read()
        pairs = index_cast(index(), divsi(rows, constant(i32(), GR)))

        def norm():
            g = c_y.acquire(1)
            for _ in range_(pairs):
                x = c_x.acquire(1)
                h = c_h.acquire(1)
                kn(x, g, h)
                h[ACT_BYTES // 4 - 2] = 0
                h[ACT_BYTES // 4 - 1] = 0
                c_h.release(1)
                c_x.release(1)
            c_y.release(1)

        def add():
            for _ in range_(pairs):
                x = c_x.acquire(1)
                y = c_y.acquire(1)
                o = c_o.acquire(1)
                kadd(x, y, o)
                c_o.release(1)
                c_y.release(1)
                c_x.release(1)

        with if_(cmpi(CmpIPredicate.eq, nbp.read(), constant(i32(), 0))) as embed:
            # Each row's token's embed seed, from its slot: row id % 4 of it.
            # No rows: the configure command's parameter sync.
            with if_(cmpi(CmpIPredicate.sgt, rows, constant(i32(), 0))):
                ids = c_y.acquire(1)
                for i in range_(pairs):
                    slot = c_x.acquire(1)
                    o = c_o.acquire(1)
                    tok = ids[
                        addi(muli(i, constant(index(), GR)), constant(index(), core))
                    ]
                    ke(slot, o, andi(tok, constant(i32(), 3)))
                    c_o.release(1)
                    c_x.release(1)
                c_y.release(1)
        with else_(embed):
            norm()
            for _ in range_(pairs):
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
                p_nb,
                p_rows,
                k,
                k_norm,
                k_af,
                k_add,
                k_embed,
            ],
            tile=Tile(k, 5),
            wait_for_parameters=True,
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
    # Split per KV half in a memtile: one drain striding both halves would
    # pass the shim BD's 2^20-word stride once T reaches 7744.
    gq_halves = [
        f
        for k in range(GQ)
        for f in of_gqrw[k]
        .cons()
        .split(
            [0, NH * KV_ROW_WORDS],
            obj_types=[np.ndarray[(NH * KV_ROW_WORDS,), np.dtype[np.int32]]] * 2,
            names=[f"gqrw{k}_{half}" for half in range(2)],
            tile=Tile((1, 6)[k], 1),
        )
    ]
    k_qkv = kp.bind("llama_prefill_qkv_row", [t_fp, t_cs, t_ch, t_rw])

    def gq_body(c_fp, c_cs, c_ch, c_rw, nbp, rowsp, k):
        # A layer command's rows, GQ at a time: core k takes rows GQ i + k.
        with if_(cmpi(CmpIPredicate.sgt, nbp.read(), constant(i32(), 0))):
            n = divsi(rowsp.read(), constant(i32(), GQ))
            for _ in range_(index_cast(index(), n)):
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
                p_nb,
                p_rows,
                k_qkv,
            ],
            tile=Tile(GR + k, 5),
            wait_for_parameters=True,
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

    def gs_body(c_in, c_out, h, nbp, rowsp, kg, ku, kq):
        # A layer command's rows, GS at a time: core k takes rows GS i + k.
        with if_(cmpi(CmpIPredicate.sgt, nbp.read(), constant(i32(), 0))):
            n = divsi(rowsp.read(), constant(i32(), GS))
            for _ in range_(index_cast(index(), n)):
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
                p_nb,
                p_rows,
                k_gate,
                k_up,
                k_quant,
            ],
            tile=Tile(GR + GQ + k, 5),
            wait_for_parameters=True,
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

    def attn_body(c_kv, c_part, c_sv, chunks, states, nbp, rowsp, mine, ktk, kb, kout):
        nb = nbp.read()
        last = addi(nb, constant(i32(), -1))
        # The rows' sub-batches of SUB queries, the last's queries past the
        # rows computed and not read.
        subs = divsi(addi(rowsp.read(), constant(i32(), SUB - 1)), constant(i32(), SUB))
        # nb 0: an embed command's parameter sync, which carries no layer.
        with if_(cmpi(CmpIPredicate.sgt, nb, constant(i32(), 0))):
            for t in range_(index_cast(index(), subs)):
                q = c_kv.acquire(1)
                ktk(q, chunks, states, SUB)
                c_kv.release(1)
                row0 = muli(index_cast(i32(), t), constant(i32(), SUB))
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
                        p_rows,
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
        [f.cons() for f in gq_halves],
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
    ids_total = ids_words(T)
    ch = CHUNK_STRIDE // 4
    cs = CS_BYTES // 4
    act = ACT_BYTES // 4
    hact = HACT_BYTES // 4

    def sequence(kind, i, wblob, table, kv, st, lmw, ids, *fifos):
        it = iter(fifos)
        gemm, gr, gr_out, gq, gq_ch, gq_rw, gs, kv_in, sv_out = (
            [next(it) for _ in range(n)] for n in sizes
        )

        def gemms(names, layer):
            array.run(
                gemm,
                [projs[n] for n in names],
                [
                    (
                        (wblob, w_total // 4, layer),
                        (table, W_LAYERS * TABLE_LAYER // 4, layer),
                        (st, stage_words, stage.at[act_of[n]]),
                    )
                    for n in names
                ],
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

        # A runtime length continues a transfer's third dimension, so each
        # per-row transfer below walks its rows there, a row in the inner two.
        def gr_rows(at, n=D):
            """The GR cores' rows at stage word `at`, n words apart, read
            into slot-sized objects, GR rows an object."""
            return tap(
                stage_words, at, [1, M, 4, SLOT_WORDS // 4], [0, n, SLOT_WORDS // 4, 1]
            )

        def gr_y(at):
            """The GR cores' fp32 rows at stage word `at`, GR rows an object."""
            return tap(stage_words, at, [1, M, 4, 512], [0, D, 512, 1])

        def gr_phase(x, out, h_out, y=None, gamma=False):
            """GR's phase over the command's rows: the rows at x, with the
            rows at y or (gamma) the gamma at y, to out: rows of ACT_BYTES
            (h_out) or fp32 rows."""
            tg = TaskGroup()
            rows = dict(length_parameter=p_rows, group=tg)
            gr[0].fill(st, tap=gr_rows(x), length_unit=SLOT_WORDS, **rows)
            if gamma:
                gr[1].fill(
                    st,
                    tap=tap(stage_words, y, [1, GR, 4, 512], [0, 0, 512, 1]),
                    group=tg,
                )
            elif y is not None:
                gr[1].fill(st, tap=gr_y(y), length_unit=D, **rows)
            if h_out:
                gr_out[0].drain(
                    st,
                    tap=tap(stage_words, out, [1, 1, M, act], [0, 0, act, 1]),
                    length_unit=act,
                    wait=True,
                    **rows,
                )
            else:
                gr_out[1].drain(st, tap=gr_y(out), length_unit=D, wait=True, **rows)
            tg.finish()

        def layer_phases(i, layer, x_in, x_out):
            """Dispatch layer i (the blob's layer `layer`) from rows x_in to
            x_out."""
            gr_phase(x_in, stage.words("a_qkv"), True, stage.gamma(i, 0), gamma=True)
            gemms(["qkv"], layer)

            tg = TaskGroup()
            rows = dict(length_parameter=p_rows, group=tg)
            gq[0].fill(
                st,
                tap=tap(
                    stage_words,
                    stage.words("out_qkv"),
                    [1, M, QKV_FP // 512, 512],
                    [0, QKV_FP, 512, 1],
                ),
                length_unit=QKV_FP,
                **rows,
            )
            gq[1].fill(
                wblob,
                tap=tap(
                    w_total // 4,
                    off_rope // 4,
                    [1, 1, M, cs],
                    [0, 0, cs, 1],
                ),
                offset_parameter=p_rope,
                length_unit=cs,
                **rows,
            )
            # Core k's rows, GQ i + k: gq_rows of them.
            core_rows = dict(length_parameter=p_gq_rows, group=tg)
            for k in range(GQ):
                gq_ch[k].drain(
                    st,
                    tap=tap(
                        stage_words,
                        stage.words("chunks") + k * ch,
                        [1, M // GQ, N_HEADS_KV, ch],
                        [0, GQ * ch, M * ch, 1],
                    ),
                    length_unit=N_HEADS_KV * ch,
                    wait=True,
                    **core_rows,
                )
                for half in range(2):
                    gq_rw[2 * k + half].drain(
                        kv,
                        tap=tap(
                            kv_total,
                            (layer * 2 + half) * kv_region + k * KV_ROW_WORDS,
                            [1, M // GQ, NH, KV_ROW_WORDS],
                            [0, GQ * KV_ROW_WORDS, KV_HEAD_BLOCK_WORDS, 1],
                        ),
                        offset_parameter=p_kv,
                        length_unit=NH * KV_ROW_WORDS,
                        wait=True,
                        **core_rows,
                    )
            tg.finish()

            # Sub-batch s runs when the pass's rows reach it (sub{s}, nb{s}).
            for s in range(SUB_BATCHES):
                gate = {} if s == 0 else dict(length_parameter=p_sub[s - 1])
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
                        **gate,
                    )
                    kv_in[half].fill(
                        kv,
                        tap=tap(
                            kv_total,
                            (layer * 2 + half) * kv_region,
                            [1, 1, 16, KV_BLOCK_WORDS // 16],
                            [0, 0, KV_BLOCK_WORDS // 16, 1],
                        ),
                        length_parameter=p_nb if s == 0 else p_sub_nb[s - 1],
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
                        **gate,
                    )
                tg.finish()

            gr_phase(stage.words("sv"), stage.words("a_o"), True)
            gemms(["o"], layer)
            gr_phase(x_in, stage.words("x1"), False, stage.words("out_o"))
            gr_phase(
                stage.words("x1"),
                stage.words("a_gu"),
                True,
                stage.gamma(i, 1),
                gamma=True,
            )
            gemms(["g", "u"], layer)

            tg = TaskGroup()
            rows = dict(length_parameter=p_rows, group=tg)
            # An object is one chunk of GS rows: the row groups are the
            # tap's iterations, gs_rep + 1 of them.
            gs[0].fill(
                st,
                tap=tap(
                    stage_words,
                    stage.words("out_gu"),
                    [M // GS, 2 * HD // SILU_CHUNK, GS, SILU_CHUNK],
                    [GS * 2 * HD, SILU_CHUNK, 2 * HD, 1],
                ),
                repeat_parameter=p_gs_rep,
                group=tg,
            )
            gs[1].drain(
                st,
                tap=tap(
                    stage_words,
                    stage.words("a_d"),
                    [1, M, 4, hact // 4],
                    [0, hact, hact // 4, 1],
                ),
                length_unit=hact,
                wait=True,
                **rows,
            )
            tg.finish()

            gemms(["d"], layer)
            gr_phase(stage.words("x1"), x_out, False, stage.words("out_d"))

        if kind == "configure":
            npu_load_pdi(device_ref=PREFILL)
            return
        if kind == "layer":
            x = (stage.words("xa"), stage.words("xb"))
            layer_phases(i, LAYER + i, x[i % 2], x[(i + 1) % 2])
            return
        # Embed command of i rows, from row xa_off / D: each row's token's
        # lm_head slot read from lmw (slot{k}: the command's row k), the
        # rows' ids from ids_off, their embed seeds into xa.
        drains = TaskGroup()
        gr[1].fill(
            ids,
            tap=tap(ids_total, 0, [1, GR, 4, 512], [0, 0, 512, 1]),
            offset_parameter=p_ids,
            group=drains,
        )
        gr_out[1].drain(
            st,
            tap=tap(
                stage_words,
                stage.words("xa"),
                [1, i, 4, 512],
                [0, D, 512, 1],
            ),
            offset_parameter=p_xa,
            group=drains,
            wait=True,
        )
        # Groups of 15 keep the shim tile's BDs within its 16; one channel's
        # fills complete in order, so a group awaits its last.
        for at in range(0, i, 15):
            fills = TaskGroup()
            group = range(at, min(at + 15, i))
            for k in group:
                gr[0].fill(
                    lmw,
                    tap=tap(
                        LMW_WORDS,
                        0,
                        [1, 1, 4, SLOT_WORDS // 4],
                        [0, 0, SLOT_WORDS // 4, 1],
                    ),
                    offset_parameter=p_slot[k],
                    group=fills,
                    wait=k == group[-1],
                )
            fills.finish()
        drains.finish()

    arg_types = [
        np.ndarray[(w_total // 4,), np.dtype[np.int32]],
        np.ndarray[(W_LAYERS * TABLE_LAYER // 4,), np.dtype[np.int32]],
        np.ndarray[(kv_total,), np.dtype[np.int32]],
        np.ndarray[(stage_words,), np.dtype[np.int32]],
        np.ndarray[(LMW_WORDS,), np.dtype[np.int32]],
        np.ndarray[(ids_total,), np.dtype[np.int32]],
        *[f for g in fifo_groups for f in g],
    ]
    # One command loads the device's PDI: a load from another command than
    # the last loader's reloads it, about 2.5 ms (measured).
    runtimes = [Runtime(partial(sequence, "configure", 0), arg_types, name="configure")]
    runtimes += [
        Runtime(partial(sequence, "embed", r), arg_types, name=f"embed{r}")
        for r in range(ROW_STEP, EMBED_ROWS + 1, ROW_STEP)
    ] + [
        Runtime(partial(sequence, "layer", i), arg_types, name=f"layer{i}")
        for i in range(N_LAYERS)
    ]
    return DeviceConfiguration(PREFILL, NPU2(), workers=workers, runtimes=runtimes)


@jit(aiecc_flags=["--placer=sa_placer", "--sa-seed=1"])
def build_prefill(
    *,
    N_LAYERS: CompileTime[int] = 1,
    LAYER: CompileTime[int] = 0,
    W_LAYERS: CompileTime[int] = 1,
    T: CompileTime[int] = 128,
):
    """The prefill alone (prefill_configuration's), for its own tests.

    Args:
        N_LAYERS, LAYER, W_LAYERS, T: prefill_configuration's.

    Returns:
        The resolved program.
    """
    config = prefill_configuration(
        N_LAYERS=N_LAYERS, LAYER=LAYER, W_LAYERS=W_LAYERS, T=T
    )
    return Program.compose([config], entry=config.runtimes[0]).resolve_program()


def main():
    p = argparse.ArgumentParser(prog="Llama 3.2 1B prefill")
    add_compile_args(
        p, with_dev=True, default_dev="npu2", dev_choices=("npu2",), with_full_elf=True
    )
    p.add_argument("--n-layers", type=int, default=1)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--w-layers", type=int, default=1)
    p.add_argument("--t", type=int, default=128)
    opts = p.parse_args(sys.argv[1:])
    run_design_cli(
        build_prefill,
        opts,
        compile_kwargs=lambda o: dict(
            N_LAYERS=o.n_layers, LAYER=o.layer, W_LAYERS=o.w_layers, T=o.t
        ),
        device=NPU2(),
    )


if __name__ == "__main__":
    main()
