#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host side of the decode (aie2_decode.py): its buffers, its per-dispatch
ScratchpadParameters, and a session that prefills and generates."""

from __future__ import annotations

import time
from dataclasses import dataclass

import aie.iron as iron
import numpy as np
from aie2_decode import (
    KV_BLOCK_WORDS,
    KV_ROW_WORDS,
    LANE_BYTES,
    LMW_GAMMA_BYTES,
    M_MAX,
    REC_BYTES,
    RROW,
    STAGE_RESID,
    WLM_SLOT,
    WLM_TOTAL,
    StageLayout,
    build,
    kv_blocks,
    kv_words,
    wblob_layout,
)
from aie2_gemv_array import LAYER, MMA_SLOT, N_CORES, sample_param_values
from llama_config import (
    CS_BYTES,
    KV_TB,
    N_HEADS_KV,
    N_TILE,
    PROJECTIONS,
    VOCAB,
    D,
)
from llama_npu import DispatchSession
from llama_oracle import xoshiro_seed
from ml_dtypes import bfloat16

# The decode's time a token at position p, fit to the README's N=16
# measurements: the lm_head, then per layer its weights and its attention
# over p rows. amdxdna times a dispatch out after about 5 s.
LM_HEAD_MS = 4.7
LAYER_MS = 1.16
LAYER_MS_PER_ROW = 0.000200
DISPATCH_BUDGET_MS = 4000.0


def pack_lane_mma(layer_w, order, core):
    """One core's weight lane for kernels/llama_gemv_mma.cc: per projection,
    its row groups of eight rows, each as two `MMA_SLOT` slots, one per half
    of 2048 columns (down: per 2048-column chunk, chunk-major), with weight
    W[n][8kt + i] of a slot's half at kt*64 + i*8 + n, then the eight rows'
    int32 biases (zero) and fp32 scales."""
    parts = []
    for p in order:
        rows, k = PROJECTIONS[p]
        w, sc = layer_w[p]
        rc = rows // N_CORES
        lo = core * rc
        groups = rc // 8
        chunks = k // D
        blk = w[lo : lo + rc].reshape(groups, 8, chunks, 2, D // 16, 8)
        tiles = blk.transpose(2, 0, 3, 4, 5, 1).reshape(chunks, groups, 2, D * 4)
        slots = np.zeros((chunks, groups, 2, MMA_SLOT), np.int8)
        slots[..., : D * 4] = tiles
        slots[..., D * 4 + 32 : D * 4 + 64] = (
            sc[lo : lo + rc].reshape(1, groups, 1, 8).view(np.int8)
        )
        parts.append(slots.reshape(-1))
    return np.concatenate(parts)


def pack_lmw(embed_i8, embed_sc, gamma):
    """The lm_head blob: the final norm's gamma, then the embed table's slots
    (aie2_decode.WLM_SLOT): per 4 rows the rows, 8 zero biases and 8 scales,
    the rows' then 1.

    Args:
        embed_i8: int8 (VOCAB, D), the tied embed table.
        embed_sc: Its per-row scales.
        gamma: The final norm's bf16 gamma.

    Returns:
        int8, LMW_GAMMA_BYTES + WLM_TOTAL.
    """
    slots = np.zeros((VOCAB // N_TILE, WLM_SLOT), np.int8)
    slots[:, : N_TILE * D] = embed_i8.reshape(-1, N_TILE * D)
    scales = np.ones((VOCAB // N_TILE, 8), np.float32)
    scales[:, :N_TILE] = embed_sc.astype(np.float32).reshape(-1, N_TILE)
    slots[:, N_TILE * D + 32 :] = scales.view(np.int8)
    wpacked = slots.reshape(-1)
    assert wpacked.size == WLM_TOTAL, (wpacked.size, WLM_TOTAL)
    gbytes = np.frombuffer(np.asarray(gamma, bfloat16).tobytes(), dtype=np.int8)
    blob = np.zeros(LMW_GAMMA_BYTES + WLM_TOTAL, dtype=np.int8)
    blob[: gbytes.size] = gbytes
    blob[LMW_GAMMA_BYTES:] = wpacked
    return blob


def tokens_of(records, passes):
    """The token ids of a dispatch's first `passes` records."""
    return [
        int(
            np.frombuffer(
                records[t * REC_BYTES + D + 4 : t * REC_BYTES + D + 8].tobytes(),
                np.int32,
            )[0]
        )
        for t in range(passes)
    ]


class DecodeShape:
    """One build of the decode: its CompileTime values and buffer layouts.

    Args:
        n_layers: Decoder layers.
        t: KV cache rows.
        kset: The sampler's resident top-k set size.
        p_max: Passes (tokens) a dispatch can run.
    """

    def __init__(self, n_layers, t, kset=8, p_max=4):
        self.n_layers = n_layers
        self.t = t
        self.kset = kset
        self.p_max = p_max
        self.stage = StageLayout(n_layers)
        self.seed_at = self.stage.slots[0, "o"] + STAGE_RESID
        self.final_at = self.stage.final + STAGE_RESID
        self.kv_words = kv_words(t, n_layers)

    @property
    def compile_kwargs(self):
        """aie2_decode.build's CompileTime values."""
        return dict(P_MAX=self.p_max, N_LAYERS=self.n_layers, T=self.t, KSET=self.kset)

    def params(self, t0, passes, sampler, batch=(), batch_ends_prompt=True):
        """The design's ScratchpadParameters for one dispatch.

        Args:
            t0: The position of the dispatch's first token.
            passes: The passes it runs, 1..p_max, one token each, except
                that a batch is the only pass.
            sampler: The sampler's (temperature, top_k).
            batch: Prompt tokens, up to M_MAX within one KV block, that the
                first pass runs together as its rows; without any, the first
                pass inputs the previous token's sample.
            batch_ends_prompt: Whether the batch's last token is the
                prompt's last: the lm_head and sampler run for it (and for
                every generated token) and give the first generated token.

        Returns:
            The parameter values by name: passes; m0, the first pass's rows;
            t0; its RoPE table row's offset; per pass t < p_max, whether it
            runs (pass{t}) and, for its first token at position p = t0 + t,
            the KV blocks it reads (kvb{t}) and its row's word offset within
            a (layer, half) region (row{t}); the sampler's; per batch row m,
            whether it runs (pg{m}), the token's lm_head weight slot (gat{m})
            and row; per pass, whether the lm_head runs (lm{t}); the rows and
            flags also in `rows`.

        Raises:
            ValueError: The batch is too long, crosses a KV block, or shares
                the dispatch with other passes.
        """
        if len(batch) > M_MAX or (batch and (t0 % KV_TB) + len(batch) > KV_TB):
            raise ValueError(f"a batch of {len(batch)} from {t0} is not one block's")
        if len(batch) > 1 and passes != 1:
            raise ValueError("a batch of several tokens is its dispatch's only pass")
        temperature, top_k = sampler
        params = {
            "passes": passes,
            "m0": max(len(batch), 1),
            "t0": t0,
            "cs_off": t0 * CS_BYTES,
            **sample_param_values(temperature, top_k, self.kset),
        }
        rows = 0
        for m in range(M_MAX):
            tok = batch[m] if m < len(batch) else 0
            pg = int(m < len(batch))
            params[f"gat{m}"] = LMW_GAMMA_BYTES + tok // N_TILE * WLM_SLOT
            params[f"pg{m}"] = pg
            rows |= (tok % N_TILE) << (2 * m) | pg << (8 + m)
        for t in range(self.p_max):
            on = t < passes
            lm = int(on and (t > 0 or not batch or batch_ends_prompt))
            params[f"lm{t}"] = lm
            rows |= lm << (12 + t)
            p = t0 + t
            params[f"pass{t}"] = int(on)
            params[f"kvb{t}"] = p // KV_TB + 1 if on else 0
            params[f"row{t}"] = (
                (p // KV_TB) * KV_BLOCK_WORDS + (p % KV_TB) * KV_ROW_WORDS if on else 0
            )
        params["rows"] = rows
        return params

    def batches(self, t0, n, largest=M_MAX):
        """How n prompt tokens from position t0 split into batches.

        Returns:
            The batches' sizes: up to `largest` (at most M_MAX) each, none
            crossing a KV block.
        """
        out = []
        while n:
            out.append(min(largest, M_MAX, n, KV_TB - t0 % KV_TB))
            t0 += out[-1]
            n -= out[-1]
        return out

    def passes_for(self, t0):
        """The tokens a dispatch from position t0 generates.

        Args:
            t0: The position of its first token.

        Returns:
            As many as p_max allows, the cache has room for and the dispatch
            budget fits, and at least one.
        """
        n, ms = 0, 0.0
        while n < min(self.p_max, self.t - t0):
            ms += LM_HEAD_MS + self.n_layers * (LAYER_MS + LAYER_MS_PER_ROW * (t0 + n))
            if n and ms > DISPATCH_BUDGET_MS:
                break
            n += 1
        return max(n, 1)

    def pack_wblob(self, layers, rope):
        """The weight blob: each layer's eight weight lanes, then the RoPE table.

        Args:
            layers: The llama_oracle.Layers.
            rope: bf16 (t, 2 * HEAD_D), numpy_llama_ref.rope_table.

        Returns:
            int8, wblob_layout(n_layers, t)'s.
        """
        off_lanes, off_rope, total = wblob_layout(self.n_layers, self.t)
        w = np.zeros(total, np.int8)
        for L, layer in enumerate(layers):
            for c in range(N_CORES):
                at = off_lanes + (L * N_CORES + c) * LANE_BYTES
                w[at : at + LANE_BYTES] = pack_lane_mma(layer.weights, LAYER, c)
        w[off_rope : off_rope + self.t * CS_BYTES] = rope.view(np.int8).ravel()
        return w

    def pack_stage(self, layers, final_gamma, seed, rng_seed):
        """The stage buffer at the start of a generation.

        Args:
            layers: The llama_oracle.Layers, for their norms' gammas.
            final_gamma: The final norm's bf16 gamma.
            seed: The first token's fp32 (D,) layer-0 residual, if it is not
                a prompt token.
            rng_seed: The sampler's 32-bit seed for the generation.

        Returns:
            int8, StageLayout's: each norm's gamma after the residual it
            norms, the first token's seed and the sampler's xoshiro state.
        """
        slots = self.stage.slots
        st = np.zeros(self.stage.total, np.int8)
        gammas = [(slots[L, "o"], layer.gamma_in) for L, layer in enumerate(layers)]
        gammas += [(slots[L, "d"], layer.gamma_post) for L, layer in enumerate(layers)]
        gammas.append((self.stage.final, final_gamma))
        for at, g in gammas:
            at += STAGE_RESID + M_MAX * RROW
            st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
        st[self.seed_at : self.seed_at + RROW] = np.asarray(seed, np.float32).view(
            np.int8
        )
        st[self.stage.rng : self.stage.rng + 16] = np.array(
            xoshiro_seed(rng_seed), np.uint32
        ).view(np.int8)
        return st

    def rng_state(self, stage):
        """The sampler's xoshiro state in a stage buffer, as 4 uint32."""
        return [
            np.uint32(w)
            for w in stage[self.stage.rng : self.stage.rng + 16].view(np.uint32)
        ]

    def kv_row_words(self, layer, positions):
        """A layer's cache rows at `positions`, as KV words.

        Args:
            layer: A llama_oracle.Layer.
            positions: Rows of its cache.

        Returns:
            int32 (N_HEADS_KV, len(positions), KV_ROW_WORDS): per head and
            row, [k int8 | v int8 | k scale | v scale].
        """
        rows = np.concatenate(
            [
                layer.kcache[:, positions],
                layer.vcache[:, positions],
                layer.k_slot[:, positions, None].view(np.int8),
                layer.v_slot[:, positions, None].view(np.int8),
            ],
            axis=2,
        )
        return np.ascontiguousarray(rows).view(np.int32)

    def kv_word_index(self, L, positions):
        """Where layer L's rows at `positions` sit in the KV cache.

        Returns:
            int64 (N_HEADS_KV, len(positions), KV_ROW_WORDS) word indices,
            matching kv_row_words.
        """
        p = np.asarray(positions)
        h = np.arange(N_HEADS_KV)[:, None]
        row = (
            ((L * 2 + h // (N_HEADS_KV // 2)) * kv_blocks(self.t) + p // KV_TB)
            * (N_HEADS_KV // 2)
            + h % (N_HEADS_KV // 2)
        ) * KV_TB + p % KV_TB
        return (row * KV_ROW_WORDS)[..., None] + np.arange(KV_ROW_WORDS)

    def pack_kv(self, layers):
        """A KV cache holding the layers' caches.

        Args:
            layers: The llama_oracle.Layers.

        Returns:
            int32 (kv_words,), block-major as the design reads it.
        """
        kv = np.zeros(self.kv_words, np.int32)
        positions = np.arange(self.t)
        for L, layer in enumerate(layers):
            kv[self.kv_word_index(L, positions)] = self.kv_row_words(layer, positions)
        return kv


@dataclass
class Generation:
    """A generation's tokens and times.

    Attributes:
        tokens: The generated token ids.
        ttft_s: Seconds from the request's start to the first generated
            token's readback.
        prefill_dispatch_s: Of those, the prefill dispatches' start-to-wait
            time, summed.
        decode_tok_s: Generated tokens per second after the first, or None
            when the first one stopped the generation.
    """

    tokens: list[int]
    ttft_s: float
    prefill_dispatch_s: float
    decode_tok_s: float | None


class Decoder:
    """The decode on the NPU: one compiled build, its weights loaded.

    Args:
        shape: The build.
        layers: The llama_oracle.Layers.
        lm: (embed_i8, embed_sc, final norm gamma).
        rope: bf16 (shape.t, 2 * HEAD_D), numpy_llama_ref.rope_table.
        elf: A prebuilt full ELF of `shape`'s build, or None for the JIT's.
    """

    def __init__(self, shape, layers, lm, rope, elf=None):
        self.shape = shape
        self.session = DispatchSession(build, elf, **shape.compile_kwargs)
        self.w_t = iron.tensor(shape.pack_wblob(layers, rope), dtype=np.int8)
        self.lmw_t = iron.tensor(pack_lmw(*lm), dtype=np.int8)
        self.kv_t = None
        self.layers = layers
        self.final_gamma = lm[2]

    def dispatch(self, kv_t, stage_t, params):
        """Run one dispatch on a KV cache and stage.

        Args:
            kv_t: The KV cache tensor.
            stage_t: The stage tensor.
            params: shape.params's.

        Returns:
            (its records, its wall time in ms).
        """
        out_t = iron.zeros([self.shape.p_max * REC_BYTES], dtype=np.int8)
        ms = self.session([self.w_t, kv_t, self.lmw_t, out_t, stage_t], **params)
        out_t.to("cpu")
        return out_t.numpy().copy(), ms

    def generate(self, prompt, max_new, sample, stop=(), started=None):
        """Prefill `prompt` into the session's KV cache, then generate.

        The first call allocates the KV cache. The prompt runs in batches
        (shape.batches), the last one's sample being the first generated
        token; then each dispatch runs shape.passes_for's many tokens.

        Args:
            prompt: The prompt's token ids.
            max_new: Tokens to generate at most.
            sample: The sampler's (temperature, top_k, seed), seeded once
                for the generation.
            stop: Token ids that end the generation (kept).
            started: The request's start (time.perf_counter), as early as
                its tokenization; None starts it here.

        Returns:
            The Generation.

        Raises:
            ValueError: The prompt and max_new do not fit the KV cache.
        """
        if not prompt or len(prompt) + max_new - 1 > self.shape.t:
            raise ValueError(
                f"{len(prompt)} prompt tokens and {max_new} more do not fit "
                f"in t={self.shape.t}"
            )
        start = time.perf_counter() if started is None else started
        # A generation reads only the KV rows it wrote, so a stale cache
        # needs no clearing between requests.
        if self.kv_t is None:
            self.kv_t = iron.tensor(
                np.zeros(self.shape.kv_words, np.int32), dtype=np.int32
            )
        temperature, top_k, rng_seed = sample
        sampler = (temperature, top_k)
        stage_t = iron.tensor(
            self.shape.pack_stage(
                self.layers,
                self.final_gamma,
                np.zeros(D, np.float32),
                rng_seed,
            ),
            dtype=np.int8,
        )
        pos = 0
        out = []
        prefill_ms = 0.0
        sizes = self.shape.batches(0, len(prompt))
        for i, n in enumerate(sizes):
            records, ms = self.dispatch(
                self.kv_t,
                stage_t,
                self.shape.params(
                    pos,
                    1,
                    sampler,
                    prompt[pos : pos + n],
                    batch_ends_prompt=i == len(sizes) - 1,
                ),
            )
            prefill_ms += ms
            pos += n
            out = tokens_of(records, 1)[:max_new]
        first_at = time.perf_counter()
        # The last generated token is not input again.
        end = len(prompt) + max_new - 1
        while pos < end and out[-1] not in stop:
            n = min(self.shape.passes_for(pos), end - pos)
            records, _ = self.dispatch(
                self.kv_t, stage_t, self.shape.params(pos, n, sampler)
            )
            for tok in tokens_of(records, n):
                out.append(tok)
                if tok in stop:
                    break
            pos += n
        done = time.perf_counter()
        return Generation(
            out,
            first_at - start,
            prefill_ms / 1e3,
            (len(out) - 1) / (done - first_at) if len(out) > 1 else None,
        )
