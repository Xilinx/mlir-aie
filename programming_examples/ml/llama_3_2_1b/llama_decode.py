#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host side of the decode (aie2_decode.py): its buffers, its per-dispatch
ScratchpadParameters, and a session that prefills and generates."""

from __future__ import annotations

import time

import aie.iron as iron
import numpy as np
from aie2_decode import (
    KV_BLOCK_WORDS,
    KV_ROW_WORDS,
    KV_TB,
    LANE_BYTES,
    LMW_GAMMA_BYTES,
    M_MAX,
    REC_BYTES,
    RQ,
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
from aie2_gemv_array import (
    LAYER,
    MMA_SLOT,
    N_CORES,
    PROJECTIONS,
    sample_param_values,
)
from llama_config import CS_BYTES, HEAD_D, N_HEADS_KV, N_TILE, VOCAB, D
from llama_npu import DispatchSession
from llama_oracle import xoshiro_seed
from ml_dtypes import bfloat16

# The decode's time a token at position p, fit to the README's N=16
# measurements: the lm_head, then per layer its weights and its attention
# over p rows. amdxdna times a dispatch out after about 5 s.
LM_HEAD_MS = 19.0
LAYER_MS = 1.80
LAYER_MS_PER_ROW = 0.000163
DISPATCH_BUDGET_MS = 4000.0


def pack_perchan_slots(w_i8, w_scales, bias_i32, n_tile, prefix_bytes: bytes = b""):
    """Per-tile slots [prefix_bytes | n_tile * K int8 | n_tile int32 bias |
    n_tile fp32 w_scales], the prefix repeated in every slot."""
    N, K = w_i8.shape
    n_tiles = N // n_tile
    pre = len(prefix_bytes)
    slot = pre + n_tile * K + n_tile * 4 + n_tile * 4
    blob = np.zeros(n_tiles * slot, dtype=np.int8)
    pre_arr = np.frombuffer(prefix_bytes, dtype=np.int8)
    for t in range(n_tiles):
        base = t * slot
        off = base
        if pre:
            blob[off : off + pre] = pre_arr
            off += pre
        rows = w_i8[t * n_tile : (t + 1) * n_tile].flatten()
        bs = bias_i32[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        ws = w_scales[t * n_tile : (t + 1) * n_tile].view(np.int8).flatten()
        blob[off : off + rows.size] = rows
        off += rows.size
        blob[off : off + bs.size] = bs
        off += bs.size
        blob[off : off + ws.size] = ws
    return blob


def pack_lane_mma(layer_w, order, core):
    """One core's weight lane for kernels/llama_gemv_mma.cc: per projection,
    its row groups of eight rows, each as two `MMA_SLOT` slots, one per half
    of 2048 columns (down: per 2048-column chunk, chunk-major), with weight
    W[n][8kt + i] of a slot's half at kt*64 + i*8 + n, then the eight rows'
    int32 biases and fp32 scales."""
    parts = []
    for p in order:
        k, rows, _ = PROJECTIONS[p]
        w, bias, sc = layer_w[p]
        rc = rows // N_CORES
        lo = core * rc
        groups = rc // 8
        chunks = k // D
        blk = w[lo : lo + rc].reshape(groups, 8, chunks, 2, D // 16, 8)
        tiles = blk.transpose(2, 0, 3, 4, 5, 1).reshape(chunks, groups, 2, D * 4)
        slots = np.zeros((chunks, groups, 2, MMA_SLOT), np.int8)
        slots[..., : D * 4] = tiles
        slots[..., D * 4 : D * 4 + 32] = (
            bias[lo : lo + rc].reshape(1, groups, 1, 8).view(np.int8)
        )
        slots[..., D * 4 + 32 : D * 4 + 64] = (
            sc[lo : lo + rc].reshape(1, groups, 1, 8).view(np.int8)
        )
        parts.append(slots.reshape(-1))
    return np.concatenate(parts)


def pack_lmw(embed_i8, embed_sc, gamma):
    """The lm_head blob: the final norm's gamma, then the embed table's slots.

    Args:
        embed_i8: int8 (VOCAB, D), the tied embed table.
        embed_sc: Its per-row scales.
        gamma: The final norm's bf16 gamma.

    Returns:
        int8, LMW_GAMMA_BYTES + WLM_TOTAL.
    """
    bias = np.zeros(VOCAB, np.int32)
    wpacked = pack_perchan_slots(
        embed_i8, embed_sc.astype(np.float32), bias, N_TILE, prefix_bytes=b"\x00" * 64
    )
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
            layers: The layers' weights (llama_oracle's layer dicts).
            rope: bf16 (t, 2 * HEAD_D), numpy_llama_ref.rope_table.

        Returns:
            int8, wblob_layout(n_layers, t)'s.
        """
        off_lanes, off_rope, total = wblob_layout(self.n_layers, self.t)
        w = np.zeros(total, np.int8)
        for L, layer in enumerate(layers):
            weights = {
                p: (
                    layer[f"w{p}_i8"],
                    layer.get(f"b{p}", np.zeros(layer[f"w{p}_i8"].shape[0], np.int32)),
                    layer[f"w{p}_sc"],
                )
                for p in LAYER
            }
            for c in range(N_CORES):
                at = off_lanes + (L * N_CORES + c) * LANE_BYTES
                w[at : at + LANE_BYTES] = pack_lane_mma(weights, LAYER, c)
        w[off_rope : off_rope + self.t * CS_BYTES] = rope.view(np.int8).ravel()
        return w

    def pack_stage(self, layers, final_gamma, seed, rng_seed):
        """The stage buffer at the start of a generation.

        Args:
            layers: The layers' norms (gamma_in, gamma_post).
            final_gamma: The final norm's bf16 gamma.
            seed: The first token's layer-0 residual (int8[D], f32 scale,
                fp32[D]), if it is not a prompt token.
            rng_seed: The sampler's 32-bit seed for the generation.

        Returns:
            int8, StageLayout's: each norm's gamma after the residual it
            norms, the first token's seed and the sampler's xoshiro state.
        """
        slots = self.stage.slots
        st = np.zeros(self.stage.total, np.int8)
        gammas = [(slots[L, "o"], layer["gamma_in"]) for L, layer in enumerate(layers)]
        gammas += [
            (slots[L, "d"], layer["gamma_post"]) for L, layer in enumerate(layers)
        ]
        gammas.append((self.stage.final, final_gamma))
        for at, g in gammas:
            at += STAGE_RESID + M_MAX * RROW
            st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
        x_i8, res_scale, x_fp = seed
        st[self.seed_at : self.seed_at + D] = x_i8
        st[self.seed_at + D : self.seed_at + D + 4] = np.frombuffer(
            np.float32(res_scale).tobytes(), np.int8
        )
        fp_at = self.seed_at + RQ
        st[fp_at : fp_at + 4 * D] = np.asarray(x_fp, np.float32).view(np.int8)
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

    def pack_kv(self, layers):
        """A KV cache holding the layers' caches.

        Args:
            layers: Per layer, t-row kcaches/vcaches and k/v slot scales.

        Returns:
            int32 (kv_words,), block-major as the design reads it.
        """
        nb = kv_blocks(self.t)
        rows = np.zeros(
            (self.n_layers, N_HEADS_KV, nb * KV_TB, KV_ROW_WORDS * 4), np.int8
        )
        t = self.t
        for L, layer in enumerate(layers):
            for h in range(N_HEADS_KV):
                rows[L, h, :t, :HEAD_D] = layer["kcaches"][h].reshape(t, HEAD_D)
                rows[L, h, :t, HEAD_D : 2 * HEAD_D] = layer["vcaches"][h].reshape(
                    t, HEAD_D
                )
                for i, name in enumerate(("k_scales_slot", "v_scales_slot")):
                    at = 2 * HEAD_D + 4 * i
                    rows[L, h, :t, at : at + 4] = (
                        layer[name][h].astype(np.float32).view(np.int8).reshape(t, 4)
                    )
        blocks = rows.reshape(self.n_layers, 2, N_HEADS_KV // 2, nb, KV_TB, -1)
        packed = np.ascontiguousarray(blocks.transpose(0, 1, 3, 2, 4, 5))
        assert packed.size == self.kv_words * 4
        return packed.view(np.int32).ravel()


class Decoder:
    """The decode on the NPU: one compiled build, its weights loaded.

    Args:
        shape: The build.
        layers: The layers' weights and norms (llama_oracle's layer dicts).
        lm: (embed_i8, embed_sc, final norm gamma).
        rope: bf16 (shape.t, 2 * HEAD_D), numpy_llama_ref.rope_table.
        elf: A prebuilt full ELF of `shape`'s build, or None for the JIT's.
    """

    def __init__(self, shape, layers, lm, rope, elf=None):
        self.shape = shape
        self.session = DispatchSession(build, elf, **shape.compile_kwargs)
        self.w_t = iron.tensor(shape.pack_wblob(layers, rope), dtype=np.int8)
        self.lmw_t = iron.tensor(pack_lmw(*lm), dtype=np.int8)
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

    def generate(self, prompt, max_new, sample, stop=()):
        """Prefill `prompt` from an empty KV cache, then generate.

        The prompt runs in batches (shape.batches), the last one's sample
        being the first generated token; then each dispatch runs
        shape.passes_for's many tokens.

        Args:
            prompt: The prompt's token ids.
            max_new: Tokens to generate at most.
            sample: The sampler's (temperature, top_k, seed), seeded once
                for the generation.
            stop: Token ids that end the generation (kept).

        Returns:
            (the generated token ids, seconds to the first one, generated
            tokens per second after it).

        Raises:
            ValueError: The prompt and max_new do not fit the KV cache.
        """
        if not prompt or len(prompt) + max_new - 1 > self.shape.t:
            raise ValueError(
                f"{len(prompt)} prompt tokens and {max_new} more do not fit "
                f"in t={self.shape.t}"
            )
        temperature, top_k, rng_seed = sample
        sampler = (temperature, top_k)
        kv_t = iron.tensor(np.zeros(self.shape.kv_words, np.int32), dtype=np.int32)
        stage_t = iron.tensor(
            self.shape.pack_stage(
                self.layers,
                self.final_gamma,
                (np.zeros(D, np.int8), 0.0, np.zeros(D, np.float32)),
                rng_seed,
            ),
            dtype=np.int8,
        )
        start = time.perf_counter()
        pos = 0
        out = []
        sizes = self.shape.batches(0, len(prompt))
        for i, n in enumerate(sizes):
            records, _ = self.dispatch(
                kv_t,
                stage_t,
                self.shape.params(
                    pos,
                    1,
                    sampler,
                    prompt[pos : pos + n],
                    batch_ends_prompt=i == len(sizes) - 1,
                ),
            )
            pos += n
            out = tokens_of(records, 1)[:max_new]
        first_at = time.perf_counter()
        # The last generated token is not input again.
        end = len(prompt) + max_new - 1
        while pos < end and out[-1] not in stop:
            n = min(self.shape.passes_for(pos), end - pos)
            records, _ = self.dispatch(kv_t, stage_t, self.shape.params(pos, n, sampler))
            for tok in tokens_of(records, n):
                out.append(tok)
                if tok in stop:
                    break
            pos += n
        done = time.perf_counter()
        tps = (len(out) - 1) / (done - first_at) if len(out) > 1 else 0.0
        return out, first_at - start, tps
