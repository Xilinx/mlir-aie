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
    GAMMA_AT,
    KV_BLOCK_WORDS,
    KV_ROW_WORDS,
    KV_TB,
    LANE_BYTES,
    LMW_GAMMA_BYTES,
    REC_BYTES,
    STAGE_RESID,
    VOCAB,
    WLM_SLOT,
    WLM_TOTAL,
    build,
    kv_blocks,
    kv_words,
    stage_layout,
    wblob_layout,
)
from aie2_gemv_array import LAYER, N_CORES, sample_param_values
from aie2_layer_mh import CS_BYTES, HEAD_D, N_HEADS_KV, N_TILE, D
from llama_npu import DispatchSession
from ml_dtypes import bfloat16
from test_gemv_array import pack_lane_mma
from test_layer_mh import pack_perchan_slots

# The decode's time a token at position p, fit to the README's N=16
# measurements: the lm_head, then per layer its weights and its attention
# over p rows. amdxdna times a dispatch out after about 5 s.
LM_HEAD_MS = 19.0
LAYER_MS = 1.85
LAYER_MS_PER_ROW = 0.0006
DISPATCH_BUDGET_MS = 4000.0


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


def embed_seed_ref(row_i8, embed_sc):
    """A token's embed seed, as llama_topk_sample.cc's embed_seed makes it.

    Args:
        row_i8: The token's int8 embed row.
        embed_sc: Its scale.

    Returns:
        (int8[D], f32 scale).
    """
    amax = int(np.max(np.abs(row_i8.astype(np.int32))))
    if amax < 1:
        amax = 1
    inv = (np.float32(127.0) / np.float32(amax)).astype(np.float32)
    xin = np.empty(D, np.int8)
    for i in range(D):
        v = np.float32(np.float32(row_i8[i]) * inv)
        r = np.floor(v + 0.5) if v >= 0 else np.ceil(v - 0.5)
        xin[i] = np.int8(np.clip(r, -128, 127))
    scale = np.float32(
        np.float32(amax) * np.float32(embed_sc) * np.float32(1.0 / 127.0)
    )
    return xin, scale


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
        p_max: Tokens a dispatch can generate.
        attn_cores: Attention cores a KV head.
    """

    def __init__(self, n_layers, t, kset=8, p_max=4, attn_cores=2):
        self.n_layers = n_layers
        self.t = t
        self.kset = kset
        self.p_max = p_max
        self.attn_cores = attn_cores
        slots, _, final, discard, self.stage_bytes = stage_layout(n_layers)
        self.seed_at = slots[0, "o"] + STAGE_RESID
        self.final_at = final + STAGE_RESID
        self.discard_at = discard
        self.kv_words = kv_words(t, n_layers)

    @property
    def compile_kwargs(self):
        """aie2_decode.build's CompileTime values."""
        return dict(
            P_MAX=self.p_max,
            N_LAYERS=self.n_layers,
            T=self.t,
            KSET=self.kset,
            ATTN_CORES=self.attn_cores,
        )

    def params(self, t0, passes, sample, prompt=()):
        """The design's ScratchpadParameters for one dispatch.

        Args:
            t0: The position of the dispatch's first token.
            passes: The tokens it generates, 1..p_max.
            sample: The sampler's (temperature, top_k, seed).
            prompt: Per token slot, the prompt token it inputs, or None (or
                past the end) to input the previous token's sample.

        Returns:
            The parameter values by name: passes; t0; its RoPE table row's
            offset; per token slot t < p_max, whether it runs (pass{t}) and,
            for a token at position p = t0 + t, the KV blocks it reads
            (kvb{t}) and its row's word offset within a (layer, half) region
            (row{t}); the sampler's; per token slot, the lm_head weight slot
            (gat{t}) and its row (in `rows`) of the slot's prompt token, and
            where the seed made of it goes (sd{t}): the stage's seed slot,
            or for a slot without a prompt token, its discard slot.
        """
        temperature, top_k, seed = sample
        params = {
            "passes": passes,
            "t0": t0,
            "cs_off": t0 * CS_BYTES,
            **sample_param_values(temperature, top_k, seed, self.kset),
        }
        rows = 0
        for t in range(self.p_max):
            tok = prompt[t] if t < len(prompt) else None
            params[f"gat{t}"] = LMW_GAMMA_BYTES + (tok or 0) // N_TILE * WLM_SLOT
            params[f"sd{t}"] = self.discard_at if tok is None else self.seed_at
            rows |= ((tok or 0) % N_TILE) << (2 * t)
            p = t0 + t
            on = t < passes
            params[f"pass{t}"] = int(on)
            params[f"kvb{t}"] = p // KV_TB + 1 if on else 0
            params[f"row{t}"] = (
                (p // KV_TB) * KV_BLOCK_WORDS + (p % KV_TB) * KV_ROW_WORDS if on else 0
            )
        params["rows"] = rows
        return params

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
            layers: The layers' weights (numpy_layer_mh's dicts).
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

    def pack_stage(self, layers, final_gamma, x_i8, res_scale):
        """The stage buffer at the start of a generation.

        Args:
            layers: The layers' norms (gamma_in, gamma_post).
            final_gamma: The final norm's bf16 gamma.
            x_i8: The first token's seed, int8[D], if it is not a prompt token.
            res_scale: Its scale.

        Returns:
            int8, stage_layout(n_layers)'s: each norm's gamma after the
            residual it norms, and the first token's seed.
        """
        slots, _, final, _, total = stage_layout(self.n_layers)
        st = np.zeros(total, np.int8)
        gammas = [(slots[L, "o"], layer["gamma_in"]) for L, layer in enumerate(layers)]
        gammas += [
            (slots[L, "d"], layer["gamma_post"]) for L, layer in enumerate(layers)
        ]
        gammas.append((final, final_gamma))
        for at, g in gammas:
            at += STAGE_RESID + GAMMA_AT
            st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
        st[self.seed_at : self.seed_at + D] = x_i8
        st[self.seed_at + D : self.seed_at + D + 4] = np.frombuffer(
            np.float32(res_scale).tobytes(), np.int8
        )
        return st

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
        layers: The layers' weights and norms (numpy_layer_mh's dicts).
        lm: (embed_i8, embed_sc, final norm gamma).
        rope: bf16 (shape.t, 2 * HEAD_D), numpy_llama_ref.rope_table.
        elf: A prebuilt full ELF of `shape`'s build, or None for the JIT's.
    """

    def __init__(self, shape, layers, lm, rope, elf=None):
        self.shape = shape
        self.session = DispatchSession(build, elf, **shape.compile_kwargs)
        self.w_t = iron.tensor(shape.pack_wblob(layers, rope), dtype=np.int8)
        self.lmw_t = iron.tensor(pack_lmw(*lm), dtype=np.int8)
        self._stage0 = shape.pack_stage(layers, lm[2], np.zeros(D, np.int8), 0.0)

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

        Each dispatch runs shape.passes_for's many tokens, the prompt's last
        token ending one; that token's sample is the first generated token.

        Args:
            prompt: The prompt's token ids.
            max_new: Tokens to generate at most.
            sample: The sampler's (temperature, top_k, seed).
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
        kv_t = iron.tensor(np.zeros(self.shape.kv_words, np.int32), dtype=np.int32)
        stage_t = iron.tensor(self._stage0, dtype=np.int8)
        out = []
        pos = 0
        start = time.perf_counter()
        first_at = None
        # The last prompt token's sample is the first generated token, and
        # the last generated token is not input again.
        end = len(prompt) + max_new - 1
        while pos < end and not (out and out[-1] in stop):
            n = min(self.shape.passes_for(pos), end - pos)
            if pos < len(prompt):
                n = min(n, len(prompt) - pos)
            records, _ = self.dispatch(
                kv_t, stage_t, self.shape.params(pos, n, sample, prompt[pos : pos + n])
            )
            for t, tok in enumerate(tokens_of(records, n)):
                if pos + t >= len(prompt) - 1 and len(out) < max_new:
                    out.append(tok)
                    if first_at is None:
                        first_at = time.perf_counter()
                    if tok in stop:
                        break
            pos += n
        done = time.perf_counter()
        assert first_at is not None
        tps = (len(out) - 1) / (done - first_at) if len(out) > 1 else 0.0
        return out, first_at - start, tps
