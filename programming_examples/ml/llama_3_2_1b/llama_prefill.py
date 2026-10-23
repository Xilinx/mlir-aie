#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host side of the prefill (aie2_prefill_layer.py): a prompt's
positions, 64 a pass, into a Decoder's KV cache.

Three hardware contexts share the decode's buffers in one process: the
gather (build_gather) copies the prompt's lm_head slots into `prompt`, 31
a dispatch; the prefill (build_prefill) runs every layer for a pass's 64
rows; the decode then runs the rest of the prompt, at least its last token,
whose sample is the first generated token. Every gather runs before the
first pass, so a request switches contexts twice: gather to prefill, and
prefill to decode.
"""

import aie.iron as iron
import numpy as np
from aie2_decode import KV_BLOCK_WORDS, LMW_GAMMA_BYTES, M_MAX, WLM_SLOT
from aie2_prefill import pack_trailers
from aie2_prefill_layer import (
    GATHER_ROWS,
    SLOT_WORDS,
    PrefillStage,
    build_gather,
    build_prefill,
    ids_words,
    prompt_words,
)
from llama_config import CS_BYTES, KV_TB, N_TILE, D
from llama_decode import LAYER_MS_PER_ROW
from llama_npu import DispatchSession
from ml_dtypes import bfloat16

# A pass's time per layer at block j, PASS_LAYER_MS + PASS_BLOCK_MS * (j + 1),
# and a decode dispatch of M_MAX prompt rows near position 0, measured on
# Strix (npu2); the decode's attention adds llama_decode's LAYER_MS_PER_ROW.
PASS_LAYER_MS = 10.3
PASS_BLOCK_MS = 0.75
DECODE_BATCH_MS = 45.0


class Prefiller:
    """The prefill on the NPU, beside a Decoder of the same build.

    Args:
        decoder: The llama_decode.Decoder whose weights, lm_head slots and
            KV cache the prefill shares.
    """

    def __init__(self, decoder):
        shape = decoder.shape
        self.n_layers = shape.n_layers
        self.t = shape.t
        self.gather = DispatchSession(build_gather, None, T=shape.t)
        self.session = DispatchSession(
            build_prefill,
            None,
            N_LAYERS=shape.n_layers,
            LAYER=0,
            W_LAYERS=shape.n_layers,
            T=shape.t,
        )
        self.table_t = iron.tensor(
            pack_trailers(decoder.w_t.data, shape.n_layers).view(np.int32).ravel(),
            dtype=np.int32,
        )
        stage = PrefillStage(shape.n_layers)
        st = np.zeros(stage.total, np.int8)
        for L, layer in enumerate(decoder.layers):
            for post, g in enumerate((layer.gamma_in, layer.gamma_post)):
                at = stage.gamma(L, post) * 4
                st[at : at + 2 * D] = np.asarray(g, bfloat16).view(np.int8)
        self.stage_t = iron.tensor(st.view(np.int32), dtype=np.int32)
        self.prompt_t = iron.zeros([prompt_words(shape.t)], dtype=np.int32)
        self.ids_t = iron.zeros([ids_words(shape.t)], dtype=np.int32)
        self.dispatch_ms = []

    def plan(self, n):
        """How an n-token prompt splits between the prefill and the decode.

        The decode runs at least the last token. A partial last pass runs
        padded to 64 rows when that beats the decode's batches of M_MAX for
        its rows.

        Args:
            n: The prompt's tokens.

        Returns:
            (prefill passes, the first position the decode runs).
        """
        rows = n - 1
        full, tail = divmod(rows, KV_TB)
        if tail and (full + 1) * KV_TB <= self.t:
            pass_ms = self.n_layers * (PASS_LAYER_MS + PASS_BLOCK_MS * (full + 1))
            batch_ms = DECODE_BATCH_MS + (
                self.n_layers * M_MAX * LAYER_MS_PER_ROW * full * KV_TB
            )
            if pass_ms < (-(-(tail + 1) // M_MAX) - 1) * batch_ms:
                return full + 1, rows
        return full, full * KV_TB

    def run(self, decoder, prompt):
        """Prefill the prompt's positions plan() gives the prefill.

        Args:
            decoder: The Decoder, its KV cache allocated.
            prompt: The prompt's token ids.

        Returns:
            (the first position the decode runs, the dispatches' wall time in
            ms); dispatch_ms holds each dispatch's, the gathers' first.
        """
        passes, pos = self.plan(len(prompt))
        self.dispatch_ms = []
        if not passes:
            return pos, 0.0
        rows = passes * KV_TB
        toks = (list(prompt[:-1]) + [prompt[-1]] * rows)[:rows]
        for at in range(0, rows, GATHER_ROWS):
            group = toks[at : at + GATHER_ROWS]
            group += [group[-1]] * (GATHER_ROWS - len(group))
            self.dispatch_ms.append(
                self.gather(
                    [decoder.lmw_t, self.prompt_t],
                    dst=at * SLOT_WORDS,
                    **{
                        f"g{i}": (LMW_GAMMA_BYTES + tok // N_TILE * WLM_SLOT) // 4
                        for i, tok in enumerate(group)
                    },
                )
            )
        with self.ids_t.mutate() as ids:
            ids[:rows] = toks
        tensors = [
            decoder.w_t,
            self.table_t,
            decoder.kv_t,
            self.stage_t,
            self.prompt_t,
            self.ids_t,
        ]
        for j in range(passes):
            self.dispatch_ms.append(
                self.session(
                    tensors,
                    nb=j + 1,
                    cs_off=j * KV_TB * CS_BYTES // 4,
                    kv_row=j * KV_BLOCK_WORDS,
                    prompt_off=j * KV_TB * SLOT_WORDS,
                    ids_off=j * KV_TB,
                )
            )
        return pos, sum(self.dispatch_ms)
