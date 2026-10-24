#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The host side of the prefill (aie2_prefill_layer.py): a prompt's
positions, 64 a pass, into a Decoder's KV cache.

The prefill runs on the Decoder's hardware context, its commands in the
same full ELF (aie2_llama.build_llama), on the decode's buffers: a pass's
64 rows as its embed commands, which read the rows' tokens' lm_head slots
from the decode's `lmw`, then a command a layer, all queued; the decode then
runs the rest of the prompt, at least its last token, whose sample is the
first generated token. A request's first prefill command and its first
decode command each load their device's PDI.
"""

import time

import aie.iron as iron
import numpy as np
from aie2_decode import KV_BLOCK_WORDS, M_MAX
from aie2_prefill import pack_trailers
from aie2_prefill_layer import PREFILL, PrefillStage, embed_commands, ids_words
from llama_config import CS_BYTES, KV_TB, D
from llama_decode import LAYER_MS_PER_ROW
from llama_npu import QUEUE_DEPTH
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
        self.session = decoder.session
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
            ms); dispatch_ms holds each pass's, from the end of the one
            before.
        """
        passes, pos = self.plan(len(prompt))
        self.dispatch_ms = []
        if not passes:
            return pos, 0.0
        rows = passes * KV_TB
        toks = (list(prompt[:-1]) + [prompt[-1]] * rows)[:rows]
        with self.ids_t.mutate() as ids:
            ids[:rows] = toks
        tensors = [
            decoder.w_t,
            self.table_t,
            decoder.kv_t,
            self.stage_t,
            decoder.lmw_t,
            self.ids_t,
        ]
        queued = []
        done = time.perf_counter()

        def wait_oldest():
            nonlocal done
            dispatch, last = queued.pop(0)
            dispatch.wait()
            if last:
                now = time.perf_counter()
                self.dispatch_ms.append((now - done) * 1e3)
                done = now

        for j in range(passes):
            commands = embed_commands(toks[j * KV_TB : (j + 1) * KV_TB], j * KV_TB)
            layer = dict(
                nb=j + 1,
                rope_off=j * KV_TB * CS_BYTES // 4,
                kv_row=j * KV_BLOCK_WORDS,
            )
            commands += [(f"{PREFILL}:layer{i}", layer) for i in range(self.n_layers)]
            for n, (command, params) in enumerate(commands):
                while len(queued) == QUEUE_DEPTH:
                    wait_oldest()
                dispatch = self.session.submit(tensors, command, **params)
                queued.append((dispatch, n == len(commands) - 1))
        while queued:
            wait_oldest()
        return pos, sum(self.dispatch_ms)
