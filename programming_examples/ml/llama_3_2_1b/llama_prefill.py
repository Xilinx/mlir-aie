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
from aie2_decode import M_MAX
from aie2_prefill import pack_trailers
from aie2_prefill_layer import (
    CONFIGURE,
    ROW_STEP,
    PrefillStage,
    ids_words,
    pass_commands,
)
from llama_config import KV_TB, D
from llama_npu import QUEUE_DEPTH
from ml_dtypes import bfloat16

# The decode's batch takes the prompt's last ROW_STEP tokens at most.
assert ROW_STEP <= M_MAX


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
        """How an n-token prompt splits between the prefill and the decode:
        the prefill takes the most rows short of the last token that come in
        ROW_STEPs, KV_TB a pass, the last pass the rest; the decode the
        prompt's last 1..ROW_STEP tokens, one batch.

        Args:
            n: The prompt's tokens.

        Returns:
            (prefill passes, the first position the decode runs).
        """
        rows = (n - 1) // ROW_STEP * ROW_STEP
        return -(-rows // KV_TB), rows

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
        with self.ids_t.mutate() as ids:
            ids[:pos] = prompt[:pos]
        tensors = [
            decoder.w_t,
            self.table_t,
            decoder.kv_t,
            self.stage_t,
            decoder.lmw_t,
            self.ids_t,
        ]
        # The array holds the decode's configuration until this loads the
        # prefill's.
        queued = [(self.session.submit(tensors, CONFIGURE), False)]
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
            tokens = prompt[j * KV_TB : min(pos, (j + 1) * KV_TB)]
            commands = pass_commands(tokens, j, self.n_layers, j * KV_TB)
            for n, (command, params) in enumerate(commands):
                while len(queued) == QUEUE_DEPTH:
                    wait_oldest()
                dispatch = self.session.submit(tensors, command, **params)
                queued.append((dispatch, n == len(commands) - 1))
        while queued:
            wait_oldest()
        return pos, sum(self.dispatch_ms)
