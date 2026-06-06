# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# (c) Copyright 2026 AMD Inc.
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Mutual ping-pong: both cores cyclostatic on a fifo whose producer is the
# other core. Each core's body:
#   for _ in range(N):
#     win = acquire(peer_in, 3)      # cyclostatic window from peer
#     out = acquire(peer_out, prod)  # producer slot to send back to peer
#     out = win[0] + win[1] + win[2]
#     release(peer_in, 1)            # slide window
#     release(peer_out, 1)           # commit produced item
#   release(peer_in, 2)              # drain window
#
# This is the actually-load-bearing 2-core test: it exercises cross-core
# cyclostatic sync AND mutual production. A naive hoist of acquire(carry)
# before the loop on both cores would block each core's pre-body wait before
# either does production work, creating an inter-core deadlock. Peel keeps
# iter-0's acquire/produce/release interleaving intact so each core's first
# produce frees a peer-side slot before the peer's next acquire waits.
#
# Fifo depth: window_size + 1. With 3-line window and depth 4, at steady
# state the consumer holds 3 and the producer side has 1 free slot — enough
# for the peer to refill without blocking on a circular dependency.
#
# Shim seeds both peer fifos with `window_size` initial lines so iter 0 has
# enough data to start.
import sys
import numpy as np

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.scf import _for as range_
from aie.extras.context import mlir_mod_ctx

N_LINES = 12
LINE_LEN = 8
WINDOW = 3
FIFO_DEPTH = WINDOW + 1  # = 4; one free producer slot at steady state
SEED_LINES = WINDOW
OUT_LEN_PER_CORE = N_LINES * LINE_LEN
OUT_LEN = 2 * OUT_LEN_PER_CORE


def build(dev):
    with mlir_mod_ctx() as ctx:

        @device(dev)
        def device_body():
            shim = tile(0, 0)
            memtile = tile(0, 1)
            core_a = tile(0, 2)
            core_b = tile(0, 3)

            line_ty = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]
            seed_ty = np.ndarray[(SEED_LINES * LINE_LEN,), np.dtype[np.int8]]
            out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

            # Seed fifos: shim -> memtile -> core's bridge into peer fifo.
            seed_a_l3l2 = object_fifo("seed_a_l3l2", shim, memtile, FIFO_DEPTH, line_ty)
            seed_b_l3l2 = object_fifo("seed_b_l3l2", shim, memtile, FIFO_DEPTH, line_ty)
            a_in_bridge = object_fifo(
                "a_in_bridge", memtile, core_a, FIFO_DEPTH, line_ty
            )
            object_fifo_link(seed_a_l3l2, a_in_bridge)
            b_in_bridge = object_fifo(
                "b_in_bridge", memtile, core_b, FIFO_DEPTH, line_ty
            )
            object_fifo_link(seed_b_l3l2, b_in_bridge)

            # Mutual peer fifos: A->B and B->A direct core-to-core.
            a_to_b = object_fifo("a_to_b", core_a, core_b, FIFO_DEPTH, line_ty)
            b_to_a = object_fifo("b_to_a", core_b, core_a, FIFO_DEPTH, line_ty)

            # Per-core output back to shim, for verification.
            a_out_l1l2 = object_fifo("a_out_l1l2", core_a, memtile, FIFO_DEPTH, line_ty)
            a_out_l2l3 = object_fifo("a_out_l2l3", memtile, shim, FIFO_DEPTH, line_ty)
            object_fifo_link(a_out_l1l2, a_out_l2l3)
            b_out_l1l2 = object_fifo("b_out_l1l2", core_b, memtile, FIFO_DEPTH, line_ty)
            b_out_l2l3 = object_fifo("b_out_l2l3", memtile, shim, FIFO_DEPTH, line_ty)
            object_fifo_link(b_out_l1l2, b_out_l2l3)

            @core(core_a)
            def core_a_body():
                for _ in range_(sys.maxsize):
                    # Seed phase: forward shim-provided lines verbatim to peer.
                    for _ in range_(SEED_LINES):
                        s = a_in_bridge.acquire(ObjectFifoPort.Consume, 1)
                        o = a_to_b.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            o[b] = s[b]
                        a_in_bridge.release(ObjectFifoPort.Consume, 1)
                        a_to_b.release(ObjectFifoPort.Produce, 1)
                    # Cyclostatic phase: consume from peer, produce back to peer.
                    for _ in range_(N_LINES):
                        win = b_to_a.acquire(ObjectFifoPort.Consume, WINDOW)
                        o_peer = a_to_b.acquire(ObjectFifoPort.Produce, 1)
                        o_host = a_out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            s = win[0][b] + win[1][b] + win[2][b]
                            o_peer[b] = s
                            o_host[b] = s
                        b_to_a.release(ObjectFifoPort.Consume, 1)
                        a_to_b.release(ObjectFifoPort.Produce, 1)
                        a_out_l1l2.release(ObjectFifoPort.Produce, 1)
                    b_to_a.release(ObjectFifoPort.Consume, WINDOW - 1)

            @core(core_b)
            def core_b_body():
                for _ in range_(sys.maxsize):
                    for _ in range_(SEED_LINES):
                        s = b_in_bridge.acquire(ObjectFifoPort.Consume, 1)
                        o = b_to_a.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            o[b] = s[b]
                        b_in_bridge.release(ObjectFifoPort.Consume, 1)
                        b_to_a.release(ObjectFifoPort.Produce, 1)
                    for _ in range_(N_LINES):
                        win = a_to_b.acquire(ObjectFifoPort.Consume, WINDOW)
                        o_peer = b_to_a.acquire(ObjectFifoPort.Produce, 1)
                        o_host = b_out_l1l2.acquire(ObjectFifoPort.Produce, 1)
                        for b in range_(LINE_LEN):
                            s = win[0][b] + win[1][b] + win[2][b]
                            o_peer[b] = s
                            o_host[b] = s
                        a_to_b.release(ObjectFifoPort.Consume, 1)
                        b_to_a.release(ObjectFifoPort.Produce, 1)
                        b_out_l1l2.release(ObjectFifoPort.Produce, 1)
                    a_to_b.release(ObjectFifoPort.Consume, WINDOW - 1)

            @runtime_sequence(seed_ty, seed_ty, out_ty)
            def sequence(SeedA, SeedB, Out):
                sa = shim_dma_single_bd_task(
                    seed_a_l3l2, SeedA, offset=0, sizes=[1, 1, 1, SEED_LINES * LINE_LEN]
                )
                sb = shim_dma_single_bd_task(
                    seed_b_l3l2, SeedB, offset=0, sizes=[1, 1, 1, SEED_LINES * LINE_LEN]
                )
                oa = shim_dma_single_bd_task(
                    a_out_l2l3, Out, offset=0, sizes=[1, 1, 1, OUT_LEN_PER_CORE]
                )
                ob = shim_dma_single_bd_task(
                    b_out_l2l3,
                    Out,
                    offset=OUT_LEN_PER_CORE,
                    sizes=[1, 1, 1, OUT_LEN_PER_CORE],
                    issue_token=True,
                )
                dma_start_task(sa, sb, oa, ob)
                dma_await_task(ob)
                dma_free_task(sa, sb, oa)

        print(ctx.module)


if __name__ == "__main__":
    dev_str = sys.argv[1] if len(sys.argv) > 1 else "npu2"
    dev = {"npu1": AIEDevice.npu1_1col, "npu2": AIEDevice.npu2}[dev_str]
    build(dev)
