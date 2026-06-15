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

import aie.iron as iron
from aie.iron import In, ObjectFifo, Out, Program, Runtime, Worker
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern

N_LINES = 12
LINE_LEN = 8
WINDOW = 3
FIFO_DEPTH = WINDOW + 1  # = 4; one free producer slot at steady state
SEED_LINES = WINDOW
OUT_LEN_PER_CORE = N_LINES * LINE_LEN
OUT_LEN = 2 * OUT_LEN_PER_CORE

LINE_TY = np.ndarray[(LINE_LEN,), np.dtype[np.int8]]


def core_body(of_bridge, of_peer_in, of_peer_out, of_host):
    for _ in range_(sys.maxsize):
        # Seed phase: forward shim-provided lines verbatim to peer.
        for _ in range_(SEED_LINES):
            s = of_bridge.acquire(1)
            o = of_peer_out.acquire(1)
            for b in range_(LINE_LEN):
                o[b] = s[b]
            of_bridge.release(1)
            of_peer_out.release(1)
        # Cyclostatic phase: consume from peer, produce back to peer.
        for _ in range_(N_LINES):
            win = of_peer_in.acquire(WINDOW)
            o_peer = of_peer_out.acquire(1)
            o_host = of_host.acquire(1)
            for b in range_(LINE_LEN):
                s = win[0][b] + win[1][b] + win[2][b]
                o_peer[b] = s
                o_host[b] = s
            of_peer_in.release(1)
            of_peer_out.release(1)
            of_host.release(1)
        of_peer_in.release(WINDOW - 1)


@iron.jit
def cyclostatic_two_core_mutual(seed_a: In, seed_b: In, out_tensor: Out):
    seed_ty = np.ndarray[(SEED_LINES * LINE_LEN,), np.dtype[np.int8]]
    out_ty = np.ndarray[(OUT_LEN,), np.dtype[np.int8]]

    # Seed fifos: shim -> memtile -> core's bridge into peer fifo.
    of_seed_a_l3l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="seed_a_l3l2")
    of_a_in_bridge = of_seed_a_l3l2.cons().forward(name="a_in_bridge", depth=FIFO_DEPTH)
    of_seed_b_l3l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="seed_b_l3l2")
    of_b_in_bridge = of_seed_b_l3l2.cons().forward(name="b_in_bridge", depth=FIFO_DEPTH)

    # Mutual peer fifos: A->B and B->A direct core-to-core.
    of_a_to_b = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="a_to_b")
    of_b_to_a = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="b_to_a")

    # Per-core output back to shim, for verification.
    of_a_out_l1l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="a_out_l1l2")
    of_a_out_l2l3 = of_a_out_l1l2.cons().forward(name="a_out_l2l3", depth=FIFO_DEPTH)
    of_b_out_l1l2 = ObjectFifo(LINE_TY, depth=FIFO_DEPTH, name="b_out_l1l2")
    of_b_out_l2l3 = of_b_out_l1l2.cons().forward(name="b_out_l2l3", depth=FIFO_DEPTH)

    # A consumes from b_to_a, produces into a_to_b. B mirrors.
    worker_a = Worker(
        core_body,
        fn_args=[
            of_a_in_bridge.cons(),
            of_b_to_a.cons(),
            of_a_to_b.prod(),
            of_a_out_l1l2.prod(),
        ],
    )
    worker_b = Worker(
        core_body,
        fn_args=[
            of_b_in_bridge.cons(),
            of_a_to_b.cons(),
            of_b_to_a.prod(),
            of_b_out_l1l2.prod(),
        ],
    )

    a_tap = TensorAccessPattern(
        (OUT_LEN,),
        offset=0,
        sizes=[1, 1, 1, OUT_LEN_PER_CORE],
        strides=[0, 0, 0, 1],
    )
    b_tap = TensorAccessPattern(
        (OUT_LEN,),
        offset=OUT_LEN_PER_CORE,
        sizes=[1, 1, 1, OUT_LEN_PER_CORE],
        strides=[0, 0, 0, 1],
    )

    rt = Runtime()
    with rt.sequence(seed_ty, seed_ty, out_ty) as (s_a, s_b, c_out):
        rt.start(worker_a, worker_b)
        rt.fill(of_seed_a_l3l2.prod(), s_a)
        rt.fill(of_seed_b_l3l2.prod(), s_b)
        rt.drain(of_a_out_l2l3.cons(), c_out, tap=a_tap)
        rt.drain(of_b_out_l2l3.cons(), c_out, tap=b_tap, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def simulate(seed_a, seed_b):
    # Mirror of the device-side execution. Each core's "input queue" is the
    # peer's output queue. Seed phase: A's output queue starts with seed_a,
    # B's with seed_b. Cyclostatic phase: each core consumes its 3-line
    # window from peer's queue, emits sum, appends sum to its own queue.
    a_to_b = list(seed_a.reshape(SEED_LINES, LINE_LEN).astype(np.int32))
    b_to_a = list(seed_b.reshape(SEED_LINES, LINE_LEN).astype(np.int32))

    a_out = np.zeros((N_LINES, LINE_LEN), dtype=np.int32)
    b_out = np.zeros((N_LINES, LINE_LEN), dtype=np.int32)

    for i in range(N_LINES):
        a_win = b_to_a[i : i + WINDOW]
        b_win = a_to_b[i : i + WINDOW]
        a_sum = a_win[0] + a_win[1] + a_win[2]
        b_sum = b_win[0] + b_win[1] + b_win[2]
        a_out[i] = a_sum
        b_out[i] = b_sum
        a_to_b.append(a_sum)
        b_to_a.append(b_sum)

    a_out_i8 = a_out.astype(np.int8).reshape(-1)
    b_out_i8 = b_out.astype(np.int8).reshape(-1)
    return np.concatenate([a_out_i8, b_out_i8])


def main():
    rng = np.random.default_rng(0)
    seed_a = rng.integers(-4, 4, size=(SEED_LINES * LINE_LEN,), dtype=np.int8)
    seed_b = rng.integers(-4, 4, size=(SEED_LINES * LINE_LEN,), dtype=np.int8)
    ref = simulate(seed_a, seed_b)

    in_a = iron.tensor(seed_a, dtype=np.int8, device="npu")
    in_b = iron.tensor(seed_b, dtype=np.int8, device="npu")
    out = iron.zeros(2 * N_LINES * LINE_LEN, dtype=np.int8, device="npu")

    cyclostatic_two_core_mutual(in_a, in_b, out)
    if not np.array_equal(out.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
