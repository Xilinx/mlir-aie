# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS
#
# Each sender core adds its own offset to a line from the host and sends it,
# out of order over packets, into its segment of a line the receiving core
# joins on its own tile; every sender shares the receiver's one S2MM channel.
# Each fifo holds a whole run of lines, so no packet waits on a full buffer
# and stalls the channel while the receiver waits on another sender's.
import argparse
import sys

import aie.iron as iron
import numpy as np
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    In,
    ObjectFifo,
    Out,
    Packet,
    Port,
    Program,
    Runtime,
    Transport,
    Worker,
)
from aie.iron.dataflow.objectfifo import ObjectFifoLink

LINE = 16
LINES = 4

parser = argparse.ArgumentParser()
parser.add_argument("--senders", type=int, choices=[2, 3], required=True)
args = parser.parse_args()
SENDERS = args.senders

LINE_TY = np.ndarray[(LINE,), np.dtype[np.int32]]
JOINED_TY = np.ndarray[(SENDERS * LINE,), np.dtype[np.int32]]
IN_TY = np.ndarray[(SENDERS * LINES * LINE,), np.dtype[np.int32]]
OUT_TY = np.ndarray[(LINES * SENDERS * LINE,), np.dtype[np.int32]]


def sender(k):
    def body(of_in, of_mid):
        for _ in range_(LINES):
            src = of_in.acquire(1)
            dst = of_mid.acquire(1)
            for i in range_(LINE):
                dst[i] = src[i] + (k + 1) * 1000
            of_in.release(1)
            of_mid.release(1)

    return body


def receiver(of_joined, of_out):
    for _ in range_(LINES):
        src = of_joined.acquire(1)
        dst = of_out.acquire(1)
        for i in range_(SENDERS * LINE):
            dst[i] = src[i] * 2
        of_joined.release(1)
        of_out.release(1)


@iron.jit
def out_of_order_join(src: In, dst: Out):
    ins = [ObjectFifo(LINE_TY, depth=2, name=f"in{k}") for k in range(SENDERS)]
    mids = [
        ObjectFifo(
            LINE_TY,
            depth=LINES,
            name=f"mid{k}",
            transport=Transport.dma(packet=Packet(), out_of_order=True),
        )
        for k in range(SENDERS)
    ]
    joined = ObjectFifo(JOINED_TY, depth=LINES, name="joined")
    out = ObjectFifo(JOINED_TY, depth=2, name="out")
    workers = [
        Worker(sender(k), fn_args=[ins[k].cons(), mids[k].prod()])
        for k in range(SENDERS)
    ]
    reader = Worker(receiver, fn_args=[joined.cons(), out.prod()])
    workers.append(reader)
    # The joined buffer is the receiver's own, read in place.
    joined.delegate_tile = reader.tile
    # Out-of-order packets share a channel only where they pin it.
    ObjectFifoLink(
        [m.cons(port=Port.dma(0)) for m in mids],
        joined.prod(),
        reader.tile,
        [k * LINE for k in range(SENDERS)],
        [],
    )

    def sequence(src_, dst_, *handles):
        *fills, drain = handles
        for k, fill in enumerate(fills):
            lines = TensorAccessPattern(
                (SENDERS * LINES * LINE,), k * LINES * LINE, [LINES * LINE], [1]
            )
            fill.fill(src_, tap=lines)
        drain.drain(dst_, wait=True)

    rt = Runtime(sequence, [IN_TY, OUT_TY, *(i.prod() for i in ins), out.cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def main():
    rng = np.random.default_rng(0)
    src = rng.integers(-1000, 1000, size=SENDERS * LINES * LINE, dtype=np.int32)
    sent = src.reshape(SENDERS, LINES, LINE) + (
        np.arange(1, SENDERS + 1, dtype=np.int32)[:, None, None] * 1000
    )
    ref = (sent.transpose(1, 0, 2).reshape(-1) * 2).astype(np.int32)

    src_tensor = iron.tensor(src, dtype=np.int32, device="npu")
    dst_tensor = iron.zeros(ref.size, dtype=np.int32, device="npu")
    out_of_order_join(src_tensor, dst_tensor)
    if not np.array_equal(dst_tensor.numpy(), ref):
        print("FAIL: output mismatch")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
