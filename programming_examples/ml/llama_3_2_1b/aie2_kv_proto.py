"""Stage 0 plumbing prototype for on-chip KV append.

Validates the ONE genuine hardware unknown before building the real append:
a worker (kv_append) consumes a HOST-FILLED kv buffer, modifies slot[pos]
in-array, and produces an updated buffer that is BOTH
  (a) consumed by a second worker (kv_read) -- read-after-write ordering, and
  (b) drained back to host
via producer fan-out to two .cons() endpoints (one worker + one drain).

This is exactly the topology the real Stage 1 needs: the append worker feeds
flowkv_mh (consumer) AND the host drain (device-owned cache round-trip).
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

HEAD_D = 64
T = 128
PREFIX = 8
SCALE_BYTES = T * 4
BODY_BYTES = T * HEAD_D
KHALF = SCALE_BYTES + BODY_BYTES
PER_HEAD = PREFIX + 2 * KHALF
READ_BYTES = 16


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def build():
    t_kv = _i8(PER_HEAD)
    t_read = _i8(READ_BYTES)

    of_kv_in = ObjectFifo(t_kv, depth=1, name="kv_in")
    of_kv_out = ObjectFifo(t_kv, depth=1, name="kv_out")
    of_read = ObjectFifo(t_read, depth=1, name="read")

    KO = "llama_kv_proto.cc.o"
    k_append = Kernel("llama_kv_proto_append", KO, [t_kv, t_kv])
    k_read = Kernel("llama_kv_proto_read", KO, [t_kv, t_read])

    def w_append(c_in, c_out, k):
        a = c_in.acquire(1)
        o = c_out.acquire(1)
        k(a, o)
        c_in.release(1)
        c_out.release(1)

    def w_read(c_in, c_out, k):
        a = c_in.acquire(1)
        o = c_out.acquire(1)
        k(a, o)
        c_in.release(1)
        c_out.release(1)

    workers = [
        Worker(w_append, [of_kv_in.cons(), of_kv_out.prod(), k_append], tile=Tile(0, 2)),
        # kv_out fans to BOTH the read worker and the host drain.
        Worker(w_read, [of_kv_out.cons(), of_read.prod(), k_read], tile=Tile(0, 3)),
    ]

    rt = Runtime()
    with rt.sequence(t_kv, t_kv, t_read) as (kv_in, kv_out, read):
        rt.start(*workers)
        rt.fill(of_kv_in.prod(), kv_in)
        # Drain the updated cache (second consumer of of_kv_out) and the proof.
        rt.drain(of_kv_out.cons(), kv_out, wait=True)
        rt.drain(of_read.cons(), read, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
