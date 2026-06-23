"""Probe 2: validate memtile->compute REPLAY+CHUNK via StaticWeightStream.

Isolates the consumer side of the M3a logits relay: a memtile holds a resident
HALF buffer; the consumer reads it in CHUNK windows, and the whole HALF is
replayed R times (CHUNKS_PER_HALF*R total pulls). Uses the proven
yolo26n/mobilenet StaticWeightStream (memtile MM2S whole-buffer infinite loop +
lock-gated chunked consumer S2MM) -- the GEMM-fill side is validated separately.

Here the HALF is a compile-time initial_value (known bytes) so the host can
check every replayed chunk. dtype = int8 (the relay just moves bytes; fp32 in
M3a moves identically). Output: out[(p*N_CHUNKS+c)*CHUNK ...] == half[c*CHUNK...]
for every pass p, chunk c.

Env: LLAMA_RP_HALF (default 256), LLAMA_RP_CHUNK (default 64), LLAMA_RP_R (3).
"""

import argparse
import os as _os
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_

from lowlevel_dma import StaticWeightStream

HALF = int(_os.environ.get("LLAMA_RP_HALF", "256"))
CHUNK = int(_os.environ.get("LLAMA_RP_CHUNK", "64"))
R = int(_os.environ.get("LLAMA_RP_R", "3"))
N_CHUNKS = HALF // CHUNK


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


# Known HALF data (also used by the test to build expected output).
HALF_DATA = (np.arange(HALF, dtype=np.int32) % 127 - 63).astype(np.int8)


def build():
    t_half = _i8(HALF)
    t_chunk = _i8(CHUNK)
    rt_out_ty = _i8(N_CHUNKS * R * CHUNK)

    of_out = ObjectFifo(t_chunk, depth=2, name="rp_out")

    relay = StaticWeightStream(
        obj_type=t_half,
        initial_value=HALF_DATA,
        name="rp_relay",
        recv_type=t_chunk,
        repeat_count=R,
        memtile_placement=Tile(1, 1),
        compute_placement=Tile(0, 2),
        mm2s_channel=0,
        s2mm_channel=0,
        mem_lock_id=0,
        comp_lock_id=0,
    )

    KO = "llama_passthrough_f32.cc.o"
    k_copy = Kernel("passthrough_i8_chunk", KO, [t_chunk, t_chunk])

    def w_consume(relay_h, c_out, k):
        for _ in range_(R):
            for _c in range_(N_CHUNKS):
                rb = relay_h.acquire(1)
                o = c_out.acquire(1)
                k(rb, o)
                c_out.release(1)
                relay_h.release(1)

    w = Worker(
        w_consume,
        [relay, of_out.prod(), k_copy],
        tile=Tile(0, 2),
        stack_size=2048,
    )

    rt = Runtime()
    with rt.sequence(rt_out_ty) as dst:
        rt.start(w)
        tg = rt.task_group()
        rt.drain(of_out.cons(), dst, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
