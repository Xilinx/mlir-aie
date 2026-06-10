"""Debug probe: flowkv_qk ONLY -- emits probs to a runtime buffer.

Lets the host inspect the EXACT fp32 probabilities the kernel computes,
so we can compare byte-for-byte against numpy's reference and locate
the divergence.
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile


def build(head_dim: int, t: int):
    q_ty = np.ndarray[(head_dim,), np.dtype[np.int8]]
    k_ty = np.ndarray[(t * head_dim,), np.dtype[np.int8]]
    # Bytes view throughout (probs are 4*T = 64 bytes for T=16). The
    # qk-bytes wrapper reinterprets the buffer as float* internally.
    probs_bytes_ty = np.ndarray[(t * 4,), np.dtype[np.int8]]

    of_q = ObjectFifo(q_ty, name="q")
    of_k = ObjectFifo(k_ty, name="k")
    of_probs = ObjectFifo(probs_bytes_ty, name="probs")  # qk CT -> relay CT
    of_probs_o = ObjectFifo(probs_bytes_ty, name="probs_o")  # relay CT -> shim

    import os

    sym = os.environ.get("LLAMA_PROBE", "llama_flowkv_qk_bytes")
    k_qk = Kernel(
        sym, "llama_flowkv.cc.o", [q_ty, k_ty, probs_bytes_ty, np.float32, np.float32]
    )

    def qk_fn(c_q, c_k, c_probs, k):
        q = c_q.acquire(1)
        kk = c_k.acquire(1)
        p = c_probs.acquire(1)
        k(q, kk, p, 0.05, 0.05)
        c_q.release(1)
        c_k.release(1)
        c_probs.release(1)

    worker_qk = Worker(
        qk_fn, [of_q.cons(), of_k.cons(), of_probs.prod(), k_qk], tile=Tile(0, 4)
    )

    # Passthrough worker uses the existing llama_pt_copy_D_to_D kernel
    # at the right size to byte-copy the probs payload to a shim-bound
    # buffer. We size t*4 = 64 bytes for T=16 fp32 probs -- matches the
    # llama_pt_copy_D_to_D's hardcoded kD=64 byte copy.
    k_copy = Kernel(
        "llama_pt_copy_D_to_D", "llama_layer_pt.cc.o", [probs_bytes_ty, probs_bytes_ty]
    )

    def relay_fn(c_in, c_out, k):
        x = c_in.acquire(1)
        o = c_out.acquire(1)
        k(x, o)
        c_in.release(1)
        c_out.release(1)

    worker_relay = Worker(
        relay_fn, [of_probs.cons(), of_probs_o.prod(), k_copy], tile=Tile(0, 5)
    )

    rt = Runtime()
    with rt.sequence(q_ty, k_ty, probs_bytes_ty) as (q, k, p):
        rt.start(worker_qk, worker_relay)
        rt.fill(of_q.prod(), q)
        rt.fill(of_k.prod(), k)
        rt.drain(of_probs_o.cons(), p, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=64)
    p.add_argument("-T", type=int, default=16)
    args = p.parse_args(sys.argv[1:])
    print(build(args.head_dim, args.T))


if __name__ == "__main__":
    main()
