"""Phase 1.9 dataflow stub: full single-layer integration.

Mirrors the cautious-eureka `build_decode_layer` topology (1 token,
M=1, FlowKV 2-CT attention) with all kernels stubbed. 16 workers; uses
the spec dimensions verbatim (D=2048, QD=2048, KVD=512, HD=8192) so the
ObjectFifo descriptors are the same shapes the real design will use.

Worker placement, mapped to placement.DECODE_PLACEMENT plus spare tiles:

  rmsnorm1   -> Tile(5, 4)   [rmsnorm]
  q_proj     -> Tile(0, 2)   [projection r2]
  k_proj     -> Tile(1, 2)
  v_proj     -> Tile(2, 2)
  rope_q     -> Tile(4, 4)   [rope]
  rope_k     -> Tile(7, 4)   [spare]
  flowkv_qk  -> Tile(0, 4)   [attn pair0 qk]
  flowkv_sv  -> Tile(0, 5)   [attn pair0 sv]
  o_proj     -> Tile(3, 2)
  add1       -> Tile(6, 5)   [spare]
  rmsnorm2   -> Tile(6, 4)   [spare]
  gate_proj  -> Tile(4, 2)
  up_proj    -> Tile(5, 2)
  silu_mul   -> Tile(4, 5)   [silu]
  down_proj  -> Tile(6, 2)
  add2       -> Tile(7, 5)   [spare]

Trace (all stubs are passthrough/tile/add; see test_layer_pt.py):
  layer_out = 6 * x_in  (mod 256, int8)
"""

import argparse
import sys

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2, Tile

D = 2048
QD = 2048
KVD = 512
HD = 8192


def _i8(n):
    return np.ndarray[(int(n),), np.dtype[np.int8]]


def build():
    t_D, t_QD, t_KVD, t_HD = _i8(D), _i8(QD), _i8(KVD), _i8(HD)

    # ObjectFifos. depth=4 on residual-hold fifos (x_in, x1) for safety.
    of_in = ObjectFifo(t_D, depth=4, name="x_in")
    of_h1 = ObjectFifo(t_D, name="h1")
    of_qf = ObjectFifo(t_QD, name="qf")
    of_kf = ObjectFifo(t_KVD, name="kf")
    of_vf = ObjectFifo(t_KVD, name="vf")
    of_qr = ObjectFifo(t_QD, name="qr")
    of_kr = ObjectFifo(t_KVD, name="kr")
    of_rec = ObjectFifo(t_QD, name="rec")
    of_af = ObjectFifo(t_QD, name="af")
    of_op = ObjectFifo(t_D, name="op")
    of_x1 = ObjectFifo(t_D, depth=4, name="x1")
    of_h2 = ObjectFifo(t_D, name="h2")
    of_gf = ObjectFifo(t_HD, name="gf")
    of_uf = ObjectFifo(t_HD, name="uf")
    of_sf = ObjectFifo(t_HD, name="sf")
    of_df = ObjectFifo(t_D, name="df")
    of_out = ObjectFifo(t_D, name="layer_out")

    KO = "llama_layer_pt.cc.o"

    # Shape-specific Kernel decls (each symbol unique).
    k_copy_D_D = Kernel("llama_pt_copy_D_to_D", KO, [t_D, t_D])
    k_copy_D_QD = Kernel("llama_pt_copy_D_to_QD", KO, [t_D, t_QD])
    k_copy_D_KVD = Kernel("llama_pt_copy_D_to_KVD", KO, [t_D, t_KVD])
    k_copy_QD_QD = Kernel("llama_pt_copy_QD_to_QD", KO, [t_QD, t_QD])
    k_copy_QD_D = Kernel("llama_pt_copy_QD_to_D", KO, [t_QD, t_D])
    k_copy_KVD_KVD = Kernel("llama_pt_copy_KVD_to_KVD", KO, [t_KVD, t_KVD])
    k_copy_HD_D = Kernel("llama_pt_copy_HD_to_D", KO, [t_HD, t_D])
    k_tile_D_HD = Kernel("llama_pt_tile_D_to_HD", KO, [t_D, t_HD])
    k_add_D = Kernel("llama_pt_add_D", KO, [t_D, t_D, t_D])
    k_add_HD = Kernel("llama_pt_add_HD", KO, [t_HD, t_HD, t_HD])
    k_first_QD_KVD = Kernel("llama_pt_first_QD_KVD", KO, [t_QD, t_KVD, t_QD])
    k_first_QD_QD = Kernel("llama_pt_first_QD_QD", KO, [t_QD, t_QD, t_QD])

    def w_unary(c_in, c_out, k):
        x = c_in.acquire(1)
        o = c_out.acquire(1)
        k(x, o)
        c_in.release(1)
        c_out.release(1)

    def w_binary(c_a, c_b, c_out, k):
        a = c_a.acquire(1)
        b = c_b.acquire(1)
        o = c_out.acquire(1)
        k(a, b, o)
        c_a.release(1)
        c_b.release(1)
        c_out.release(1)

    workers = [
        # rmsnorm1: x_in -> h1
        Worker(w_unary, [of_in.cons(), of_h1.prod(), k_copy_D_D], tile=Tile(5, 4)),
        # q_proj
        Worker(w_unary, [of_h1.cons(), of_qf.prod(), k_copy_D_QD], tile=Tile(0, 2)),
        # k_proj
        Worker(w_unary, [of_h1.cons(), of_kf.prod(), k_copy_D_KVD], tile=Tile(1, 2)),
        # v_proj
        Worker(w_unary, [of_h1.cons(), of_vf.prod(), k_copy_D_KVD], tile=Tile(2, 2)),
        # rope_q
        Worker(w_unary, [of_qf.cons(), of_qr.prod(), k_copy_QD_QD], tile=Tile(4, 4)),
        # rope_k
        Worker(w_unary, [of_kf.cons(), of_kr.prod(), k_copy_KVD_KVD], tile=Tile(7, 4)),
        # flowkv_qk: drops kr
        Worker(
            w_binary,
            [of_qr.cons(), of_kr.cons(), of_rec.prod(), k_first_QD_KVD],
            tile=Tile(0, 4),
        ),
        # flowkv_sv: drops vf  (rec comes first to match k_first_QD_KVD signature)
        Worker(
            w_binary,
            [of_rec.cons(), of_vf.cons(), of_af.prod(), k_first_QD_KVD],
            tile=Tile(0, 5),
        ),
        # o_proj
        Worker(w_unary, [of_af.cons(), of_op.prod(), k_copy_QD_D], tile=Tile(3, 2)),
        # add1
        Worker(
            w_binary,
            [of_op.cons(), of_in.cons(), of_x1.prod(), k_add_D],
            tile=Tile(6, 5),
        ),
        # rmsnorm2
        Worker(w_unary, [of_x1.cons(), of_h2.prod(), k_copy_D_D], tile=Tile(6, 4)),
        # gate_proj
        Worker(w_unary, [of_h2.cons(), of_gf.prod(), k_tile_D_HD], tile=Tile(4, 2)),
        # up_proj
        Worker(w_unary, [of_h2.cons(), of_uf.prod(), k_tile_D_HD], tile=Tile(5, 2)),
        # silu_mul
        Worker(
            w_binary,
            [of_gf.cons(), of_uf.cons(), of_sf.prod(), k_add_HD],
            tile=Tile(4, 5),
        ),
        # down_proj
        Worker(w_unary, [of_sf.cons(), of_df.prod(), k_copy_HD_D], tile=Tile(6, 2)),
        # add2
        Worker(
            w_binary,
            [of_df.cons(), of_x1.cons(), of_out.prod(), k_add_D],
            tile=Tile(7, 5),
        ),
    ]

    rt = Runtime()
    with rt.sequence(t_D, t_D) as (a, o):
        rt.start(*workers)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), o, wait=True)

    return Program(NPU2(), rt).resolve_program()


def main():
    argparse.ArgumentParser().parse_args(sys.argv[1:])
    print(build())


if __name__ == "__main__":
    main()
