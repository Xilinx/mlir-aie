"""Phase 1.9 dataflow sanity test: full single-layer integration.

Computes the expected output of the all-stubs single-layer pipeline
for a random x_in and bit-exact compares.

Trace (all on int8; int8 add wraps via .astype(np.int8) which numpy
guarantees as modular reduction):

  h1   = x_in                          (rmsnorm1: copy D->D)
  qf   = x_in                          (q_proj: copy D->QD, QD == D)
  kf   = x_in[:KVD]                    (k_proj: copy D->KVD)
  vf   = x_in[:KVD]                    (v_proj)
  qr   = qf                            (rope_q: copy)
  kr   = kf                            (rope_k: copy)
  rec  = qr                            (flowkv_qk stub drops kr)
  af   = rec                           (flowkv_sv stub drops vf)
  op   = af                            (o_proj: copy QD->D, QD == D)
  x1   = op + x_in = 2*x_in            (add1, int8 wrap)
  h2   = x1                            (rmsnorm2: copy)
  gf   = tile(h2 -> HD) = h2 repeated 4x  (gate_proj: tile, HD = 4*D)
  uf   = tile(h2 -> HD) = same
  sf   = gf + uf = 2*gf                (silu_mul, int8 wrap)
  df   = sf[:D] = first D bytes of sf  (down_proj: copy HD->D)
       = first D bytes of 2*tile(2*x_in) = 2*2*x_in = 4*x_in
  out  = df + x1 = 4*x_in + 2*x_in = 6*x_in   (add2, int8 wrap)
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime


D = 2048


def expected_output(x_in: np.ndarray) -> np.ndarray:
    # All adds on int8 with wrap. numpy's int8 arithmetic on int8 inputs
    # wraps modularly per IEEE-style two's-complement semantics, but only
    # safely via int32 intermediate -> .astype(int8). int8 op int8 in
    # numpy promotes to int8 anyway, but with overflow warnings.
    # Use uint8 view for clean modular arithmetic.
    u = x_in.view(np.uint8)
    six_x = (6 * u.astype(np.uint16)).astype(np.uint8)
    return six_x.view(np.int8)


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    rng = np.random.default_rng(0)
    x_in = rng.integers(-128, 128, size=D, dtype=np.int8)

    in_t  = iron.tensor(x_in, dtype=np.int8)
    out_t = iron.zeros([D], dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [in_t, out_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    out_t.to("cpu")
    actual = out_t.numpy()
    expected = expected_output(x_in)

    n_diff = int((actual != expected).sum())
    if n_diff == 0:
        print(f"BIT-EXACT NPU full single-layer dataflow OK ({D} bytes)")
        return 0

    print(f"MISMATCH: {n_diff}/{D} bytes differ")
    # Diagnose: is the output 0, x_in, ~x_in, or something else?
    if int((actual == np.zeros_like(actual)).sum()) == D:
        print("  output = all zeros -> pipeline almost certainly did not run")
    elif int((actual == x_in).sum()) == D:
        print("  output = x_in -> add1 or add2 dropped a fifo, or no wrap")
    else:
        diffs = np.argwhere(actual != expected)[:8]
        for (i,) in diffs:
            print(f"  out[{i}]={actual[i]}  expected={expected[i]}  x_in[{i}]={x_in[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
