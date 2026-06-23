"""Test the persistent on-device decode loop vs a numpy tied-LM autoregressive ref."""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_persist_decode_probe import V, D, T_STEPS


def numpy_decode(table, seed_tok, T):
    """Mirror the kernel: next = (sum of embed[tok] bytes as uint8) mod V."""
    toks = []
    t = int(seed_tok)
    V_ = table.shape[0]
    for _ in range(T):
        emb_u8 = table[t].astype(np.uint8).astype(np.uint32)
        t = int(emb_u8.sum() % V_)
        toks.append(t)
    return toks


def main():
    opts = test_utils.create_default_argparser().parse_args()
    seed_tok = 0
    table = np.random.default_rng(0).integers(-128, 128, size=(V, D), dtype=np.int8)
    ref = numpy_decode(table, seed_tok, T_STEPS)
    print(f"numpy seq has {len(set(ref))} distinct tokens")

    tbl_t = iron.tensor(table.flatten(), dtype=np.int8)
    seed_t = iron.tensor(np.array([seed_tok], np.int32), dtype=np.int32)
    out_t = iron.zeros([T_STEPS], dtype=np.int32)
    npu = test_utils.create_npu_kernel(opts).npu_kernel
    rc = DefaultNPURuntime.run_test(
        npu, [tbl_t, seed_t, out_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        print(f"dispatch returned {rc}", file=sys.stderr)
        return rc
    out_t.to("cpu")
    dev = [int(x) for x in out_t.numpy()]

    ok = dev == ref
    for k in range(T_STEPS):
        m = "ok" if dev[k] == ref[k] else "DIFF"
        print(f"  step {k}: dev={dev[k]:4d} ref={ref[k]:4d} {m}")
    print(f"\npersist_decode: {'PASS' if ok else 'FAIL'}  (V={V} D={D} T={T_STEPS})")
    print(f"  device tokens:  {dev}")
    print(f"  numpy tokens:   {ref}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
