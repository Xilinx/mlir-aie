"""Phase 6a sample test: bit-exact greedy + multinomial + top-k.

Compares NPU sampler against numpy_sample.py (which mirrors the kernel's
xoshiro128++ PRNG + exp LUT + sw_recip arithmetic bit-for-bit).

Three test modes, all bit-exact:
  - greedy   (temperature=0.0): NPU token == numpy argmax (first-occurrence)
  - full SM  (temperature=0.7, top_k=0): 4 seeds match numpy multinomial draw
  - top-k    (temperature=0.7, top_k=40): 4 seeds match numpy top-k draw
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from numpy_sample import sample_reference, EXP_QUANT_SCALE
from gen_exp_lut import exp_lut


def pack_params(temperature, top_k, seed):
    """Pack 3 scalars into a uint32[3] matching the kernel's layout:
    [0]=temperature fp32 bits, [1]=top_k int32, [2]=seed uint32."""
    t_bits = np.frombuffer(np.float32(temperature).tobytes(), dtype=np.uint32)[0]
    tk_bits = np.frombuffer(np.int32(top_k).tobytes(), dtype=np.uint32)[0]
    return np.asarray([t_bits, tk_bits, np.uint32(seed & 0xFFFFFFFF)], dtype=np.uint32)


def run_npu(npu_opts, logits, temperature, top_k, seed, opts):
    V = logits.size
    l_t = iron.tensor(logits, dtype=np.int8)
    t_t = iron.zeros([1], dtype=np.int32)
    p_t = iron.tensor(pack_params(temperature, top_k, seed), dtype=np.uint32)

    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [l_t, t_t, p_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return None
    t_t.to("cpu")
    return int(t_t.numpy()[0])


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("-V", type=int, default=1024)
    opts = p.parse_args()

    V = opts.V
    rng = np.random.default_rng(0)
    logits = rng.integers(-128, 128, size=V, dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    # --- Greedy ---
    actual = run_npu(npu_opts, logits, temperature=0.0, top_k=0, seed=0, opts=opts)
    expected = sample_reference(logits, temperature=0.0, top_k=0, seed=0, lut=lut)
    print(
        f"[greedy ]  V={V}  NPU={actual}  ref={expected}  "
        f"logit_NPU={int(logits[actual])}  logit_ref={int(logits[expected])}"
    )
    greedy_ok = actual == expected

    # --- Full-vocab multinomial across seeds ---
    seeds = [0, 1, 7, 42]
    full_ok = True
    print(f"[temp 0.7] full-vocab multinomial:")
    for s in seeds:
        a = run_npu(npu_opts, logits, temperature=0.7, top_k=0, seed=s, opts=opts)
        e = sample_reference(logits, temperature=0.7, top_k=0, seed=s, lut=lut)
        ok = a == e
        full_ok = full_ok and ok
        print(f"   seed={s:3d}  NPU={a:5d}  ref={e:5d}  {'OK' if ok else 'MISMATCH'}")

    # --- Top-k multinomial across seeds ---
    topk_ok = True
    print(f"[topk 40]  temperature=0.7, top_k=40:")
    for s in seeds:
        a = run_npu(npu_opts, logits, temperature=0.7, top_k=40, seed=s, opts=opts)
        e = sample_reference(logits, temperature=0.7, top_k=40, seed=s, lut=lut)
        ok = a == e
        topk_ok = topk_ok and ok
        print(f"   seed={s:3d}  NPU={a:5d}  ref={e:5d}  {'OK' if ok else 'MISMATCH'}")

    all_ok = greedy_ok and full_ok and topk_ok
    print()
    print(f"  greedy: {'PASS' if greedy_ok else 'FAIL'}")
    print(f"  full multinomial: {'PASS' if full_ok else 'FAIL'}")
    print(f"  top-k multinomial: {'PASS' if topk_ok else 'FAIL'}")
    if all_ok:
        print("BIT-EXACT PASS (all sampler modes)")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
