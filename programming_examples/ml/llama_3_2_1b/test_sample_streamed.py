"""M2 test: streamed full sampler over a DDR fp32[V] logits buffer, bit-exact
vs numpy_sample_streamed at real vocab V=128256.

Three modes (mirroring test_sample.py but at real V and on fp32 logits):
  - greedy (temperature=0): device token == numpy first-occurrence argmax
  - full softmax (temp=0.7, top_k=0): 4 seeds match the inverse-CDF draw
  - top-k (temp=0.7, top_k=40): 4 seeds match the top-k draw

Run:
  make sample_streamed
  python test_sample_streamed.py -x build/final_sample_streamed.xclbin \\
      -i build/insts_sample_streamed.bin -k MLIR_AIE
"""

from __future__ import annotations

import struct
import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_sample_streamed import VOCAB
from numpy_sample_streamed import sample_streamed_reference
from numpy_sample import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut


def pack_params(temperature, top_k, seed):
    t_bits = np.frombuffer(np.float32(temperature).tobytes(), dtype=np.uint32)[0]
    tk_bits = np.frombuffer(np.int32(top_k).tobytes(), dtype=np.uint32)[0]
    return np.asarray([t_bits, tk_bits, np.uint32(seed & 0xFFFFFFFF)], dtype=np.uint32)


def run_npu(npu_kernel, logits, temperature, top_k, seed, opts):
    l_t = iron.tensor(logits.astype(np.float32), dtype=np.float32)
    p_t = iron.tensor(pack_params(temperature, top_k, seed), dtype=np.uint32)
    tok_t = iron.zeros([1], dtype=np.int32)
    rc = DefaultNPURuntime.run_test(
        npu_kernel, [l_t, p_t, tok_t], {}, verify=False, verbosity=opts.verbosity
    )
    if rc != 0:
        return None
    tok_t.to("cpu")
    return int(tok_t.numpy()[0])


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    rng = np.random.default_rng(0)
    # fp32 logits in a realistic range (lm_head logits are ~[-15, 15]).
    logits = rng.uniform(-12.0, 12.0, size=VOCAB).astype(np.float32)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    print(f"sample_streamed  V={VOCAB}", flush=True)

    fails = 0

    a = run_npu(npu_kernel, logits, 0.0, 0, 0, opts)
    e = sample_streamed_reference(logits, 0.0, 0, 0, lut)
    ok = a == e
    fails += not ok
    print(f"[greedy ]  dev={a}  ref={e}  {'PASS' if ok else 'FAIL'}")

    seeds = [0, 1, 7, 42]
    for s in seeds:
        a = run_npu(npu_kernel, logits, 0.7, 0, s, opts)
        e = sample_streamed_reference(logits, 0.7, 0, s, lut)
        ok = a == e
        fails += not ok
        print(
            f"[full   ]  seed={s:3d}  dev={a:6d}  ref={e:6d}  {'OK' if ok else 'MISMATCH'}"
        )

    for s in seeds:
        a = run_npu(npu_kernel, logits, 0.7, 40, s, opts)
        e = sample_streamed_reference(logits, 0.7, 40, s, lut)
        ok = a == e
        fails += not ok
        print(
            f"[topk 40]  seed={s:3d}  dev={a:6d}  ref={e:6d}  {'OK' if ok else 'MISMATCH'}"
        )

    print(f"\nsample_streamed: {'BIT-EXACT PASS' if fails == 0 else f'{fails} FAIL'}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
