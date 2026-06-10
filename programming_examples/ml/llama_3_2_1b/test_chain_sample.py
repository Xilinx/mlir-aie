"""Phase 6b: chain + lm_head + sample integration bit-exact test.

End-to-end: random per-layer chain fixtures (same as test_chain_real) +
random lm_head matrix, dispatched through ONE xclbin that runs all 16
decoder layers, the lm_head GEMM, and the sampler on-device. Output is
a single int32 token id; compared against
    numpy_sample.sample_reference(
        numpy_gemm(numpy_chain(x_in), lm_w),  # lm_head replayed in numpy
        temperature, top_k, seed)
for three (temperature, top_k, seed) triples: greedy, full-vocab
multinomial, top-k multinomial.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_sample import (
    D,
    QD,
    KVD,
    HD,
    HEAD_D,
    N_HEADS,
    N_KV,
    T,
    N_LAYERS,
    V,
    PER_LAYER_W,
    PER_LAYER_KV,
    TOTAL_W,
    TOTAL_KV,
    LMW_BYTES,
    PARAMS_BYTES,
    AUX_OFF_LMW,
    AUX_OFF_PARAMS,
    AUX_BYTES,
    OFF_K,
    OFF_V,
    KCACHE_BYTES,
    VCACHE_BYTES,
    RIGHT_SHIFT,
)
from test_chain_real import (
    gen_layer,
    pack_one_layer,
    numpy_single_layer,
    numpy_gemm,
)
from test_sample import pack_params
from numpy_sample import sample_reference, EXP_QUANT_SCALE
from gen_exp_lut import exp_lut


def numpy_lm_head(x_i8, lm_w_i8, lm_b_i32):
    """D-in i8 activation -> V-out i8 logits via banker-SRS requant.
    Same path as test_chain_real.numpy_gemm with K=D, N=V."""
    return numpy_gemm(x_i8, lm_w_i8, lm_b_i32, D, V, RIGHT_SHIFT)


def gen_lm_head(rng):
    return {
        "w": rng.integers(-32, 33, size=(V, D), dtype=np.int8),
        "b": rng.integers(-100, 100, size=V, dtype=np.int32),
    }


def pack_lm_head(buf, lm):
    """Pack into [V*D int8 weights | V*4 int32 bias] -- same packed
    layout the chain projections use."""
    flat = lm["w"].flatten()
    buf[0 : flat.size] = flat
    bb = lm["b"].view(np.int8).flatten()
    buf[flat.size : flat.size + bb.size] = bb


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    import os

    seed = int(os.environ.get("LLAMA_CHAIN_SEED", "0"))
    rng = np.random.default_rng(seed)

    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)
    layers = [gen_layer(rng) for _ in range(N_LAYERS)]
    lm = gen_lm_head(rng)

    # --- Numpy reference: chain -> lm_head -> logits ---
    x = x_in.copy()
    for layer in layers:
        x = numpy_single_layer(x, layer)
    expected_logits = numpy_lm_head(x, lm["w"], lm["b"])

    # --- Pack the chain weight blob (same as test_chain_real) ---
    wblob = np.zeros(TOTAL_W, dtype=np.int8)
    kvblob = np.zeros(TOTAL_KV, dtype=np.int8)
    for L, layer in enumerate(layers):
        pack_one_layer(wblob, L * PER_LAYER_W, layer)
        base_kv = L * PER_LAYER_KV
        kvblob[base_kv + OFF_K : base_kv + OFF_K + KCACHE_BYTES] = layer["kcache"]
        kvblob[base_kv + OFF_V : base_kv + OFF_V + VCACHE_BYTES] = layer["vcache"]

    npu_opts = test_utils.create_npu_kernel(opts)
    lut = exp_lut(EXP_QUANT_SCALE).astype(np.float32)

    def pack_aux(temperature, top_k, prng_seed):
        """Stuff lm_head matrix + sampler params into one DRAM blob; the
        IRON design pulls each section by offset. Done so we stay under
        DefaultNPURuntime's ~5-XRT-arg ceiling (6 args segfaults)."""
        aux = np.zeros(AUX_BYTES, dtype=np.int8)
        pack_lm_head(aux[AUX_OFF_LMW : AUX_OFF_LMW + LMW_BYTES], lm)
        params = pack_params(temperature, top_k, prng_seed)
        aux[AUX_OFF_PARAMS : AUX_OFF_PARAMS + PARAMS_BYTES] = params.view(
            np.int8
        ).flatten()
        return aux

    def run_npu(temperature, top_k, prng_seed):
        x_t = iron.tensor(x_in.copy(), dtype=np.int8)
        w_t = iron.tensor(wblob, dtype=np.int8)
        kv_t = iron.tensor(kvblob, dtype=np.int8)
        aux_t = iron.tensor(pack_aux(temperature, top_k, prng_seed), dtype=np.int8)
        tok_t = iron.zeros([1], dtype=np.int32)
        rc = DefaultNPURuntime.run_test(
            npu_opts.npu_kernel,
            [x_t, w_t, kv_t, aux_t, tok_t],
            {},
            verify=False,
            verbosity=opts.verbosity,
        )
        if rc != 0:
            return None
        tok_t.to("cpu")
        return int(tok_t.numpy()[0])

    modes = [
        ("greedy   ", 0.0, 0, 0),
        ("temp 0.7 ", 0.7, 0, 1),
        ("topk 40  ", 0.7, 40, 42),
    ]

    all_ok = True
    print(f"chain+lm_head+sample: N_LAYERS={N_LAYERS}  D={D}  V={V}")
    for label, temperature, top_k, prng_seed in modes:
        npu = run_npu(temperature, top_k, prng_seed)
        expt = sample_reference(expected_logits, temperature, top_k, prng_seed, lut)
        ok = npu == expt
        all_ok = all_ok and ok
        print(
            f"  [{label}] seed={prng_seed:3d}  NPU={npu:5d}  ref={expt:5d}  "
            f"logit_NPU={int(expected_logits[npu]):4d}  "
            f"logit_ref={int(expected_logits[expt]):4d}  "
            f"{'OK' if ok else 'MISMATCH'}"
        )

    print()
    if all_ok:
        print("BIT-EXACT PASS  (16-layer chain + lm_head + sample, all 3 modes)")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
