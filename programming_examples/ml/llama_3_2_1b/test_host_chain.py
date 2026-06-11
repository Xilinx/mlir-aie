"""Debug: host-orchestrated N-layer chain using the single-layer xclbin.

Dispatches the SINGLE-LAYER xclbin (test_layer_real.py's design) N times
sequentially from the host, feeding each layer's output as the next
layer's input. Compares against the SAME numpy chain reference that
test_chain_real.py uses.

If this matches the numpy chain bit-exact, then test_chain_real.py's
IRON-dataflow chain has a real bug (something added by the dataflow
chain itself). If THIS also diverges, then the numpy chain reference is
missing some precision detail of how the kernels actually compose.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

# Use the SINGLE-layer constants and packing.
from aie2_layer_real import (
    D,
    HD,
    HEAD_D,
    T,
    OFF_GAMMA_IN,
    OFF_GAMMA_POST,
    OFF_WQ,
    OFF_WO,
    OFF_WG,
    OFF_WU,
    OFF_WD,
    OFF_CS,
    GAMMA_BYTES,
    WQ_BYTES,
    WO_BYTES,
    WG_BYTES,
    WU_BYTES,
    WD_BYTES,
    CS_BYTES,
    KCACHE_BYTES,
    VCACHE_BYTES,
    TOTAL_W,
    KV_TOTAL,
    OFF_K,
    OFF_V,
)
from test_chain_real import (
    gen_layer,
    pack_one_layer,
    numpy_single_layer,
)
from aie2_chain_real import N_LAYERS, PER_LAYER_KV


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    import os

    seed = int(os.environ.get("LLAMA_CHAIN_SEED", "0"))
    rng = np.random.default_rng(seed)

    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)
    layers = [gen_layer(rng) for _ in range(N_LAYERS)]

    # --- Numpy chain reference (same as test_chain_real) ---
    x = x_in.copy()
    for layer in layers:
        x = numpy_single_layer(x, layer)
    expected = x

    print(f"x_in[:8]={x_in[:8]}", flush=True)
    print(f"expected[:8]={expected[:8]}", flush=True)

    # --- Host-orchestrated NPU chain via the SINGLE-LAYER xclbin ---
    npu_opts = test_utils.create_npu_kernel(opts)
    print("npu_opts created", flush=True)

    x = x_in.copy()
    for L, layer in enumerate(layers):
        print(f"  L={L} start", flush=True)
        # Compute numpy BEFORE NPU to rule out aliasing
        layer_expected = numpy_single_layer(x, layer)
        print(f"  L={L} numpy (pre-NPU): out[:8]={layer_expected[:8]}", flush=True)
        print(
            f"  L={L} layer['wq'].sum()={int(layer['wq'].sum())} mean={float(np.abs(layer['wq']).mean()):.2f}",
            flush=True,
        )
        wblob = np.zeros(TOTAL_W, dtype=np.int8)
        pack_one_layer(wblob, 0, layer)
        print(f"  L={L} packed", flush=True)
        kvbuf = np.zeros(KV_TOTAL, dtype=np.int8)
        kvbuf[OFF_K : OFF_K + KCACHE_BYTES] = layer["kcache"]
        kvbuf[OFF_V : OFF_V + VCACHE_BYTES] = layer["vcache"]

        x_t = iron.tensor(x.copy(), dtype=np.int8)
        w_t = iron.tensor(wblob, dtype=np.int8)
        kv_t = iron.tensor(kvbuf, dtype=np.int8)
        o_t = iron.zeros([D], dtype=np.int8)

        rc = DefaultNPURuntime.run_test(
            npu_opts.npu_kernel,
            [x_t, w_t, kv_t, o_t],
            {},
            verify=False,
            verbosity=0,
        )
        if rc != 0:
            print(f"layer {L}: NPU dispatch failed rc={rc}")
            return rc

        o_t.to("cpu")
        layer_out = o_t.numpy()

        # (layer_expected was computed pre-NPU above)
        diff = (layer_out != layer_expected).sum()
        print(f"  layer {L}: NPU vs numpy mismatches={int(diff)}/{D}", flush=True)
        if diff > 0:
            print(f"    NPU   [:8]={layer_out[:8]}", flush=True)
            print(f"    numpy [:8]={layer_expected[:8]}", flush=True)
            print(f"    x_in  [:8]={x[:8]}", flush=True)
        x = layer_out

    diff = (x != expected).sum()
    print(
        f"host-orchestrated chain: N_LAYERS={N_LAYERS}  total mismatches={int(diff)}/{D}"
    )
    if diff == 0:
        print("BIT-EXACT vs numpy chain (so IRON-dataflow chain has its own bug)")
    else:
        print("NUMPY REF DIVERGES from kernel chain even with single-layer xclbin")
    return 0


if __name__ == "__main__":
    sys.exit(main())
