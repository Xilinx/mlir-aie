"""Phase 7b perf snapshot helper. Times the chain_mh xclbin's NPU
dispatch (steady-state) and the host-side numpy calibration + blob
pack stages, so the per-token decode bottleneck is obvious.

Run:
  python bench_mh.py -x build/final_chain_mh_N16_T128.xclbin \\
                     -i build/insts_chain_mh_N16_T128.bin -k MLIR_AIE
"""

import sys
import time

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_dynscale_mh import D, N_LAYERS, SILU_GATE_SCALE
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward
from test_chain_mh import pack_blobs
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    opts = p.parse_args()
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel

    rng = np.random.default_rng(0)
    x = rng.integers(-32, 33, size=D, dtype=np.int8)
    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for _ in range(N_LAYERS):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        layers.append(layer)

    t0 = time.time()
    xc = x.copy()
    for L in range(N_LAYERS):
        xc, scales = numpy_layer_mh_forward(xc, layers[L])
        layers[L]["scales"] = scales
    t_numpy = time.time() - t0

    t0 = time.time()
    wblob, kvblob = pack_blobs(layers)
    t_pack = time.time() - t0

    x_t = iron.tensor(x, dtype=np.int8)
    w_t = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t = iron.zeros([D], dtype=np.int8)

    for _ in range(opts.warmup):
        DefaultNPURuntime.run_test(
            npu_kernel, [x_t, w_t, kv_t, o_t], {}, verify=False, verbosity=0
        )
        o_t.to("cpu")
    times = []
    for _ in range(opts.iters):
        t0 = time.time()
        DefaultNPURuntime.run_test(
            npu_kernel, [x_t, w_t, kv_t, o_t], {}, verify=False, verbosity=0
        )
        o_t.to("cpu")
        times.append(time.time() - t0)
    dispatch_ms = float(np.median(times)) * 1000.0
    total_ms = (t_numpy + t_pack) * 1000.0 + dispatch_ms

    print(f"chain_mh N={N_LAYERS} perf:")
    print(f"  numpy calib (host, {N_LAYERS} layers):  {t_numpy*1000:8.1f} ms")
    print(f"  blob pack (host):                    {t_pack *1000:8.1f} ms")
    print(
        f"  NPU dispatch (median of {opts.iters}):     {dispatch_ms:8.1f} ms"
        f"  ({dispatch_ms/N_LAYERS:.2f} ms/layer)"
    )
    print(
        f"  wblob: {wblob.nbytes/1024/1024:.1f} MB, "
        f"kvblob: {kvblob.nbytes/1024:.1f} KB"
    )
    print(
        f"  end-to-end per token:                {total_ms:8.1f} ms"
        f"  ({1000.0/total_ms:.2f} tok/s)"
    )
    print(
        f"  NPU-only ceiling:                                "
        f"({1000.0/dispatch_ms:.2f} tok/s)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
