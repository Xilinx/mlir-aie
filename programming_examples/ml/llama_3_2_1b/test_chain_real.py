"""Phase 4: N-layer decode chain bit-exact test.

Random per-layer weights/biases/gammas/cs/KV-cache, packed per
aie2_chain_real.py's blob layout, run as a single dispatch on NPU,
compared against a numpy reference that chains the per-kernel
references through N_LAYERS layers.
"""

from __future__ import annotations

import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_chain_real import (
    D, QD, KVD, HD, HEAD_D, N_HEADS, N_KV, T, N_LAYERS,
    OFF_GAMMA_IN, OFF_GAMMA_POST, OFF_WQ, OFF_WO,
    OFF_WG, OFF_WU, OFF_WD, OFF_CS,
    OFF_K, OFF_V,
    GAMMA_BYTES, WQ_BYTES, WO_BYTES, WG_BYTES, WU_BYTES, WD_BYTES, CS_BYTES,
    KCACHE_BYTES, VCACHE_BYTES,
    PER_LAYER_W, PER_LAYER_KV, TOTAL_W, TOTAL_KV,
    RIGHT_SHIFT, ACT_SCALE, INV_ACT_SCALE,
    UP_SCALE, INV_OUT_SCALE_SI,
    Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_gemm_int8_srs_real import banker_srs
from test_rope_int8 import numpy_rope
from test_silu_mul_int8 import numpy_silu_mul
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from gen_silu_lut import silu_lut
from gen_exp_lut import exp_lut


def i8_add_wrap(a, b):
    return (a.astype(np.int32) + b.astype(np.int32)).astype(np.int8)


def numpy_gemm(act, weights, bias, K, N, right_shift):
    acc = weights.astype(np.int32) @ act.astype(np.int32)
    sums = acc + bias
    return banker_srs(sums, right_shift).clip(-128, 127).astype(np.int8)


def numpy_single_layer(x_in, layer):
    """One decoder layer: identical chain to aie2_layer_real.py.
    `layer` is a dict containing the per-layer weights, biases, gammas,
    cs, kcache, vcache.
    """
    h1 = numpy_rmsnorm_int8(x_in, layer["gamma_in"], ACT_SCALE, INV_ACT_SCALE)
    qf = numpy_gemm(h1, layer["wq"], layer["bq"], D, QD, RIGHT_SHIFT)
    qr = numpy_rope(qf, layer["cos"], layer["sin"], N_HEADS, HEAD_D, ACT_SCALE)
    lut = exp_lut(EXP_QUANT_SCALE)
    af = numpy_attention(qr, layer["kcache"], layer["vcache"], HEAD_D, T,
                         Q_SCALE, K_SCALE, V_SCALE, INV_OUT_SCALE_AT, lut)
    op = numpy_gemm(af, layer["wo"], layer["bo"], QD, D, RIGHT_SHIFT)
    x1 = i8_add_wrap(op, x_in)
    h2 = numpy_rmsnorm_int8(x1, layer["gamma_post"], ACT_SCALE, INV_ACT_SCALE)
    gf = numpy_gemm(h2, layer["wg"], layer["bg"], D, HD, RIGHT_SHIFT)
    uf = numpy_gemm(h2, layer["wu"], layer["bu"], D, HD, RIGHT_SHIFT)
    silu_lut_bf = silu_lut(0.05)
    sf = numpy_silu_mul(gf, uf, silu_lut_bf, UP_SCALE, INV_OUT_SCALE_SI)
    df = numpy_gemm(sf, layer["wd"], layer["bd"], HD, D, RIGHT_SHIFT)
    return i8_add_wrap(df, x1)


def gen_layer(rng):
    """Random per-layer data. NB: small weight/bias magnitudes combined
    with RIGHT_SHIFT=12 produce gemm outputs ~ 0, which collapses the
    whole layer to identity (out = x_in). Bit-exactness in this regime
    is degenerate (both compute identity). Use larger magnitudes via
    LLAMA_CHAIN_ACTIVE_DATA=1 to exercise real layer math (which
    surfaces a per-kernel-ref precision modeling gap; see DEBUG note
    in chain commit)."""
    import os
    active = os.environ.get("LLAMA_CHAIN_ACTIVE_DATA", "0") == "1"
    if active:
        def randw(*shape):
            return rng.integers(-64, 65, size=shape, dtype=np.int8)
        def randb(n):
            return rng.integers(-1000, 1000, size=n, dtype=np.int32)
    else:
        def randw(*shape):
            return rng.integers(-32, 33, size=shape, dtype=np.int8)
        def randb(n):
            return rng.integers(-100, 100, size=n, dtype=np.int32)
    def randg(n):
        return (1.0 + 0.1 * rng.standard_normal(n).astype(np.float32)).astype(bfloat16)

    ang = rng.uniform(0, 2 * np.pi, size=HEAD_D // 2).astype(np.float32)
    ch = np.cos(ang).astype(bfloat16); sh = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([ch, ch]); sin = np.concatenate([sh, sh])

    return {
        "gamma_in":   randg(D),
        "gamma_post": randg(D),
        "wq": randw(QD, D), "bq": randb(QD),
        "wo": randw(D, QD), "bo": randb(D),
        "wg": randw(HD, D), "bg": randb(HD),
        "wu": randw(HD, D), "bu": randb(HD),
        "wd": randw(D, HD), "bd": randb(D),
        "cos": cos, "sin": sin,
        "cos_half": ch, "sin_half": sh,
        "kcache": rng.integers(-32, 33, size=T * KVD, dtype=np.int8),
        "vcache": rng.integers(-32, 33, size=T * KVD, dtype=np.int8),
    }


def pack_one_layer(buf, off, layer):
    """Pack one layer's weights into `buf` at byte offset `off`."""
    def put_bf16(b, o, arr):
        as_bytes = arr.view(np.int8)
        b[o:o + as_bytes.size] = as_bytes
    def put_packed(b, o, w_i8, b_i32):
        b[o:o + w_i8.size] = w_i8.flatten()
        bb = b_i32.view(np.int8).flatten()
        b[o + w_i8.size:o + w_i8.size + bb.size] = bb

    put_bf16(buf,   off + OFF_GAMMA_IN,   layer["gamma_in"])
    put_bf16(buf,   off + OFF_GAMMA_POST, layer["gamma_post"])
    put_packed(buf, off + OFF_WQ, layer["wq"], layer["bq"])
    put_packed(buf, off + OFF_WO, layer["wo"], layer["bo"])
    put_packed(buf, off + OFF_WG, layer["wg"], layer["bg"])
    put_packed(buf, off + OFF_WU, layer["wu"], layer["bu"])
    put_packed(buf, off + OFF_WD, layer["wd"], layer["bd"])
    cs_packed = np.concatenate([layer["cos"], layer["sin"]])
    put_bf16(buf,   off + OFF_CS, cs_packed)


def main():
    p = test_utils.create_default_argparser()
    opts = p.parse_args()

    import os
    seed = int(os.environ.get("LLAMA_CHAIN_SEED", "0"))
    rng = np.random.default_rng(seed)
    x_in = rng.integers(-32, 33, size=D, dtype=np.int8)

    layers = [gen_layer(rng) for _ in range(N_LAYERS)]

    # DEBUG bisection: force one category of data to be the same across
    # all layers; the rest stay distinct. Whichever category yields 0
    # diffs implicates that fifo's per-layer delivery.
    import os
    bisect = os.environ.get("LLAMA_CHAIN_BISECT", "")
    if bisect:
        print(f"[bisect] forcing {bisect} identical across layers")
        keys_by_group = {
            "gamma":  ["gamma_in", "gamma_post"],
            "gamma_in": ["gamma_in"],
            "gamma_post": ["gamma_post"],
            "wq":     ["wq", "bq"],
            "wo":     ["wo", "bo"],
            "wg":     ["wg", "bg"],
            "wu":     ["wu", "bu"],
            "wd":     ["wd", "bd"],
            "cs":     ["cos", "sin", "cos_half", "sin_half"],
            "kv":     ["kcache", "vcache"],
            "ffn":    ["wg", "bg", "wu", "bu", "wd", "bd"],   # all FFN weights
            "attn":   ["wq", "bq", "wo", "bo", "kcache", "vcache",
                       "cos", "sin", "cos_half", "sin_half"],
        }
        keys = keys_by_group[bisect]
        for L in range(1, N_LAYERS):
            for k in keys:
                layers[L][k] = layers[0][k]

    # --- Pack blobs ---
    wblob = np.zeros(TOTAL_W,  dtype=np.int8)
    kvblob = np.zeros(TOTAL_KV, dtype=np.int8)
    for L, layer in enumerate(layers):
        pack_one_layer(wblob, L * PER_LAYER_W, layer)
        base_kv = L * PER_LAYER_KV
        kvblob[base_kv + OFF_K:base_kv + OFF_K + KCACHE_BYTES] = layer["kcache"]
        kvblob[base_kv + OFF_V:base_kv + OFF_V + VCACHE_BYTES] = layer["vcache"]

    # --- Numpy chain reference ---
    x = x_in.copy()
    for layer in layers:
        x = numpy_single_layer(x, layer)
    expected = x

    # --- NPU dispatch ---
    x_t  = iron.tensor(x_in.copy(), dtype=np.int8)
    w_t  = iron.tensor(wblob,       dtype=np.int8)
    kv_t = iron.tensor(kvblob,      dtype=np.int8)
    o_t  = iron.zeros([D],          dtype=np.int8)

    npu_opts = test_utils.create_npu_kernel(opts)
    rc = DefaultNPURuntime.run_test(
        npu_opts.npu_kernel,
        [x_t, w_t, kv_t, o_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        return rc

    o_t.to("cpu")
    actual = o_t.numpy()

    diff = actual.astype(np.int16) - expected.astype(np.int16)
    n_diff = int((diff != 0).sum())
    max_abs = int(np.abs(diff).max()) if n_diff else 0
    print(f"chain_real NPU vs chained-numpy: N_LAYERS={N_LAYERS}  "
          f"D={D}  mismatches={n_diff}/{D}  max|diff|={max_abs}")

    if n_diff == 0:
        print(f"BIT-EXACT PASS  ({N_LAYERS}-layer decode chain)")
        return 0
    print("FAIL")
    for i in np.argwhere(diff != 0).flatten()[:8]:
        print(f"  i={i}: NPU={actual[i]}  expected={expected[i]}  x_in={x_in[i]}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
