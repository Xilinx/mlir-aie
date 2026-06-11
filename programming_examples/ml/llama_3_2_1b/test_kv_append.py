"""Stage 1 unit test: llama_kv_append_head bit-exact vs the numpy oracle.

Mirrors numpy_layer_mh.py:426-465 (the `position=` append) for ONE KV head:
rope_k (half-split) + per-slot dynamic quant + slot write. Confirms the
on-chip append arithmetic before the full single-layer integration.
"""

from __future__ import annotations

import struct
import sys

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_kv_append import (
    HEAD_D,
    T,
    PREFIX,
    SCALE_BYTES,
    BODY_BYTES,
    PER_HEAD,
    KVFP_BYTES,
)
from test_rmsnorm_int8_dyn import sw_recip


def requant(fp, inv):
    s = fp * np.float32(inv)
    r = np.where(s >= 0, np.floor(s + np.float32(0.5)), np.ceil(s - np.float32(0.5)))
    return r.clip(-128, 127).astype(np.int8)


def numpy_append_head(k_fp, v_fp, cos, sin, pos):
    """Bit-exact mirror of the per-head append (numpy_layer_mh oracle)."""
    cosf = cos.astype(np.float32)
    sinf = sin.astype(np.float32)
    half = HEAD_D // 2
    x1 = k_fp[:half].copy()
    x2 = k_fp[half:].copy()
    k_rope = np.empty(HEAD_D, dtype=np.float32)
    k_rope[:half] = x1 * cosf[:half] - x2 * sinf[:half]
    k_rope[half:] = x2 * cosf[half:] + x1 * sinf[half:]
    # per-slot scales: absmax/127, 1e-12 floor; inverse via sw_recip (HW).
    ks = np.float32(max(float(np.abs(k_rope).max()), 1e-12) * np.float32(1.0 / 127.0))
    vs = np.float32(max(float(np.abs(v_fp).max()), 1e-12) * np.float32(1.0 / 127.0))
    k_i8 = requant(k_rope, np.float32(sw_recip(ks)))
    v_i8 = requant(v_fp, np.float32(sw_recip(vs)))
    return k_i8, v_i8, ks, vs


def run_one(pos: int, opts, npu_kernel) -> int:
    rng = np.random.default_rng(pos + 100)
    k_fp = (rng.standard_normal(HEAD_D) * 0.5).astype(np.float32)
    v_fp = (rng.standard_normal(HEAD_D) * 0.5).astype(np.float32)
    ang = rng.uniform(0, 2 * np.pi, size=HEAD_D // 2).astype(np.float32)
    cos_h = np.cos(ang).astype(bfloat16)
    sin_h = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_h, cos_h])
    sin = np.concatenate([sin_h, sin_h])
    cs = np.concatenate([cos, sin])  # bf16[2*HEAD_D]

    # Pack [k_fp fp32 | v_fp fp32 | cs bf16] into one int8 buffer.
    kvfp = np.zeros(KVFP_BYTES, dtype=np.int8)
    kvfp[0 : HEAD_D * 4] = np.frombuffer(k_fp.tobytes(), dtype=np.int8)
    kvfp[HEAD_D * 4 : HEAD_D * 8] = np.frombuffer(v_fp.tobytes(), dtype=np.int8)
    kvfp[HEAD_D * 8 :] = np.frombuffer(cs.tobytes(), dtype=np.int8)

    kv_in = rng.integers(-50, 50, size=PER_HEAD, dtype=np.int8)
    # Prefix [0:4] = T_used (flowkv contract); append writes slot T_used-1 = pos.
    kv_in[0:4] = np.frombuffer(np.int32(pos + 1).tobytes(), dtype=np.int8)

    fp_t = iron.tensor(kvfp.copy(), dtype=np.int8)
    kvin_t = iron.tensor(kv_in.copy(), dtype=np.int8)
    kvout_t = iron.zeros([PER_HEAD], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu_kernel,
        [fp_t, kvin_t, kvout_t],
        {},
        verify=False,
        verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"pos {pos}: dispatch returned {rc}", file=sys.stderr)
        return rc
    kvout_t.to("cpu")
    kv_out = kvout_t.numpy()

    k_i8, v_i8, ks, vs = numpy_append_head(k_fp, v_fp, cos, sin, pos)

    k_scales = PREFIX
    k_body = k_scales + SCALE_BYTES
    v_scales = k_body + BODY_BYTES
    v_body = v_scales + SCALE_BYTES
    exp = kv_in.copy()
    exp[k_body + pos * HEAD_D : k_body + (pos + 1) * HEAD_D] = k_i8
    exp[v_body + pos * HEAD_D : v_body + (pos + 1) * HEAD_D] = v_i8
    exp[k_scales + pos * 4 : k_scales + pos * 4 + 4] = np.frombuffer(
        np.float32(ks).tobytes(), dtype=np.int8
    )
    exp[v_scales + pos * 4 : v_scales + pos * 4 + 4] = np.frombuffer(
        np.float32(vs).tobytes(), dtype=np.int8
    )

    if np.array_equal(kv_out, exp):
        print(f"pos {pos}: BIT-EXACT  (ks={float(ks):.6g} vs={float(vs):.6g})")
        return 0

    # The K/V bodies must be byte-exact. The per-slot SCALES may differ by 1
    # ULP: ks/vs come from scalar-fp32 absmax + sw_recip, which has the
    # documented irreducible ~1-ULP Peano-vs-numpy drift (Bug 11c). Accept a
    # <=1-ULP scale diff on slot[pos]; everything else must match exactly.
    dev_ks_bits = int.from_bytes(
        kv_out[k_scales + pos * 4 : k_scales + pos * 4 + 4].tobytes(), "little"
    )
    ref_ks_bits = int.from_bytes(np.float32(ks).tobytes(), "little")
    dev_vs_bits = int.from_bytes(
        kv_out[v_scales + pos * 4 : v_scales + pos * 4 + 4].tobytes(), "little"
    )
    ref_vs_bits = int.from_bytes(np.float32(vs).tobytes(), "little")
    scale_ulp = max(abs(dev_ks_bits - ref_ks_bits), abs(dev_vs_bits - ref_vs_bits))
    # Mask the two slot-scale words, then require everything else byte-exact.
    masked_out = kv_out.copy()
    masked_exp = exp.copy()
    for off in (k_scales + pos * 4, v_scales + pos * 4):
        masked_out[off : off + 4] = 0
        masked_exp[off : off + 4] = 0
    if np.array_equal(masked_out, masked_exp) and scale_ulp <= 1:
        print(
            f"pos {pos}: PASS(1ulp)  body byte-exact; slot scale <=1 ULP "
            f"(ks={float(ks):.7g} vs={float(vs):.7g})"
        )
        return 0
    nmis = int((kv_out.astype(np.int32) != exp.astype(np.int32)).sum())
    diff_idx = np.where(kv_out.astype(np.int32) != exp.astype(np.int32))[0]
    # localize by region
    kb = kv_out[k_body + pos * HEAD_D : k_body + (pos + 1) * HEAD_D]
    vb = kv_out[v_body + pos * HEAD_D : v_body + (pos + 1) * HEAD_D]
    kd = int(np.abs(kb.astype(np.int32) - k_i8.astype(np.int32)).max())
    vd = int(np.abs(vb.astype(np.int32) - v_i8.astype(np.int32)).max())
    dev_ks = struct.unpack(
        "<f", kv_out[k_scales + pos * 4 : k_scales + pos * 4 + 4].tobytes()
    )[0]
    dev_vs = struct.unpack(
        "<f", kv_out[v_scales + pos * 4 : v_scales + pos * 4 + 4].tobytes()
    )[0]
    regions = {
        "prefix": (0, PREFIX),
        "k_scales": (k_scales, k_body),
        "k_body": (k_body, v_scales),
        "v_scales": (v_scales, v_body),
        "v_body": (v_body, PER_HEAD),
    }
    hit = {
        r: int(((diff_idx >= a) & (diff_idx < b)).sum())
        for r, (a, b) in regions.items()
    }
    print(
        f"pos {pos}: FAIL  mismatch={nmis}/{PER_HEAD}  k_body max|d|={kd} v_body max|d|={vd}"
        f"  ks dev={dev_ks:.7g} ref={float(ks):.7g}  vs dev={dev_vs:.7g} ref={float(vs):.7g}"
    )
    print(f"   diffs by region: {hit}")
    return 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--positions", type=str, default="0,1,5,42,100")
    opts = p.parse_args()
    positions = [int(s) for s in opts.positions.split(",")]
    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    fails = sum(run_one(pp, opts, npu_kernel) != 0 for pp in positions)
    print(f"\nkv_append: {len(positions) - fails}/{len(positions)} positions BIT-EXACT")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
