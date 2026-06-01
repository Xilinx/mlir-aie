"""Phase 7a: bit-exact test for the single-layer multi-head GQA xclbin.

For each seed:
  1. gen_layer_mh -> random per-layer fixture (full Q_DIM, KV_DIM shapes).
  2. numpy_layer_mh_forward -> golden y_int8 + per-Q-head calibrated scales.
  3. Pack wblob (with the new 448 B wq-mh prefix and the af_scales region)
     + kvblob (8 KV heads, 4 B header each).
  4. Dispatch xclbin, compare bit-exact.

Run:
  make layer_mh
  python test_layer_mh.py -x build/final_layer_mh_T128.xclbin \
                          -i build/insts_layer_mh_T128.bin \
                          -k MLIR_AIE --seeds 0,1,7,42
"""

from __future__ import annotations

import sys

import numpy as np

import aie.iron as iron
import aie.utils.test as test_utils
from aie.utils import DefaultNPURuntime

from aie2_layer_mh import (
    D, HD, HEAD_D, N_HEADS_Q, N_HEADS_KV, REP, QD, T, N_TILE,
    WQ_SLOT, WO_SLOT, WG_SLOT, WU_SLOT, WD_SLOT, WQ_PREFIX,
    N_TILES_Q, N_TILES_O, N_TILES_G, N_TILES_U, N_TILES_D,
    OFF_GAMMA_IN, OFF_WQ, OFF_CS, OFF_WO, OFF_AF_SCALES,
    OFF_GAMMA_POST, OFF_WG, OFF_WU, OFF_WD,
    GAMMA_BYTES, CS_BYTES, AF_SCALES_BYTES,
    KCACHE_BYTES, VCACHE_BYTES, KCACHE_PADDED, VCACHE_PADDED, KV_HEADER,
    PER_KV_HEAD_BYTES, kv_off_k, kv_off_v, KV_BYTES, WEIGHTS_BYTES,
    ACT_SCALE, INV_ACT_SCALE, SILU_GATE_SCALE,
)
from numpy_layer_mh import gen_layer_mh, numpy_layer_mh_forward
from test_ffn_half import pack_perchan_slots, fp32_bytes
from test_flowkv import EXP_QUANT_SCALE
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut


def wq_mh_prefix(act_scale: float, q_inv_outs: np.ndarray,
                 q_out_scales: np.ndarray,
                 sv_inv_out_scales: np.ndarray) -> bytes:
    """448-byte prefix for q_proj-mh slots.
      [0..4]   act_scale fp32
      [4..8]   spare
      [8..136] 32 q_inv_outs fp32
      [136..392] 32 * 8 B [q_out_scale, sv_inv_out_scale] pairs (interleaved)
      [392..448] padding
    """
    assert q_inv_outs.shape == (N_HEADS_Q,)
    assert q_out_scales.shape == (N_HEADS_Q,)
    assert sv_inv_out_scales.shape == (N_HEADS_Q,)
    buf = bytearray(WQ_PREFIX)
    buf[0:4] = np.float32(act_scale).tobytes()
    # spare [4..8] left zero
    buf[8:8 + 4 * N_HEADS_Q] = q_inv_outs.astype(np.float32).tobytes()
    # interleaved pairs
    pairs = np.empty(2 * N_HEADS_Q, dtype=np.float32)
    pairs[0::2] = q_out_scales.astype(np.float32)
    pairs[1::2] = sv_inv_out_scales.astype(np.float32)
    buf[136:136 + 8 * N_HEADS_Q] = pairs.tobytes()
    return bytes(buf)


def af_scales_bytes(sv_out_scales: np.ndarray, o_inv_act_scale: float) -> bytes:
    """192-byte af_concat scales buffer."""
    assert sv_out_scales.shape == (N_HEADS_Q,)
    buf = bytearray(AF_SCALES_BYTES)
    buf[0:4 * N_HEADS_Q] = sv_out_scales.astype(np.float32).tobytes()
    buf[128:132] = np.float32(o_inv_act_scale).tobytes()
    return bytes(buf)


def pack_blobs(layer):
    sc = layer["scales"]
    wblob = np.zeros(WEIGHTS_BYTES, dtype=np.int8)

    # gamma_in
    wblob[OFF_GAMMA_IN:OFF_GAMMA_IN + GAMMA_BYTES] = layer["gamma_in"].view(np.int8)

    # wq-mh: 448 B per-slot prefix
    wq_pre = wq_mh_prefix(ACT_SCALE, sc["q_inv_outs"],
                          sc["q_out_scales"], sc["sv_inv_out_scales"])
    wq_packed = pack_perchan_slots(layer["wq_i8"], layer["wq_sc"], layer["bq"],
                                   N_TILE, prefix_bytes=wq_pre)
    assert wq_packed.size == N_TILES_Q * WQ_SLOT, \
        f"wq packed {wq_packed.size} vs expected {N_TILES_Q * WQ_SLOT}"
    wblob[OFF_WQ:OFF_WQ + wq_packed.size] = wq_packed

    # cs
    cs_packed = np.concatenate([layer["cos"], layer["sin"]])
    wblob[OFF_CS:OFF_CS + CS_BYTES] = cs_packed.view(np.int8)

    # wo-mh: standard 64 B prefix (o_act_scale, INV_ACT_SCALE)
    wo_pre = fp32_bytes(float(sc["o_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
    wo_packed = pack_perchan_slots(layer["wo_i8"], layer["wo_sc"], layer["bo"],
                                   N_TILE, prefix_bytes=wo_pre)
    assert wo_packed.size == N_TILES_O * WO_SLOT
    wblob[OFF_WO:OFF_WO + wo_packed.size] = wo_packed

    # af_scales
    af_sc = af_scales_bytes(sc["sv_out_scales"], float(sc["o_inv_act_scale"]))
    wblob[OFF_AF_SCALES:OFF_AF_SCALES + AF_SCALES_BYTES] = \
        np.frombuffer(af_sc, dtype=np.int8)

    # gamma_post
    wblob[OFF_GAMMA_POST:OFF_GAMMA_POST + GAMMA_BYTES] = \
        layer["gamma_post"].view(np.int8)

    # wg (no prefix, closure-baked scales)
    wg_packed = pack_perchan_slots(layer["wg_i8"], layer["wg_sc"], layer["bg"],
                                   N_TILE, prefix_bytes=b"")
    assert wg_packed.size == N_TILES_G * WG_SLOT
    wblob[OFF_WG:OFF_WG + wg_packed.size] = wg_packed

    # wu: 64 B prefix (ACT_SCALE, up_inv_out, silu_up_scale, silu_inv_out_scale)
    wu_pre = (fp32_bytes(ACT_SCALE, float(sc["up_inv_out"]),
                         float(sc["silu_up_scale"]),
                         float(sc["silu_inv_out_scale"]))
              + b"\x00" * 48)
    wu_packed = pack_perchan_slots(layer["wu_i8"], layer["wu_sc"], layer["bu"],
                                   N_TILE, prefix_bytes=wu_pre)
    assert wu_packed.size == N_TILES_U * WU_SLOT
    wblob[OFF_WU:OFF_WU + wu_packed.size] = wu_packed

    # wd: 64 B prefix (down_act_scale, INV_ACT_SCALE)
    wd_pre = fp32_bytes(float(sc["down_act_scale"]), INV_ACT_SCALE) + b"\x00" * 56
    wd_packed = pack_perchan_slots(layer["wd_i8"], layer["wd_sc"], layer["bd"],
                                   N_TILE, prefix_bytes=wd_pre)
    assert wd_packed.size == N_TILES_D * WD_SLOT
    wblob[OFF_WD:OFF_WD + wd_packed.size] = wd_packed

    # kvblob: 8 KV heads sequentially.
    kvblob = np.zeros(KV_BYTES, dtype=np.int8)
    for h in range(N_HEADS_KV):
        k_off = kv_off_k(h)
        v_off = kv_off_v(h)
        # 4 B k_scale header
        kvblob[k_off:k_off + 4] = np.frombuffer(
            fp32_bytes(float(layer["k_scales"][h])), dtype=np.int8)
        kvblob[k_off + KV_HEADER:k_off + KV_HEADER + KCACHE_BYTES] = \
            layer["kcaches"][h]
        # 4 B v_scale header
        kvblob[v_off:v_off + 4] = np.frombuffer(
            fp32_bytes(float(layer["v_scales"][h])), dtype=np.int8)
        kvblob[v_off + KV_HEADER:v_off + KV_HEADER + VCACHE_BYTES] = \
            layer["vcaches"][h]

    return wblob, kvblob


def run_one_seed(seed: int, opts, lut_exp, lut_silu, npu_kernel) -> int:
    rng = np.random.default_rng(seed)
    layer = gen_layer_mh(rng)
    layer["lut_exp"]  = lut_exp
    layer["lut_silu"] = lut_silu

    x = rng.integers(-32, 33, size=D, dtype=np.int8)
    y_ref, scales = numpy_layer_mh_forward(x, layer)
    layer["scales"] = scales

    wblob, kvblob = pack_blobs(layer)

    x_t  = iron.tensor(x,     dtype=np.int8)
    w_t  = iron.tensor(wblob, dtype=np.int8)
    kv_t = iron.tensor(kvblob, dtype=np.int8)
    o_t  = iron.zeros([D], dtype=np.int8)
    rc = DefaultNPURuntime.run_test(
        npu_kernel, [x_t, w_t, kv_t, o_t],
        {}, verify=False, verbosity=opts.verbosity,
    )
    if rc != 0:
        print(f"seed {seed}: NPU dispatch returned {rc}", file=sys.stderr)
        return rc
    o_t.to("cpu")
    y_dev = o_t.numpy()

    diff = (y_dev.astype(np.int32) - y_ref.astype(np.int32))
    n_mismatch = int((diff != 0).sum())
    if n_mismatch == 0:
        print(f"seed {seed}: BIT-EXACT  (sat={int((y_ref==127).sum()+(y_ref==-128).sum())}/{D})")
        return 0
    # diagnostic
    max_abs = int(np.abs(diff).max())
    idx = np.where(diff != 0)[0][:10]
    print(f"seed {seed}: FAIL  mismatch={n_mismatch}/{D}  max|d|={max_abs}")
    for i in idx:
        print(f"  i={i}: dev={int(y_dev[i])} ref={int(y_ref[i])}")
    return 1


def main():
    p = test_utils.create_default_argparser()
    p.add_argument("--seeds", type=str, default="0,1,7,42",
                   help="comma-separated seed list")
    opts = p.parse_args()
    seeds = [int(s) for s in opts.seeds.split(",")]

    npu_kernel = test_utils.create_npu_kernel(opts).npu_kernel
    lut_exp  = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)

    fails = 0
    for s in seeds:
        fails += (run_one_seed(s, opts, lut_exp, lut_silu, npu_kernel) != 0)
    print(f"\nlayer_mh: {len(seeds) - fails}/{len(seeds)} seeds BIT-EXACT")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
