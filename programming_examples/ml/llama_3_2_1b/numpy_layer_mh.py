"""Phase 7a: multi-head GQA reference + per-Q-head dynamic-scale
calibration for the single-layer multi-head xclbin (aie2_layer_mh.py).

Mirrors numpy_layer_forward in test_chain_dynscale.py but at full
Llama 3.2 1B attn shapes: 32 Q heads, 8 KV heads (GQA, REP=4 Q per
KV), HEAD_DIM=64, Q_DIM=2048, KV_DIM=512.

Key changes vs single-head:
  - wq, wo are full (Q_DIM, D) / (D, Q_DIM)
  - q_out_scale, sv_inv_out_scale are PER Q HEAD (length 32)
  - k_scale, v_scale are PER KV HEAD (length 8)
  - sv outputs from each Q head are concatenated, dequanted with each
    head's sv_out_scale, then globally requanted to int8 with a single
    o_act_scale before o_proj. This matches the on-device topology
    where 8 attn workers emit per-head int8 sv chunks and an "af-concat"
    worker stitches + requantizes for the (single-act_scale) o_proj.
  - FFN stays unchanged (per-token scales, single act for gate/up/down).
"""

from __future__ import annotations

import numpy as np
from ml_dtypes import bfloat16

from aie2_layer_d2048 import (
    ACT_SCALE,
    INV_ACT_SCALE,
    SILU_GATE_SCALE,
    GATE_INV_OUT_SCALE,
)
from test_rmsnorm_int8 import numpy_rmsnorm_int8
from test_rmsnorm_int8_dyn import numpy_rmsnorm_int8_dyn, sw_recip
from test_rope_int8 import numpy_rope_dyn
from test_flowkv import numpy_attention, EXP_QUANT_SCALE
from test_silu_mul_int8 import numpy_silu_mul
from test_attn_half import compute_sv_fp
from test_ffn_half import numpy_gemm_perchan, i8_add_wrap
from gen_exp_lut import exp_lut
from gen_silu_lut import silu_lut
from gen_llama_data import quant_int8_perchan_absmax

# --- Multi-head GQA shape constants ---
D = 2048
N_HEADS_Q = 32
N_HEADS_KV = 8
HEAD_DIM = 64
REP = N_HEADS_Q // N_HEADS_KV  # 4
Q_DIM = N_HEADS_Q * HEAD_DIM  # 2048
KV_DIM = N_HEADS_KV * HEAD_DIM  # 512
HD = 8192
T = 128


def requant(fp, inv):
    s = fp * np.float32(inv)
    r = np.where(s >= 0, np.floor(s + np.float32(0.5)), np.ceil(s - np.float32(0.5)))
    return r.clip(-128, 127).astype(np.int8)


def _dyn_scale(fp):
    """Per-token dynamic scale, bit-identical to the device rescale-add kernel:
    absmax (fp32, 1e-12 floor) * (1.0f/127.0f). Using float64 /127.0 diverges
    by ~1 ULP and flips knife-edge requant roundings, so keep everything fp32
    (matches the dyn-rmsnorm HW mirror in test_rmsnorm_int8_dyn)."""
    absmax = np.float32(max(float(np.abs(fp).max()), 1e-12))
    return np.float32(absmax * np.float32(1.0 / 127.0))


def attention_perslot(
    q_i8, k_i8, v_i8, head_dim, t, q_scale, k_slot_scales, v_slot_scales, inv_out_scale
):
    """Per-cache-slot-scale attention (the correct multi-token decode path).

    Each cached position is dequantized by ITS OWN k/v scale, fixing the
    per-head-scalar bug where the whole cache was dequantized with the latest
    token's scale. fp32 softmax (no exp-LUT) — this defines the NEW on-device
    target (the flowkv kernel + KV fifo format must carry a per-slot scale).
    q_i8: (head_dim,) int8; k_i8/v_i8: (t*head_dim,) int8; *_slot_scales: (t,).
    Returns int8[head_dim] at inv_out_scale.
    """
    qf = q_i8.astype(np.float32) * np.float32(q_scale)
    k_mat = k_i8.astype(np.float32).reshape(t, head_dim) * k_slot_scales[:t, None]
    v_mat = v_i8.astype(np.float32).reshape(t, head_dim) * v_slot_scales[:t, None]
    scale = np.float32(1.0) / np.sqrt(np.float32(head_dim))
    scores = (k_mat @ qf) * scale  # (t,)
    scores = scores - scores.max()
    e = np.exp(scores)
    probs = e / e.sum()
    sv = probs @ v_mat  # (head_dim,)
    return requant(sv, np.float32(inv_out_scale)), sv.astype(np.float32)


def numpy_attention_perslot_lut(
    q_i8,
    k_i8,
    v_i8,
    head_dim,
    t,
    q_scale,
    k_slot_scales,
    v_slot_scales,
    inv_out_scale,
    lut,
):
    """Bit-exact mirror of the device flowkv_mh per-slot kernel.

    Per-cache-slot k/v scales (fixes the per-head-scalar bug) AND the exp-LUT
    softmax (matches the device kernel, unlike the fp32-softmax attention_perslot
    which is the QUALITY golden). Op order replicates llama_flowkv_mh exactly:
      score[i] = dot_i32 * (q_scale*0.125 * k_slot[i])     (strict fp32)
      sv[j]   += (lut_exp*inv_sum) * v_slot[i] * v[i,j]    (strict fp32)
    q_i8: (head_dim,); k_i8/v_i8: (t*head_dim,); *_slot_scales: (>=t,).
    """
    from test_flowkv import quant_shifted, round_to_i8, sw_recip

    qs = np.float32(q_scale)
    inv_sqrt = np.float32(0.125)

    k_mat = k_i8.astype(np.int32).reshape(t, head_dim)
    dots = (k_mat @ q_i8.astype(np.int32)).astype(np.int32)  # (t,)
    ksl = np.asarray(k_slot_scales, dtype=np.float32)[:t]
    # Scale order (q_scale * k_slot[i]) * 0.125 mirrors the kernel and the
    # bit-exact per-head predecessor; (q*0.125)*k diverges ~1 ULP and flips
    # LUT-quant boundaries.
    qk = ((qs * ksl).astype(np.float32) * inv_sqrt).astype(np.float32)
    scores = (dots.astype(np.float32) * qk).astype(np.float32)

    max_s = np.float32(scores.max())
    shifted = (scores - max_s).astype(np.float32)
    q = quant_shifted(shifted)
    exp_v = lut[q + 128].astype(np.float32)

    sum_e = np.float32(0.0)
    for i in range(t):
        sum_e = (sum_e + exp_v[i]).astype(np.float32)
    inv_sum = sw_recip(sum_e)
    probs = (exp_v * inv_sum).astype(np.float32)

    # sv: kernel accumulates acc[j] += p[i] * v_slot[i] * v[i,j], strictly
    # left-to-right per output channel.
    vsl = np.asarray(v_slot_scales, dtype=np.float32)[:t]
    v_mat = v_i8.astype(np.int32).reshape(t, head_dim).astype(np.float32)
    acc = np.zeros(head_dim, dtype=np.float32)
    for i in range(t):
        acc = (acc + (probs[i] * vsl[i]).astype(np.float32) * v_mat[i]).astype(
            np.float32
        )
    out_f = (acc * np.float32(inv_out_scale)).astype(np.float32)
    return round_to_i8(out_f), acc.astype(np.float32)


def gen_real_layer_mh(L: int, data_dir, rng: np.random.Generator) -> dict:
    """Phase 7a real-weight loader: full Q_DIM/KV_DIM Llama 3.2 1B weights
    sliced into the 8 KV-head layout the mh xclbin expects. KV cache stays
    random per-seed (chain doesn't model real k_proj/v_proj yet).
    Mirrors test_chain_dynscale.gen_real_layer but without head-0 slicing.
    """
    from pathlib import Path

    ld = Path(data_dir) / f"layer_{L:02d}"

    def _i8(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=np.int8).reshape(shape).copy()
        )

    def _f32(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=np.float32)
            .reshape(shape)
            .copy()
        )

    def _bf16(name, shape):
        return (
            np.frombuffer((ld / name).read_bytes(), dtype=bfloat16)
            .reshape(shape)
            .copy()
        )

    wq_i8 = _i8("wq.i8.bin", (Q_DIM, D))
    wq_sc = _f32("wq.scales.f32.bin", (Q_DIM,))
    bq = np.zeros(Q_DIM, np.int32)

    # k_proj, v_proj: (KV_DIM, D) -- Phase 8a real K/V projection.
    wk_i8 = _i8("wk.i8.bin", (KV_DIM, D))
    wk_sc = _f32("wk.scales.f32.bin", (KV_DIM,))
    wv_i8 = _i8("wv.i8.bin", (KV_DIM, D))
    wv_sc = _f32("wv.scales.f32.bin", (KV_DIM,))

    wo_i8 = _i8("wo.i8.bin", (D, Q_DIM))
    wo_sc = _f32("wo.scales.f32.bin", (D,))
    bo = np.zeros(D, np.int32)

    wg_i8 = _i8("wg.i8.bin", (HD, D))
    wg_sc = _f32("wg.scales.f32.bin", (HD,))
    bg = np.zeros(HD, np.int32)
    wu_i8 = _i8("wu.i8.bin", (HD, D))
    wu_sc = _f32("wu.scales.f32.bin", (HD,))
    bu = np.zeros(HD, np.int32)
    wd_i8 = _i8("wd.i8.bin", (D, HD))
    wd_sc = _f32("wd.scales.f32.bin", (D,))
    bd = np.zeros(D, np.int32)

    gamma_in = _bf16("gamma_in.bf16.bin", (D,))
    gamma_post = _bf16("gamma_post.bf16.bin", (D,))

    half = HEAD_DIM // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # Phase 8a: empty caches; the host harness fills slots 0..t_cur as
    # tokens are processed. k/v_scales start at a tiny epsilon and get
    # overwritten per token by numpy_layer_mh_forward(position=...).
    kcaches = [np.zeros(T * HEAD_DIM, dtype=np.int8) for _ in range(N_HEADS_KV)]
    vcaches = [np.zeros(T * HEAD_DIM, dtype=np.int8) for _ in range(N_HEADS_KV)]
    k_scales = np.full(N_HEADS_KV, 1e-6, dtype=np.float32)
    v_scales = np.full(N_HEADS_KV, 1e-6, dtype=np.float32)

    return dict(
        wq_i8=wq_i8,
        wq_sc=wq_sc,
        bq=bq,
        wk_i8=wk_i8,
        wk_sc=wk_sc,
        wv_i8=wv_i8,
        wv_sc=wv_sc,
        wo_i8=wo_i8,
        wo_sc=wo_sc,
        bo=bo,
        wg_i8=wg_i8,
        wg_sc=wg_sc,
        bg=bg,
        wu_i8=wu_i8,
        wu_sc=wu_sc,
        bu=bu,
        wd_i8=wd_i8,
        wd_sc=wd_sc,
        bd=bd,
        cos=cos,
        sin=sin,
        kcaches=kcaches,
        vcaches=vcaches,
        k_scales=k_scales,
        v_scales=v_scales,
        gamma_in=gamma_in,
        gamma_post=gamma_post,
    )


def gen_layer_mh(rng: np.random.Generator) -> dict:
    """Random per-layer fixture at multi-head GQA shapes."""

    def random_w(out_dim, in_dim):
        base = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05
        row_scale = rng.uniform(0.1, 1.0, size=out_dim).astype(np.float32)
        return base * row_scale[:, None]

    wq_fp = random_w(Q_DIM, D)
    bq = np.zeros(Q_DIM, np.int32)
    wo_fp = random_w(D, Q_DIM)
    bo = np.zeros(D, np.int32)
    wg_fp = random_w(HD, D)
    bg = np.zeros(HD, np.int32)
    wu_fp = random_w(HD, D)
    bu = np.zeros(HD, np.int32)
    wd_fp = random_w(D, HD)
    bd = np.zeros(D, np.int32)
    wq_i8, wq_sc = quant_int8_perchan_absmax(wq_fp)
    wo_i8, wo_sc = quant_int8_perchan_absmax(wo_fp)
    wg_i8, wg_sc = quant_int8_perchan_absmax(wg_fp)
    wu_i8, wu_sc = quant_int8_perchan_absmax(wu_fp)
    wd_i8, wd_sc = quant_int8_perchan_absmax(wd_fp)

    # Rope: one (cos, sin) pair for the single decode position.
    half = HEAD_DIM // 2
    ang = rng.uniform(0, 2 * np.pi, size=half).astype(np.float32)
    cos_half = np.cos(ang).astype(bfloat16)
    sin_half = np.sin(ang).astype(bfloat16)
    cos = np.concatenate([cos_half, cos_half])
    sin = np.concatenate([sin_half, sin_half])

    # Per-KV-head cache + scale.
    kcaches = [
        rng.integers(-32, 33, size=T * HEAD_DIM, dtype=np.int8)
        for _ in range(N_HEADS_KV)
    ]
    vcaches = [
        rng.integers(-32, 33, size=T * HEAD_DIM, dtype=np.int8)
        for _ in range(N_HEADS_KV)
    ]
    k_scales = np.full(N_HEADS_KV, 0.05, dtype=np.float32)
    v_scales = np.full(N_HEADS_KV, 0.05, dtype=np.float32)

    gamma_in = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(bfloat16)
    gamma_post = (1.0 + 0.1 * rng.standard_normal(D).astype(np.float32)).astype(
        bfloat16
    )

    # Phase 8a: wk/wv drawn LAST so adding them doesn't shift earlier rng
    # consumers (kcaches/vcaches/gammas) -- preserves test_chain_mh BIT-EXACT.
    wk_fp = random_w(KV_DIM, D)
    wv_fp = random_w(KV_DIM, D)
    wk_i8, wk_sc = quant_int8_perchan_absmax(wk_fp)
    wv_i8, wv_sc = quant_int8_perchan_absmax(wv_fp)

    return dict(
        wq_i8=wq_i8,
        wq_sc=wq_sc,
        bq=bq,
        wk_i8=wk_i8,
        wk_sc=wk_sc,
        wv_i8=wv_i8,
        wv_sc=wv_sc,
        wo_i8=wo_i8,
        wo_sc=wo_sc,
        bo=bo,
        wg_i8=wg_i8,
        wg_sc=wg_sc,
        bg=bg,
        wu_i8=wu_i8,
        wu_sc=wu_sc,
        bu=bu,
        wd_i8=wd_i8,
        wd_sc=wd_sc,
        bd=bd,
        cos=cos,
        sin=sin,
        kcaches=kcaches,
        vcaches=vcaches,
        k_scales=k_scales,
        v_scales=v_scales,
        gamma_in=gamma_in,
        gamma_post=gamma_post,
    )


def numpy_layer_mh_forward(
    x,
    layer,
    position=None,
    residual_fp32=False,
    residual_dyn=False,
    attn_perslot=None,
    attn_lut=False,
):
    """Multi-head GQA single-layer forward + per-Q-head scale calibration.

    Returns (x_out, scales) where scales is a dict packing every
    dynamic scale that the on-device design needs in its wblob/kvblob
    prefixes/tails.

    residual_fp32 (proof-of-concept): when True, the residual stream is
    carried in fp32 instead of int8@static-0.05, and each rmsnorm input is
    quantized per-token. This isolates whether the int8 residual format is
    the dominant quality killer (it is — int8@0.05 clips at +-6.35 while the
    residual grows to ~6.7 per-token by layer 15). All OTHER mh quantizations
    (per-Q-head, per-KV-head, silu LUT, af-concat, gate/up/down perchan) are
    unchanged, so a quality recovery here pins the residual as the cause.
    `x` is fp32 in and fp32 out when this flag is set. The int8 path (flag
    False) preserves the device-test bit-exact contract.
    """
    lut_exp = layer["lut_exp"]
    lut_silu = layer["lut_silu"]
    gamma_in = layer["gamma_in"]
    gamma_post = layer["gamma_post"]
    wq_i8 = layer["wq_i8"]
    wq_sc = layer["wq_sc"]
    bq = layer["bq"]
    wo_i8 = layer["wo_i8"]
    wo_sc = layer["wo_sc"]
    bo = layer["bo"]
    wg_i8 = layer["wg_i8"]
    wg_sc = layer["wg_sc"]
    bg = layer["bg"]
    wu_i8 = layer["wu_i8"]
    wu_sc = layer["wu_sc"]
    bu = layer["bu"]
    wd_i8 = layer["wd_i8"]
    wd_sc = layer["wd_sc"]
    bd = layer["bd"]
    cos = layer["cos"]
    sin = layer["sin"]
    kcaches = layer["kcaches"]
    vcaches = layer["vcaches"]
    k_scales = layer["k_scales"]
    v_scales = layer["v_scales"]

    # Residual entering this layer. Three modes:
    #  - int8 static (default): x is int8[D] at the baked ACT_SCALE=0.05 (the
    #    quality-collapsing legacy path; kept for the device bit-exact contract
    #    of the OLD design).
    #  - residual_fp32 (Phase A proof): x is fp32, quantized per-token here.
    #  - residual_dyn (Phase B, device-faithful): x is a tuple (x_i8, x_scale)
    #    carrying a per-token dynamic residual scale; o_proj/down emit dynamic
    #    int8+scale; each add requants to a new per-token residual scale. This
    #    is the on-device target.
    if residual_dyn:
        x_i8, res_scale = x  # int8[D], fp32
        x_i8 = np.asarray(x_i8, dtype=np.int8)
        res_scale = np.float32(res_scale)
        x_resid_fp = x_i8.astype(np.float32) * res_scale  # for fp32 adds below
        rms_in_scale = float(res_scale)
    elif residual_fp32:
        x_resid_fp = x.astype(np.float32)
        res_scale = float(np.maximum(np.abs(x_resid_fp).max(), 1e-12)) / 127.0
        x_i8 = requant(x_resid_fp, np.float32(1.0) / np.float32(res_scale))
        rms_in_scale = res_scale
    else:
        x_i8 = x
        rms_in_scale = ACT_SCALE

    # 1) rmsnorm1 -> h1 (per-token dynamic output scale).
    h1, act_scale1 = numpy_rmsnorm_int8_dyn(x_i8, gamma_in, rms_in_scale)

    # 1b) Phase 8a: real K/V projection + rope_k + per-KV-head requant +
    # cache append at `position`. Only runs when caller passes position;
    # otherwise the test-fixture caches are used as-is (Phase 7 behavior).
    if position is not None:
        wk_i8 = layer["wk_i8"]
        wk_sc = layer["wk_sc"]
        wv_i8 = layer["wv_i8"]
        wv_sc = layer["wv_sc"]
        k_fp = (
            (wk_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
            * np.float32(act_scale1)
            * wk_sc.astype(np.float32)
        )  # (KV_DIM,)
        v_fp = (
            (wv_i8.astype(np.int32) @ h1.astype(np.int32)).astype(np.float32)
            * np.float32(act_scale1)
            * wv_sc.astype(np.float32)
        )
        # Apply rope to k per KV head (Llama half-split convention).
        # cos / sin are full HEAD_DIM (two copies of half), bf16; cast to fp32.
        cosf = np.asarray(cos, dtype=np.float32)
        sinf = np.asarray(sin, dtype=np.float32)
        half = HEAD_DIM // 2
        k_rope = np.empty_like(k_fp)
        for h in range(N_HEADS_KV):
            kh = k_fp[h * HEAD_DIM : (h + 1) * HEAD_DIM]
            # Snapshot both halves as COPIES — x1/x2 must not be views of kh,
            # or writing the first half corrupts x1 before the second half
            # reads it (in-place rotate-half aliasing bug).
            x1 = kh[:half].copy()
            x2 = kh[half:].copy()
            out = np.empty(HEAD_DIM, dtype=np.float32)
            out[:half] = x1 * cosf[:half] - x2 * sinf[:half]
            out[half:] = x2 * cosf[half:] + x1 * sinf[half:]
            k_rope[h * HEAD_DIM : (h + 1) * HEAD_DIM] = out
        # Per-slot scale storage (per position per head). Needed for correct
        # multi-token decode: each cached position must be dequantized by ITS
        # OWN scale, not the latest token's (the per-head-scalar k/v_scales
        # bug). Lazily created; (N_HEADS_KV, T).
        k_scales_slot = layer.get("k_scales_slot")
        if k_scales_slot is None:
            k_scales_slot = np.full((N_HEADS_KV, T), 1e-6, dtype=np.float32)
            v_scales_slot = np.full((N_HEADS_KV, T), 1e-6, dtype=np.float32)
            layer["k_scales_slot"] = k_scales_slot
            layer["v_scales_slot"] = v_scales_slot
        else:
            v_scales_slot = layer["v_scales_slot"]

        # Per-KV-head dynamic quant + cache append at slot `position`.
        for h in range(N_HEADS_KV):
            kh = k_rope[h * HEAD_DIM : (h + 1) * HEAD_DIM]
            vh = v_fp[h * HEAD_DIM : (h + 1) * HEAD_DIM]
            ks = float(np.maximum(np.abs(kh).max(), 1e-12)) / 127.0
            vs = float(np.maximum(np.abs(vh).max(), 1e-12)) / 127.0
            k_scales[h] = np.float32(ks)
            v_scales[h] = np.float32(vs)
            k_scales_slot[h, position] = np.float32(ks)
            v_scales_slot[h, position] = np.float32(vs)
            kcaches[h][position * HEAD_DIM : (position + 1) * HEAD_DIM] = requant(
                kh, np.float32(1.0) / np.float32(ks)
            )
            vcaches[h][position * HEAD_DIM : (position + 1) * HEAD_DIM] = requant(
                vh, np.float32(1.0) / np.float32(vs)
            )

    # 2) q_proj (full Q_DIM=2048)
    fp_q_full = (
        (wq_i8.astype(np.int32) @ h1.astype(np.int32) + bq).astype(np.float32)
        * np.float32(act_scale1)
        * wq_sc.astype(np.float32)
    )

    # 3) Per-Q-head requant. qf shape (Q_DIM,).
    q_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    q_inv_outs = np.zeros(N_HEADS_Q, dtype=np.float32)
    qf = np.zeros(Q_DIM, dtype=np.int8)
    for h in range(N_HEADS_Q):
        slice_h = fp_q_full[h * HEAD_DIM : (h + 1) * HEAD_DIM]
        s = float(np.maximum(np.abs(slice_h).max(), 1e-12)) / 127.0
        q_out_scales[h] = np.float32(s)
        q_inv_outs[h] = np.float32(1.0) / np.float32(s)
        qf[h * HEAD_DIM : (h + 1) * HEAD_DIM] = requant(slice_h, q_inv_outs[h])

    # 4) rope (per-head). numpy_rope_dyn reshapes to (n_heads, head_dim) internally.
    qr = numpy_rope_dyn(qf, cos, sin, N_HEADS_Q, HEAD_DIM)

    # 5) Per-Q-head attention. GQA: kvh = h_q // REP. Each Q head gets
    #    its own sv_out_scale + sv_inv_out_scale. T_used (Phase 8c):
    #    when caller passes position, only the first position+1 cache
    #    slots are considered (causal mask). Otherwise attend over all
    #    T slots (Phase 7 / 8a behavior, preserves BIT-EXACT for
    #    random-fixture tests).
    t_used = (position + 1) if position is not None else T
    # attn_perslot selects the per-cache-slot-scale fp32-softmax attention
    # (the KV-fix path). It defaults to follow residual_dyn so the quality
    # bench gets both fixes, but is decoupled so the device single-layer test
    # (whose flowkv_mh is still the legacy per-head-scalar exp-LUT kernel) can
    # validate the residual fix in isolation with attn_perslot=False.
    if attn_perslot is None:
        attn_perslot = residual_dyn
    sv_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    sv_inv_out_scales = np.zeros(N_HEADS_Q, dtype=np.float32)
    sv_i8_per_head = []
    for h_q in range(N_HEADS_Q):
        kvh = h_q // REP
        q_h = qr[h_q * HEAD_DIM : (h_q + 1) * HEAD_DIM]
        k_slice = kcaches[kvh][: t_used * HEAD_DIM]
        v_slice = vcaches[kvh][: t_used * HEAD_DIM]
        if attn_perslot or attn_lut:
            # Per-slot-scale attention: each cached position dequantized by
            # its own k/v scale (fixes the per-head-scalar KV bug).
            # attn_lut=True is the DEVICE-FAITHFUL path: exp-LUT softmax
            # (bit-exact mirror of flowkv_mh). attn_lut=False keeps the fp32
            # np.exp softmax (the quality golden). Both use per-slot scales.
            ksl = layer["k_scales_slot"][kvh]
            vsl = layer["v_scales_slot"][kvh]
            attn_fn = (
                (lambda *a: numpy_attention_perslot_lut(*a, lut_exp))
                if attn_lut
                else attention_perslot
            )
            # First pass to derive this head's sv_out_scale.
            _, sv_fp_h = attn_fn(
                q_h,
                k_slice,
                v_slice,
                HEAD_DIM,
                t_used,
                float(q_out_scales[h_q]),
                ksl,
                vsl,
                1.0,
            )
            s = float(np.maximum(np.abs(sv_fp_h).max(), 1e-12)) / 127.0
            sv_out_scales[h_q] = np.float32(s)
            sv_inv_out_scales[h_q] = np.float32(1.0) / np.float32(s)
            sv_i8, _ = attn_fn(
                q_h,
                k_slice,
                v_slice,
                HEAD_DIM,
                t_used,
                float(q_out_scales[h_q]),
                ksl,
                vsl,
                float(sv_inv_out_scales[h_q]),
            )
            sv_i8_per_head.append(sv_i8)
            continue
        sv_fp_h = compute_sv_fp(
            q_h,
            k_slice,
            v_slice,
            HEAD_DIM,
            t_used,
            float(q_out_scales[h_q]),
            float(k_scales[kvh]),
            float(v_scales[kvh]),
            lut_exp,
        )
        s = float(np.maximum(np.abs(sv_fp_h).max(), 1e-12)) / 127.0
        sv_out_scales[h_q] = np.float32(s)
        sv_inv_out_scales[h_q] = np.float32(1.0) / np.float32(s)
        sv_i8 = numpy_attention(
            q_h,
            k_slice,
            v_slice,
            HEAD_DIM,
            t_used,
            float(q_out_scales[h_q]),
            float(k_scales[kvh]),
            float(v_scales[kvh]),
            float(sv_inv_out_scales[h_q]),
            lut_exp,
        )
        sv_i8_per_head.append(sv_i8)

    # 6) af-concat: per-head dequant (using each head's sv_out_scale),
    #    concat to Q_DIM fp32, then global requant with o_act_scale.
    af_fp = np.concatenate(
        [
            sv_i8_per_head[h].astype(np.float32) * np.float32(sv_out_scales[h])
            for h in range(N_HEADS_Q)
        ]
    )
    o_act_scale = float(np.maximum(np.abs(af_fp).max(), 1e-12)) / 127.0
    o_inv_act_scale = float(np.float32(1.0) / np.float32(o_act_scale))
    af = requant(af_fp, o_inv_act_scale)

    _cap = layer.get("_cap")
    if _cap is not None:
        _cap["h1_fp"] = h1.astype(np.float32) * np.float32(act_scale1)
        _cap["af_fp"] = af_fp.copy()

    # 7) o_proj. op_fp = (acc * o_act_scale * wo_sc) is the true fp32 o_proj
    # output (o_act_scale is the per-token af input scale, already dynamic).
    op_acc = wo_i8.astype(np.int32) @ af.astype(np.int32) + bo
    op_fp = (
        op_acc.astype(np.float32) * np.float32(o_act_scale) * wo_sc.astype(np.float32)
    )
    if residual_dyn:
        # Rescale-add: fp32 sum of (residual + o_proj), requant to a new
        # per-token residual scale. (Device: o_proj emits dynamic int8+scale,
        # rescale-add kernel reconstructs both and requants.)
        x1_fp = x_resid_fp + op_fp
        # Scale computed EXACTLY as the device rescale-add kernel:
        # absmax (fp32, 1e-12 floor) * (1.0f/127.0f), inverse via sw_recip.
        # The /127.0-in-float64 form diverges by ~1 ULP and flips dozens of
        # knife-edge requant roundings (matches the dyn-rmsnorm HW mirror).
        x1_scale = _dyn_scale(x1_fp)
        x1_i8 = requant(x1_fp, np.float32(sw_recip(x1_scale)))
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1_i8, gamma_post, float(x1_scale))
    elif residual_fp32:
        x1 = x_resid_fp + op_fp  # fp32 residual
        x1_scale = float(np.maximum(np.abs(x1).max(), 1e-12)) / 127.0
        x1_i8 = requant(x1, np.float32(1.0) / np.float32(x1_scale))
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1_i8, gamma_post, x1_scale)
    else:
        op = numpy_gemm_perchan(af, o_act_scale, wo_i8, wo_sc, bo, INV_ACT_SCALE)
        x1 = i8_add_wrap(op, x)
        h2, act_scale2 = numpy_rmsnorm_int8_dyn(x1, gamma_post, ACT_SCALE)

    if _cap is not None:
        _cap["x1"] = x1_fp.copy() if residual_dyn else np.asarray(x1, np.float32).copy()

    # 9) gate (closure-baked GATE_INV_OUT_SCALE = 1/SILU_GATE_SCALE)
    fp_gate = (
        (wg_i8.astype(np.int32) @ h2.astype(np.int32) + bg).astype(np.float32)
        * np.float32(act_scale2)
        * wg_sc.astype(np.float32)
    )
    gf = requant(fp_gate, GATE_INV_OUT_SCALE)

    # 10) up (dynamic out scale)
    fp_up = (
        (wu_i8.astype(np.int32) @ h2.astype(np.int32) + bu).astype(np.float32)
        * np.float32(act_scale2)
        * wu_sc.astype(np.float32)
    )
    up_out_scale = float(np.maximum(np.abs(fp_up).max(), 1e-12)) / 127.0
    up_inv_out = float(np.float32(1.0) / np.float32(up_out_scale))
    uf = requant(fp_up, up_inv_out)

    # 11) silu_mul
    silu_up_scale = up_out_scale
    s_gate_fp = lut_silu[gf.astype(np.int32) + 128].astype(np.float32)
    s_up_fp = uf.astype(np.float32) * np.float32(silu_up_scale)
    sf_fp = s_gate_fp * s_up_fp
    silu_out_scale = float(np.maximum(np.abs(sf_fp).max(), 1e-12)) / 127.0
    silu_inv_out_scale = float(np.float32(1.0) / np.float32(silu_out_scale))
    sf = numpy_silu_mul(gf, uf, lut_silu, silu_up_scale, silu_inv_out_scale)

    if _cap is not None:
        _cap["h2_fp"] = h2.astype(np.float32) * np.float32(act_scale2)
        _cap["gate_fp"] = fp_gate.copy()
        _cap["up_fp"] = fp_up.copy()
        _cap["sf_fp"] = sf_fp.copy()

    # 12) down + 13) residual add.
    down_act_scale = silu_out_scale
    if residual_dyn:
        df_acc = wd_i8.astype(np.int32) @ sf.astype(np.int32) + bd
        df_fp = (
            df_acc.astype(np.float32)
            * np.float32(down_act_scale)
            * wd_sc.astype(np.float32)
        )
        # The device add2 reconstructs the residual from the int8+scale that
        # flowed through the of_x1 fifo (x1_i8 * x1_scale), NOT the exact
        # pre-quant x1_fp. Mirror that: reconstruct from the quantized x1.
        x1_resid_fp = x1_i8.astype(np.float32) * np.float32(x1_scale)
        xo_fp = x1_resid_fp + df_fp
        # Scale computed EXACTLY as the device rescale-add kernel (see add1).
        xo_scale = _dyn_scale(xo_fp)
        xo_i8 = requant(xo_fp, np.float32(sw_recip(xo_scale)))
        x_out = (xo_i8, float(xo_scale))  # (int8[D], fp32) per-token residual
    elif residual_fp32:
        df_acc = wd_i8.astype(np.int32) @ sf.astype(np.int32) + bd
        df_fp = (
            df_acc.astype(np.float32)
            * np.float32(down_act_scale)
            * wd_sc.astype(np.float32)
        )
        x_out = x1 + df_fp  # fp32 residual out
    else:
        df = numpy_gemm_perchan(sf, down_act_scale, wd_i8, wd_sc, bd, INV_ACT_SCALE)
        # 13) add
        x_out = i8_add_wrap(df, x1)

    scales = dict(
        q_out_scales=q_out_scales,
        q_inv_outs=q_inv_outs,
        sv_out_scales=sv_out_scales,
        sv_inv_out_scales=sv_inv_out_scales,
        o_act_scale=o_act_scale,
        o_inv_act_scale=o_inv_act_scale,
        k_scales=k_scales,
        v_scales=v_scales,
        up_out_scale=up_out_scale,
        up_inv_out=up_inv_out,
        silu_up_scale=silu_up_scale,
        silu_inv_out_scale=silu_inv_out_scale,
        down_act_scale=down_act_scale,
    )
    return x_out, scales


def _selftest():
    """Quick numpy-only sanity: layer runs end-to-end, output shape/dtype
    matches, no NaNs, scale magnitudes are plausible."""
    rng = np.random.default_rng(0)
    layer = gen_layer_mh(rng)
    layer["lut_exp"] = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    layer["lut_silu"] = silu_lut(SILU_GATE_SCALE)

    x = rng.integers(-32, 33, size=D, dtype=np.int8)
    y, scales = numpy_layer_mh_forward(x, layer)

    assert y.shape == (D,) and y.dtype == np.int8
    assert not np.isnan(y).any()
    sat = int((y == 127).sum() + (y == -128).sum())

    print(f"selftest: out shape={y.shape} dtype={y.dtype} sat={sat}/{D}")
    print(
        f"  q_out_scales:     min={scales['q_out_scales'].min():.4f} "
        f"max={scales['q_out_scales'].max():.4f}"
    )
    print(
        f"  sv_out_scales:    min={scales['sv_out_scales'].min():.4f} "
        f"max={scales['sv_out_scales'].max():.4f}"
    )
    print(f"  o_act_scale:      {scales['o_act_scale']:.4f}")
    print(f"  up_out_scale:     {scales['up_out_scale']:.4f}")
    print(f"  silu_out_scale:   {scales['silu_inv_out_scale']:.4f}^-1")
    print(f"  down_act_scale:   {scales['down_act_scale']:.4f}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_selftest())
