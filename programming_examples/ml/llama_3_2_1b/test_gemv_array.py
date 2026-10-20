#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Test the shared 8-lane GEMV array against the per-channel kernels' numpy mirror.

Every projection of every layer is checked byte for byte: fp32 outputs as
`float(acc + bias) * act_scale * w_scale`, gate as that times `GATE_INV`
rounded the way the kernels round. `--time` reports device time and the
achieved weight bandwidth.

Interfaces, per layer unit and core c (one output object of OUT_BYTES each):
- an unfused projection: its fp32 rows (or int8 for gate) for rows
  [c * rows/8, (c + 1) * rows/8), at offset 0;
- `--fuse-qkv`, one unit on one activation: KV group c's combined chunk
  (`COMB_BYTES`), as llama_qkv_combine lays it out: Q heads 4c..4c+3
  requanted as llama_q_requant does (256 B body, 4 x 8 B scale tail), then
  KV head c's k and v fp32 rows (256 B each), then 256 B reserved for S2's
  RoPE cos/sin (zeros). RoPE itself is not applied yet.
- `--fuse-resid`, the o and down units: every activation object grows to
  `RESID_ACT_BYTES` and carries the residual int8[2048] plus its fp32 scale
  at `RESID_OFF`. Those units send no rows to the host; the glue core
  returns llama_rescale_add_D's output (`RESID_OUT_BYTES`: int8[2048], the
  new scale, 4 zero bytes) per unit, in unit order.
"""

import argparse
import sys
from pathlib import Path

import aie.iron as iron
import aie_fp
import numpy as np
from aie.utils import DefaultNPURuntime
from aie2_gemv_array import (
    ACT_BYTES,
    COMB_BYTES,
    LM_OUT_BYTES,
    LM_SLOT,
    N_CORES,
    OUT_BYTES,
    PROJECTIONS,
    RESID_ACT_BYTES,
    RESID_OFF,
    RESID_OUT_BYTES,
    RESID_UNITS,
    SLOT,
    D,
    build,
    build_lm,
    lane_slots,
    layer_units,
    rows_per_slot,
)
from llama_npu import add_artifact_args, design_kernel

DATA_DIR = Path(__file__).parent / "data"
FILES = {"q": "wq", "k": "wk", "v": "wv", "o": "wo", "g": "wg", "u": "wu", "d": "wd"}


def real_layer(layer, order):
    ld = DATA_DIR / f"layer_{layer:02d}"
    out = {}
    for p in order:
        k, rows, _ = PROJECTIONS[p]
        w = np.fromfile(ld / f"{FILES[p]}.i8.bin", np.int8).reshape(rows, k)
        sc = np.fromfile(ld / f"{FILES[p]}.scales.f32.bin", np.float32)
        out[p] = (w, np.zeros(rows, np.int32), sc)
    return out


def synthetic_layer(rng, order):
    out = {}
    for p in order:
        k, rows, _ = PROJECTIONS[p]
        w = rng.integers(-127, 128, size=(rows, k), dtype=np.int8)
        bias = rng.integers(-(1 << 16), 1 << 16, size=rows, dtype=np.int32)
        sc = (rng.random(rows, dtype=np.float32) * 2e-3 + 1e-4).astype(np.float32)
        out[p] = (w, bias, sc)
    return out


def pack_lane(layer_w, order, core):
    parts = []
    for p in order:
        k, rows, _ = PROJECTIONS[p]
        w, bias, sc = layer_w[p]
        rc = rows // N_CORES
        rps = rows_per_slot(k)
        n = rc // rps
        lo = core * rc
        slots = np.zeros((n, SLOT), np.int8)
        slots[:, : rps * k] = w[lo : lo + rc].reshape(n, rps * k)
        slots[:, rps * k : rps * k + 4 * rps] = (
            bias[lo : lo + rc].reshape(n, rps).view(np.int8)
        )
        slots[:, rps * k + 4 * rps : rps * k + 8 * rps] = (
            sc[lo : lo + rc].reshape(n, rps).view(np.int8)
        )
        parts.append(slots.reshape(-1))
    return np.concatenate(parts)


def make_act(rng, k):
    a = np.zeros(ACT_BYTES, np.int8)
    a[:k] = rng.integers(-127, 128, size=k, dtype=np.int8)
    scale = np.float32(rng.random() * 0.09 + 0.01)
    a[k : k + 4] = np.frombuffer(scale.tobytes(), np.int8)
    return a, scale


def reference(layer_w, p, act, act_scale, gate_inv):
    k, _, kind = PROJECTIONS[p]
    w, bias, sc = layer_w[p]
    acc = w.astype(np.int32) @ act[:k].astype(np.int32) + bias
    fp = (acc.astype(np.float32) * np.float32(act_scale)).astype(np.float32) * sc
    if kind == "i8":
        return aie_fp.round_to_i8((fp * np.float32(gate_inv)).astype(np.float32))
    return fp.astype(np.float32)


def comb_reference(layer_w, act, act_scale, core):
    """KV group `core`'s combined chunk: llama_q_requant on its four Q heads, then k, v fp32."""
    q = reference(layer_w, "q", act, act_scale, 0.0)[core * 256 : (core + 1) * 256]
    k = reference(layer_w, "k", act, act_scale, 0.0)[core * 64 : (core + 1) * 64]
    v = reference(layer_w, "v", act, act_scale, 0.0)[core * 64 : (core + 1) * 64]
    body = np.empty(256, np.int8)
    tail = np.zeros((4, 2), np.float32)
    for h in range(4):
        head = q[h * 64 : (h + 1) * 64]
        absmax = np.float32(max(float(np.abs(head).max()), 1e-12))
        scale = np.float32(absmax * np.float32(1.0 / 127.0))
        body[h * 64 : (h + 1) * 64] = aie_fp.round_to_i8(head * aie_fp.sw_recip(scale))
        tail[h, 0] = scale
    return np.concatenate(
        [
            body,
            tail.view(np.int8).reshape(-1),
            k.view(np.int8),
            v.view(np.int8),
            np.zeros(256, np.int8),
        ]
    )


def rescale_reference(proj, resid, resid_scale):
    """llama_rescale_add_D: x = resid * scale + proj, requanted with its own absmax scale."""
    x = aie_fp.aie_add(
        (resid.astype(np.float32) * np.float32(resid_scale)).astype(np.float32), proj
    )
    absmax = np.float32(max(float(np.abs(x).max()), 1e-12))
    scale = np.float32(absmax * np.float32(1.0 / 127.0))
    body = aie_fp.round_to_i8((x * aie_fp.sw_recip(scale)).astype(np.float32))
    return np.concatenate([body, np.array([scale, 0.0], np.float32).view(np.int8)])


def topk_reference(logits, gidx_base, kset):
    set_logit, set_gidx = [], []
    for i, v in enumerate(logits):
        if len(set_logit) < kset:
            set_logit.append(v)
            set_gidx.append(gidx_base + i)
            continue
        mi = int(np.argmin(np.asarray(set_logit, np.float32)))
        if v > set_logit[mi]:
            set_logit[mi] = v
            set_gidx[mi] = gidx_base + i
    return np.asarray(set_logit, np.float32), np.asarray(set_gidx, np.int32)


def run_lm(opts):
    vocab = opts.lm_tiles * 4
    rng = np.random.default_rng(opts.seed)
    if opts.weights == "real":
        embed = np.fromfile(DATA_DIR / "embed.i8.bin", np.int8).reshape(-1, D)[:vocab]
        embed_sc = np.fromfile(DATA_DIR / "embed.scales.f32.bin", np.float32)[:vocab]
    else:
        embed = rng.integers(-127, 128, size=(vocab, D), dtype=np.int8)
        embed_sc = (rng.random(vocab, dtype=np.float32) * 2e-3 + 1e-4).astype(
            np.float32
        )
    tbl = np.zeros((opts.lm_tiles, LM_SLOT), np.int8)
    tbl[:, 64 : 64 + 4 * D] = embed.reshape(opts.lm_tiles, 4 * D)
    tbl[:, 64 + 4 * D + 16 :] = embed_sc.reshape(opts.lm_tiles, 4).view(np.int8)
    act, act_scale = make_act(rng, D)
    act = act[: D + 8]

    logits = np.empty(vocab, np.float32)
    for r0 in range(0, vocab, 8192):
        acc = embed[r0 : r0 + 8192].astype(np.int32) @ act[:D].astype(np.int32)
        logits[r0 : r0 + 8192] = (
            acc.astype(np.float32) * np.float32(act_scale)
        ).astype(np.float32) * embed_sc[r0 : r0 + 8192]

    w_t = iron.tensor(tbl.reshape(-1), dtype=np.int8)
    a_t = iron.tensor(act, dtype=np.int8)
    s_t = iron.zeros([N_CORES * LM_OUT_BYTES], dtype=np.int8)
    npu = design_kernel(build_lm, opts, LM_TILES=opts.lm_tiles, KSET=opts.kset)
    buffers = [w_t, a_t, s_t]
    DefaultNPURuntime.load_and_run(npu, buffers)
    s_t.to("cpu")
    sets = s_t.numpy().reshape(N_CORES, LM_OUT_BYTES)

    ok = True
    per_core = vocab // N_CORES
    k = opts.kset
    cand = []
    for c in range(N_CORES):
        ref_l, ref_g = topk_reference(
            logits[c * per_core : (c + 1) * per_core], c * per_core, k
        )
        dev_l = sets[c, : 4 * k].view(np.float32)
        dev_g = sets[c, 4 * k : 8 * k].view(np.int32)
        dev_s = sets[c, 8 * k : 12 * k].view(np.float32)
        dev_n = int(sets[c, 12 * k : 12 * k + 4].view(np.int32)[0])
        same = (
            dev_n == k
            and np.array_equal(dev_l.view(np.int32), ref_l.view(np.int32))
            and np.array_equal(dev_g, ref_g)
            and np.array_equal(dev_s.view(np.int32), embed_sc[ref_g].view(np.int32))
        )
        print(f"core {c}: set {'BIT-EXACT' if same else 'DIFF'}")
        ok &= same
        cand.extend(zip(dev_l.tolist(), dev_g.tolist()))
    best = max(cand, key=lambda t: (t[0], -t[1]))[1]
    ref_best = int(np.argmax(logits))
    print(f"merged argmax: dev={best} ref={ref_best}")
    ok &= best == ref_best

    if opts.time:
        for _ in range(opts.n_warmup):
            DefaultNPURuntime.load_and_run(npu, buffers)
        times = []
        for _ in range(opts.n_iters):
            _h, result = DefaultNPURuntime.load_and_run(npu, buffers)
            times.append(result.npu_time / 1e3)
        t = np.array(times)
        print(
            f"time: n={len(t)} min={t.min():.1f} us median={np.median(t):.1f} us  "
            f"table={tbl.nbytes / 1e6:.2f} MB  {tbl.nbytes / (t.min() * 1e3):.2f} GB/s"
        )
    print(f"\ngemv_array lm_head: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main():
    p = argparse.ArgumentParser()
    add_artifact_args(p)
    p.add_argument("--lm", action="store_true", help="run lm_head instead of a layer")
    p.add_argument("--lm-tiles", type=int, default=32064)
    p.add_argument("--kset", type=int, default=8)
    p.add_argument("--weights", choices=("synthetic", "real"), default="synthetic")
    p.add_argument("--layers", type=int, nargs="+", default=[0])
    p.add_argument("--order", default="qkvogud")
    p.add_argument("--gate-inv", type=float, default=20.0)
    p.add_argument("--memtile-depth", type=int, default=0)
    p.add_argument("--fuse-qkv", action="store_true")
    p.add_argument("--fuse-resid", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", action="store_true")
    p.add_argument("--no-check", action="store_true")
    p.add_argument("--n-warmup", type=int, default=5)
    p.add_argument("--n-iters", type=int, default=10)
    opts = p.parse_args()
    if opts.lm:
        return run_lm(opts)
    order = opts.order
    n_layers = len(opts.layers)
    rng = np.random.default_rng(opts.seed)

    layers = []
    for layer in opts.layers:
        if opts.weights == "real":
            layers.append(real_layer(layer, order))
        else:
            layers.append(
                synthetic_layer(np.random.default_rng(opts.seed + layer), order)
            )

    lane_bytes = lane_slots(order) * SLOT
    wblob = np.empty(N_CORES * n_layers * lane_bytes, np.int8)
    for li, lw in enumerate(layers):
        for c in range(N_CORES):
            off = (li * N_CORES + c) * lane_bytes
            wblob[off : off + lane_bytes] = pack_lane(lw, order, c)

    units = layer_units(order, opts.fuse_qkv)
    resid_units = [u for u in units if opts.fuse_resid and u in RESID_UNITS]
    act_bytes = RESID_ACT_BYTES if opts.fuse_resid else ACT_BYTES
    acts = []
    scales = []
    resids = []
    for li in range(n_layers):
        for unit in units:
            a, s = make_act(rng, PROJECTIONS[unit[0]][0])
            a = np.concatenate([a, np.zeros(act_bytes - ACT_BYTES, np.int8)])
            if unit in resid_units:
                r = rng.integers(-127, 128, size=D, dtype=np.int8)
                rs = np.float32(rng.random() * 0.09 + 0.01)
                a[RESID_OFF : RESID_OFF + D] = r
                a[RESID_OFF + D : RESID_OFF + D + 4] = np.frombuffer(
                    rs.tobytes(), np.int8
                )
                resids.append((r, rs))
            acts.append(a)
            scales.append(s)
    acts_blob = np.concatenate(acts)
    n_steps = n_layers * (len(units) - len(resid_units))
    n_res = n_layers * len(resid_units)

    w_t = iron.tensor(wblob, dtype=np.int8)
    a_t = iron.tensor(acts_blob, dtype=np.int8)
    o_t = iron.zeros([max(N_CORES * n_steps * OUT_BYTES, 1)], dtype=np.int8)
    r_t = iron.zeros([max(n_res * RESID_OUT_BYTES, 1)], dtype=np.int8)
    npu = design_kernel(
        build,
        opts,
        N_LAYERS=n_layers,
        ORDER=order,
        GATE_INV=opts.gate_inv,
        MEMTILE_DEPTH=opts.memtile_depth,
        FUSE_QKV=opts.fuse_qkv,
        FUSE_RESID=opts.fuse_resid,
    )
    buffers = [w_t, a_t, o_t, r_t]
    _h, result = DefaultNPURuntime.load_and_run(npu, buffers)

    ok = True
    if not opts.no_check:
        o_t.to("cpu")
        r_t.to("cpu")
        out = o_t.numpy()[: N_CORES * n_steps * OUT_BYTES].reshape(
            N_CORES, n_steps, OUT_BYTES
        )
        res = r_t.numpy()[: n_res * RESID_OUT_BYTES].reshape(n_res, RESID_OUT_BYTES)
        step = 0
        act_i = 0
        res_i = 0
        for li, lw in enumerate(layers):
            for proj in units:
                ai = act_i
                act_i += 1
                if proj in resid_units:
                    fp = reference(lw, proj, acts[ai], scales[ai], opts.gate_inv)
                    ref = rescale_reference(fp, *resids[res_i])
                    bad = int(np.count_nonzero(res[res_i] != ref))
                    tag = "BIT-EXACT" if bad == 0 else f"MISMATCH {bad} B"
                    print(f"layer {opts.layers[li]:2d} {proj}+rescale_add: {tag}")
                    ok &= bad == 0
                    res_i += 1
                    continue
                if proj == "qkv":
                    ref = np.stack(
                        [
                            comb_reference(lw, acts[ai], scales[ai], c)
                            for c in range(N_CORES)
                        ]
                    )
                    bad = int(np.count_nonzero(out[:, step, :COMB_BYTES] != ref))
                    tag = "BIT-EXACT" if bad == 0 else f"MISMATCH {bad} B"
                    print(f"layer {opts.layers[li]:2d} qkv: 8 combined chunks  {tag}")
                    ok &= bad == 0
                    step += 1
                    continue
                k, rows, kind = PROJECTIONS[proj]
                ref = reference(lw, proj, acts[ai], scales[ai], opts.gate_inv)
                rc = rows // N_CORES
                nbytes = rc * (1 if kind == "i8" else 4)
                dev = out[:, step, :nbytes].reshape(-1)
                dev = dev if kind == "i8" else dev.view(np.float32)
                ref_bytes = ref.view(np.int8) if kind == "i8" else ref
                bad = int(
                    np.count_nonzero(
                        dev.view(np.int8) != np.asarray(ref_bytes).view(np.int8)
                    )
                )
                tag = "BIT-EXACT" if bad == 0 else f"MISMATCH {bad} B"
                print(f"layer {opts.layers[li]:2d} {proj}: {rows} rows  {tag}")
                ok &= bad == 0
                step += 1

    if opts.time:
        for _ in range(opts.n_warmup):
            DefaultNPURuntime.load_and_run(npu, buffers)
        times = []
        for _ in range(opts.n_iters):
            _h, result = DefaultNPURuntime.load_and_run(npu, buffers)
            times.append(result.npu_time / 1e3)
        t = np.array(times)
        gbps = wblob.nbytes / (t.min() * 1e3)
        print(
            f"time: n={len(t)} min={t.min():.1f} us median={np.median(t):.1f} us  "
            f"weights={wblob.nbytes / 1e6:.2f} MB  {gbps:.2f} GB/s at min"
        )

    print(f"\ngemv_array: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
