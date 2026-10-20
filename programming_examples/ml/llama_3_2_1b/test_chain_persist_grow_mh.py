#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The decode (aie2_decode.py) against the numpy oracle.

Each dispatch generates up to 4 tokens from a position t0, each token's
embed seed feeding the next through the stage, and appends each token's KV row
to the DDR cache the next tokens attend. The layers are a seeded random fixture
with a random KV prefix of t0 rows; the lm_head is the real embed table (or
a seeded synthetic one). The oracle follows the device's tokens: every token
must be the oracle's (greedy: its argmax or a near-tie; sampled: its draw),
every record's seed the oracle's embed seed of that token, the last token's
final residual in the stage the oracle's, and the whole KV cache the
oracle's.

Run (compiles the design, or reuses its JIT cache entry):
  python test_chain_persist_grow_mh.py --synthetic-weights
  python test_chain_persist_grow_mh.py --n-layers 16 --time
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import time
from pathlib import Path

import aie.iron as iron
import numpy as np
from aie2_decode import (
    ACT_BYTES,
    KV_BLOCK_WORDS,
    KV_TB,
    LMW_GAMMA_BYTES,
    M_MAX,
    REC_BYTES,
    RROW,
    WLM_TOTAL,
    wblob_layout,
)
from gen_exp_lut import exp_lut
from gen_llama_data import quant_int8_perchan_absmax
from gen_silu_lut import silu_lut
from llama_config import EXP_QUANT_SCALE, HEAD_D, N_HEADS_KV, SILU_GATE_SCALE, VOCAB, D
from llama_decode import Decoder, DecodeShape, embed_seed_ref, tokens_of
from ml_dtypes import bfloat16
from numpy_layer_mh import (
    gen_layer_mh,
    numpy_layer_mh_forward,
    numpy_rmsnorm_int8_dyn,
    requant,
)
from numpy_llama_ref import _load_bf16, _load_qw, rope_table
from numpy_topk_sample import topk_sample_reference

DATA_DIR = Path(__file__).parent / "data"


def synthetic_lm_weights(seed, rows_per_chunk=8192):
    """A seeded stand-in for data/'s embed table and final_norm gamma.

    Same shapes, dtypes and per-row INT8 quantization as gen_llama_data.py
    writes; the distributions follow Llama 3.2 1B's (embed std ~0.022, gamma
    ~N(2.35, 0.28)). Built in row chunks: the full fp32 table would be 1 GB.
    """
    rng = np.random.default_rng(seed)
    embed_i8 = np.empty((VOCAB, D), np.int8)
    embed_sc = np.empty(VOCAB, np.float32)
    for r0 in range(0, VOCAB, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, VOCAB)
        rows = rng.normal(0.0, 0.022, size=(r1 - r0, D)).astype(np.float32)
        embed_i8[r0:r1], embed_sc[r0:r1] = quant_int8_perchan_absmax(rows)
    gamma = rng.normal(2.35, 0.28, size=D).astype(bfloat16)
    return embed_i8, embed_sc, gamma


def build_fixtures(shape, seed, p0, n_tok):
    """Random layers and a shape.t-row KV cache with p0 rows filled, the rows
    of the n_tok tokens generated from position p0 on left zero."""
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = np.float32(np.maximum(np.abs(x_fp).max(), 1e-12) / 127.0)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    lut_exp = exp_lut(EXP_QUANT_SCALE).astype(np.float32)
    lut_silu = silu_lut(SILU_GATE_SCALE)
    layers = []
    for L in range(shape.n_layers):
        layer = gen_layer_mh(rng)
        layer["lut_exp"] = lut_exp
        layer["lut_silu"] = lut_silu
        for name in ("kcaches", "vcaches"):
            layer[name] = [
                rng.integers(-32, 33, size=shape.t * HEAD_D, dtype=np.int8)
                for _ in range(N_HEADS_KV)
            ]
        k_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, shape.t)).astype(np.float32)
        v_slot = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, shape.t)).astype(np.float32)
        for h in range(N_HEADS_KV):
            layer["kcaches"][h][p0 * HEAD_D :] = 0
            layer["vcaches"][h][p0 * HEAD_D :] = 0
        k_slot[:, p0:] = 0.0
        v_slot[:, p0:] = 0.0
        layer["k_scales_slot"] = k_slot
        layer["v_scales_slot"] = v_slot
        layers.append(layer)
    return x_i8, res_scale, layers


def chain_forward_logits(x_i8, res_scale, layers, position, embed_i8, embed_sc, gamma):
    """One token's forward at `position`, attending [0, position].

    Appends the token's KV row to the layers' caches and sets each layer's
    `scales`.

    Returns:
        (fp32 logits over the vocab, the final residual: int8[D], its scale).
    """
    x_cur = (x_i8.copy(), float(res_scale))
    for L in range(len(layers)):
        x_cur, scales = numpy_layer_mh_forward(
            x_cur,
            layers[L],
            position=position,
            residual_dyn=True,
            attn_perslot=True,
            attn_flowkv=True,
            attn_tb=KV_TB,
            attn_cores=2,
            attn_interleaved=True,
        )
        layers[L]["scales"] = scales
    xo_i8, xo_scale = x_cur
    normed_i8, norm_scale = numpy_rmsnorm_int8_dyn(xo_i8, gamma, np.float32(xo_scale))
    acc = embed_i8.astype(np.int64) @ normed_i8.astype(np.int64)
    logits = (
        acc.astype(np.float32) * np.float32(norm_scale) * embed_sc.astype(np.float32)
    ).astype(np.float32)
    return logits, xo_i8, np.float32(xo_scale)


DDR_BW_GBPS = 120.0  # NPU2 LPDDR5 ~120 GB/s


def time_decode(shape, session, buffers, params, opts):
    """Print per-token device time and the DMA floor, after main has checked
    the same dispatch against the oracle (bytes/token counts the live KV
    prefix the first token reads)."""
    passes = params["passes"]
    print(
        f"timing: warmup x{opts.n_warmup}, time x{opts.n_iters}  ({passes} tokens/dispatch)",
        flush=True,
    )
    for _ in range(opts.n_warmup):
        session(buffers, **params)
    times_ms = [session(buffers, **params) for _ in range(opts.n_iters)]
    arr = np.array(times_ms)
    disp_med = float(np.median(arr))
    per_token_ms = disp_med / passes
    kv_read = 2 * shape.n_layers * params["kvb0"] * KV_BLOCK_WORDS * 4
    bytes_per_token = (
        wblob_layout(shape.n_layers, shape.t)[1] + LMW_GAMMA_BYTES + WLM_TOTAL + kv_read
    )
    dma_floor_ms = bytes_per_token / (DDR_BW_GBPS * 1e9) * 1e3
    ratio = per_token_ms / dma_floor_ms
    verdict = "DMA-bound" if ratio < 1.5 else "compute-bound"
    print(
        f"per-dispatch: n={opts.n_iters} median={disp_med:.2f} ms "
        f"min={arr.min():.2f} max={arr.max():.2f} std={arr.std():.2f}"
    )
    print(f"per-token:    {per_token_ms:.2f} ms  ({1000.0/per_token_ms:.2f} tok/s)")
    print(
        f"bytes/token:  {bytes_per_token/1e6:.1f} MB  DMA floor {dma_floor_ms:.2f} ms"
    )
    print(f"ratio:        {ratio:.2f}x  -> {verdict}")


# Device-trajectory replay tolerances: the oracle follows the DEVICE's tokens
# (not its own greedy trajectory) since the token feeds back ON-CHIP. A device
# token passes if it is the oracle argmax OR a near-tie judged RELATIVE to the
# logit spread (gap/std): the vector fp32 multiplies the oracle does not mirror
# yet (rmsnorm, silu, up_requant) leave rare 1-ULP differences that grow with
# depth. A flip within the top cluster is quality-neutral; one far down would be
# real drift.
TOL_GAP_STD = 0.15  # gap as a fraction of the logit std
TOL_RANK = 5  # dev_tok must be within the oracle top-K


def set_pristine(layers, pristine):
    kcaches, vcaches, kslot, vslot = pristine
    for L in range(len(layers)):
        layers[L]["kcaches"] = [c.copy() for c in kcaches[L]]
        layers[L]["vcaches"] = [c.copy() for c in vcaches[L]]
        layers[L]["k_scales_slot"] = kslot[L].copy()
        layers[L]["v_scales_slot"] = vslot[L].copy()


def replay(shape, records, passes, x_i8, res_scale, layers, p0, rope, lm, sample, plan):
    """Check a dispatch's records against the numpy oracle.

    The oracle runs from the caches the caller set and position p0 on,
    following the device's tokens.

    Args:
        records: The dispatch's packed records.
        passes: Its passes.
        x_i8, res_scale: The first token's seed.
        layers: The oracle's layers, their caches as the dispatch found them.
        p0: The first token's position.
        rope: The RoPE table.
        lm: (embed_i8, embed_sc, final_norm gamma).
        sample: The sampler's (temperature, top_k, seed).
        plan: Per token, (the prompt token it inputs, or None to input the
            previous token's sample (the first token: x_i8); the index of
            the record the lm_head and sampler left for it, or None).

    Returns:
        (failed tokens, the last token's final residual and its scale).
    """
    embed_i8, embed_sc, gamma = lm
    temperature, top_k, seed = sample
    fails = 0
    hidden = None
    cur_x, cur_sc = x_i8, res_scale
    dev_toks = tokens_of(records, passes)
    for t, (prompt, rec_at) in enumerate(plan):
        if prompt is not None:
            cur_x, cur_sc = embed_seed_ref(embed_i8[prompt], float(embed_sc[prompt]))
        for L in range(len(layers)):
            layers[L]["cos"] = rope[p0 + t, :HEAD_D]
            layers[L]["sin"] = rope[p0 + t, HEAD_D:]
        logits, *hidden = chain_forward_logits(
            cur_x, cur_sc, layers, p0 + t, embed_i8, embed_sc, gamma
        )
        if rec_at is None:
            print(f"  token {t} (pos {p0+t}): prompt, no sample", flush=True)
            continue
        dev_tok = dev_toks[rec_at]
        if temperature > 0:
            ref = topk_sample_reference(
                logits, temperature, top_k, seed, k_set=shape.kset
            )
            ok = dev_tok == ref
            tag = "sampled"
        else:
            ref = int(np.argmax(logits))
            gap = float(logits.max() - logits[dev_tok])
            spread = float(logits.std())
            rank = int(np.where(np.argsort(logits)[::-1] == dev_tok)[0][0])
            near_tie = (gap / spread <= TOL_GAP_STD) and (rank < TOL_RANK)
            ok = dev_tok == ref or near_tie
            tag = (
                "exact"
                if dev_tok == ref
                else f"near-tie gap/std={gap/spread:.3f} rank={rank}"
            )
        cur_x, cur_sc = embed_seed_ref(embed_i8[dev_tok], float(embed_sc[dev_tok]))
        rec = records[rec_at * REC_BYTES : (rec_at + 1) * REC_BYTES]
        seed_ok = np.array_equal(rec[:D], cur_x) and rec[D : D + 4].tobytes() == (
            np.float32(cur_sc).tobytes()
        )
        if not seed_ok:
            tag += ", seed DIFF"
        ok = ok and seed_ok
        fails += not ok
        print(
            f"  token {t} (pos {p0+t}): {'PASS' if ok else 'FAIL'}  "
            f"dev={dev_tok} ref={ref}  [{tag}]",
            flush=True,
        )
    return fails, hidden


def check_final(shape, stage, hidden, row):
    """Print whether `stage` holds `hidden` as final residual row `row`;
    return 1 if not."""
    xo_i8, xo_scale = hidden
    at = stage[shape.final_at + row * RROW : shape.final_at + row * RROW + D + 4]
    ok = np.array_equal(at[:D], xo_i8) and at[D:].tobytes() == xo_scale.tobytes()
    print(f"  final hidden: {'PASS' if ok else 'FAIL'}", flush=True)
    return int(not ok)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-layers", type=int, default=2, help="decoder layers (16: the model)"
    )
    p.add_argument("--t", type=int, default=128, help="KV cache rows")
    p.add_argument(
        "--kset", type=int, default=8, help="the sampler's resident top-k set"
    )
    p.add_argument(
        "--p-max",
        type=int,
        default=inspect.signature(DecodeShape).parameters["p_max"].default,
        help="tokens a dispatch can generate",
    )
    p.add_argument(
        "--elf",
        help="run this full ELF, built by aie2_decode.py --full-elf-path "
        "at this test's configuration, instead of the JIT cache's",
    )
    p.add_argument(
        "--sample-temp",
        type=float,
        default=0.0,
        help="the sampler's temperature (0: greedy)",
    )
    p.add_argument(
        "--sample-topk",
        type=int,
        default=0,
        help="sample from the resident set's best k (0: the whole set)",
    )
    p.add_argument("--sample-seed", type=int, default=0, help="the sampler's seed")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time", action="store_true", help="measure per-token device time")
    p.add_argument("--n-iters", type=int, default=1)
    p.add_argument("--n-warmup", type=int, default=0)
    p.add_argument(
        "--synthetic-weights",
        action="store_true",
        help="use a seeded synthetic embed table + final_norm (same layout as data/) "
        "instead of the real Llama weights; the oracle runs on the same table",
    )
    p.add_argument(
        "--dispatches",
        type=int,
        default=1,
        help="dispatches run back to back like a generation loop: each carries the "
        "KV cache and the last token's seed into the next; every token is checked",
    )
    p.add_argument(
        "--idle-s",
        type=float,
        default=0.0,
        help="seconds to leave the NPU idle before each later dispatch; past "
        "amdxdna's autosuspend delay (5 s) the device suspends in between",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="run the first dispatch this many times in one process from the same "
        "inputs; every repeat must produce the first's tokens",
    )
    p.add_argument(
        "--start-pos",
        type=int,
        default=None,
        help="KV prefix length before the first generated token "
        "(default: --t // 2, or as much as leaves room for every generated token)",
    )
    p.add_argument(
        "--passes",
        type=int,
        help="tokens generated per dispatch (default: DecodeShape.passes_for's, per dispatch)",
    )
    p.add_argument(
        "--split",
        action="store_true",
        help="also generate the first dispatch's tokens one per dispatch, from the "
        "same inputs; the records, KV cache and stage must be byte-identical",
    )
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=0,
        help="the first this many tokens are a prompt's (seeded random ids), each "
        "input from its own embed row as the device gathers it, not the sample before",
    )
    p.add_argument(
        "--batch",
        type=int,
        choices=range(1, M_MAX + 1),
        default=M_MAX,
        help="the prompt's tokens a dispatch runs together, at most",
    )
    p.add_argument(
        "--vs-batch",
        type=int,
        choices=range(1, M_MAX + 1),
        help="also run the prompt in batches of at most this many tokens; the "
        "sampled records, the KV cache, the last final residual and the seed "
        "left must be byte-identical",
    )
    p.add_argument(
        "--save-run",
        help="run the dispatches and save what the device left to this .npz, "
        "without the oracle (which takes about 30 min a token at 128k)",
    )
    p.add_argument(
        "--from-run",
        help="check a --save-run .npz, saved with the same flags, without the NPU",
    )
    opts = p.parse_args()
    shape = DecodeShape(opts.n_layers, opts.t, opts.kset, opts.p_max)
    if opts.passes is not None and not 1 <= opts.passes <= shape.p_max:
        p.error(f"--passes must be 1 to --p-max ({shape.p_max})")
    fseed = opts.seed
    sample = (opts.sample_temp, opts.sample_topk, opts.sample_seed)
    n_disp = opts.dispatches
    p0 = opts.start_pos
    if p0 is None:
        p0 = min(
            shape.t // 2,
            shape.t - n_disp * (opts.passes or shape.p_max) - opts.prompt_tokens,
        )
    prompt_ids = [
        int(t)
        for t in np.random.default_rng(fseed + 1).integers(0, VOCAB, opts.prompt_tokens)
    ]

    def schedule_of(largest):
        """The run's dispatches with the prompt in batches of `largest` at most.

        Returns:
            (per dispatch, (position, passes, its batch of prompt tokens,
            whether the batch ends the prompt): the prompt's batches first,
            then --dispatches of generated tokens; per dispatch, per token,
            (its prompt token or None, the index of the record the lm_head
            and sampler leave for it, or None)).
        """
        schedule = []
        pos = p0
        sizes = shape.batches(p0, opts.prompt_tokens, largest)
        for i, n in enumerate(sizes):
            at = pos - p0
            schedule.append((pos, 1, prompt_ids[at : at + n], i == len(sizes) - 1))
            pos += n
        for _ in range(n_disp):
            schedule.append((pos, opts.passes or shape.passes_for(pos), [], True))
            pos += schedule[-1][1]
        plans = [
            (
                [
                    (tok, 0 if i == len(batch) - 1 and ends else None)
                    for i, tok in enumerate(batch)
                ]
                if batch
                else [(None, t) for t in range(n)]
            )
            for _, n, batch, ends in schedule
        ]
        return schedule, plans

    schedule, plans = schedule_of(opts.batch)
    passes = schedule[0][1]
    n_tok = sum(len(b) or n for _, n, b, _ in schedule)
    if p0 < 0 or p0 + n_tok > shape.t:
        p.error(f"{n_tok} tokens from position {p0} do not fit in T={shape.t}")
    if (n_disp > 1) + (opts.repeats > 1) + opts.split > 1:
        p.error("--dispatches, --repeats and --split are separate modes")
    if (opts.save_run or opts.from_run or opts.prompt_tokens) and (
        opts.time or opts.split or opts.repeats > 1
    ):
        p.error(
            "--save-run, --from-run and --prompt-tokens are for runs of --dispatches"
        )
    if opts.vs_batch and (opts.save_run or opts.from_run or not opts.prompt_tokens):
        p.error("--vs-batch is for a --prompt-tokens run on the NPU")
    config = json.dumps(
        dict(
            n_layers=shape.n_layers,
            t=shape.t,
            kset=shape.kset,
            seed=fseed,
            synthetic=opts.synthetic_weights,
            schedule=schedule,
            sample=sample,
            prompt_tokens=opts.prompt_tokens,
        )
    )

    assert VOCAB == VOCAB and D == D
    if opts.synthetic_weights:
        print(f"building synthetic embed/final_norm (seed {fseed}) ...", flush=True)
        embed_i8, embed_sc, gamma = synthetic_lm_weights(fseed)
    else:
        print(f"loading embed/final_norm from {DATA_DIR} ...", flush=True)
        embed_i8, embed_sc = _load_qw(DATA_DIR, "embed", (VOCAB, D))
        gamma = _load_bf16(DATA_DIR, "final_norm", (D,))
    lm = (embed_i8, embed_sc, gamma)

    x_i8, res_scale, layers = build_fixtures(shape, fseed, p0, n_tok)
    rope = rope_table(shape.t, HEAD_D)

    # Snapshot the PRISTINE caches (before any append) so the device gets the same
    # starting KV the numpy loop starts from. The device's growing-append then
    # accumulates the tokens on-chip exactly as numpy does in place.
    pristine = (
        [[c.copy() for c in lyr["kcaches"]] for lyr in layers],
        [[c.copy() for c in lyr["vcaches"]] for lyr in layers],
        [lyr["k_scales_slot"].copy() for lyr in layers],
        [lyr["v_scales_slot"].copy() for lyr in layers],
    )

    kv0 = shape.pack_kv(layers)
    stage0 = shape.pack_stage(layers, gamma, x_i8, res_scale)

    # --from-run leaves the NPU alone.
    decoder = None
    if not opts.from_run:
        print("packing the weights ...", flush=True)
        decoder = Decoder(shape, layers, lm, rope, opts.elf)
    print(
        f"chain_persist_grow_mh shape.n_layers={shape.n_layers} "
        f"passes={[len(b) or n for _, n, b, _ in schedule]}"
        f" P0={p0} KSET={shape.kset}"
        f"  temp={sample[0]} topk={sample[1]} seed={sample[2]}  fixture_seed={fseed}"
        f"  dispatches={n_disp} repeats={opts.repeats}",
        flush=True,
    )

    # The KV tensor is created once per run of dispatches and stays on the
    # device, so a generation loop's dispatches carry the cache with no host
    # copy.
    def dispatch(stage_t, kv_t, params):
        assert decoder is not None
        return decoder.dispatch(kv_t, stage_t, params)[0]

    def run_schedule(schedule):
        """Run a schedule's dispatches back to back on one KV and stage tensor.

        Returns:
            Per dispatch, (its records, the stage it left, the KV words it
            changed, their values).
        """
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        kv_prev = kv0
        runs = []
        for d, (pos, passes, batch, ends) in enumerate(schedule):
            if d:
                time.sleep(opts.idle_s)
            records = dispatch(
                stage_t,
                kv_t,
                shape.params(pos, passes, sample, batch, batch_ends_prompt=ends),
            )
            kv_t.to("cpu")
            stage_t.to("cpu")
            kv_now = kv_t.numpy()
            kv_idx = np.flatnonzero(kv_now != kv_prev)
            runs.append((records, stage_t.numpy().copy(), kv_idx, kv_now[kv_idx]))
            kv_prev = kv_now.copy()
        return runs

    # The oracle follows the device's tokens across dispatches, from the
    # seed each dispatch left; after each, the device's whole KV cache must
    # equal the oracle's.
    dev_tokens = []
    tok_fails = carry_fails = hidden_fails = vs_fails = 0
    if opts.split:
        # One dispatch of `passes` tokens, then the same tokens one per
        # dispatch: a token slot that does not run moves nothing.
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        whole = dispatch(stage_t, kv_t, shape.params(p0, passes, sample))
        kv_t.to("cpu")
        stage_t.to("cpu")
        kv_whole, stage_whole = kv_t.numpy().copy(), stage_t.numpy().copy()
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        recs = [
            dispatch(stage_t, kv_t, shape.params(p0 + t, 1, sample))
            for t in range(passes)
        ]
        kv_t.to("cpu")
        stage_t.to("cpu")
        n_rec = passes * REC_BYTES
        checks = {
            "records": np.array_equal(
                whole[:n_rec], np.concatenate([r[:REC_BYTES] for r in recs])
            ),
            "untouched records": not np.any(whole[n_rec:])
            and not any(np.any(r[REC_BYTES:]) for r in recs),
            "KV cache": np.array_equal(kv_whole, kv_t.numpy()),
            "stage": np.array_equal(stage_whole, stage_t.numpy()),
        }
        for name, same in checks.items():
            print(
                f"  {passes} tokens vs {passes} x 1, {name}: {'same' if same else 'DIFFER'}"
            )
            carry_fails += not same
        dev_tokens = tokens_of(whole, passes)
        tok_fails, hidden = replay(
            shape,
            whole,
            passes,
            x_i8,
            res_scale,
            layers,
            p0,
            rope,
            lm,
            sample,
            [(None, t) for t in range(passes)],
        )
        hidden_fails += check_final(shape, stage_whole, hidden, 0)
        wrong = int(np.count_nonzero(kv_whole != shape.pack_kv(layers)))
        print(f"  KV cache: {'PASS' if not wrong else f'{wrong} words FAIL'}")
        carry_fails += wrong != 0
    elif opts.repeats > 1:
        first = None
        for r in range(opts.repeats):
            records = dispatch(
                iron.tensor(stage0, dtype=np.int8),
                iron.tensor(kv0, dtype=np.int32),
                shape.params(p0, passes, sample),
            )
            toks = tokens_of(records, passes)
            if r == 0:
                first = records
                dev_tokens = toks
            same = toks == dev_tokens
            print(
                f"repeat {r}: tokens {toks} {'same' if same else 'DIFFER'}", flush=True
            )
            if not same:
                print(f"repeat {r} differs from repeat 0", file=sys.stderr)
                return 1
        set_pristine(layers, pristine)
        tok_fails, _ = replay(
            shape,
            first,
            passes,
            x_i8,
            res_scale,
            layers,
            p0,
            rope,
            lm,
            sample,
            [(None, t) for t in range(passes)],
        )
    else:
        # Like the KV tensor, the stage tensor stays on the device: the host
        # writes the first dispatch's seed into it, and each dispatch leaves
        # the next one's there. Per dispatch, the device's records, stage and
        # the KV words it changed; then the checks, which --save-run and
        # --from-run split off.
        if opts.from_run:
            saved = np.load(opts.from_run)
            if str(saved["config"]) != config:
                p.error(f"{opts.from_run} was run as {saved['config']}, not {config}")
            runs = [
                tuple(
                    saved[f"{k}{d}"] for k in ("records", "stage", "kv_idx", "kv_val")
                )
                for d in range(n_disp)
            ]
        else:
            runs = run_schedule(schedule)
            if opts.save_run:
                np.savez(
                    opts.save_run,
                    config=config,
                    **{
                        f"{k}{d}": v
                        for d, run in enumerate(runs)
                        for k, v in zip(("records", "stage", "kv_idx", "kv_val"), run)
                    },
                )
                print(f"saved the device's run to {opts.save_run}")
                return 0
        kv_dev = kv0.copy()
        x_d, sc_d = x_i8, res_scale
        for d, ((pos, passes, _, _), (toks_out, stage, kv_idx, kv_val)) in enumerate(
            zip(schedule, runs)
        ):
            plan = plans[d]
            toks = [
                tokens_of(toks_out, passes)[rec_at]
                for _, rec_at in plan
                if rec_at is not None
            ]
            dev_tokens += toks
            if len(schedule) > 1:
                print(f"dispatch {d} (pos {pos}): tokens {toks}", flush=True)
            fails, hidden = replay(
                shape,
                toks_out,
                passes,
                x_d,
                sc_d,
                layers,
                pos,
                rope,
                lm,
                sample,
                plans[d],
            )
            tok_fails += fails
            # The device appended exactly the rows the oracle did, and wrote
            # nothing else.
            kv_dev[kv_idx] = kv_val
            wrong = int(np.count_nonzero(kv_dev != shape.pack_kv(layers)))
            print(f"  KV cache: {'PASS' if not wrong else f'{wrong} words FAIL'}")
            carry_fails += wrong != 0
            # A batch's rows are rows of its one pass; a pass's token is row 0.
            hidden_fails += check_final(
                shape, stage, hidden, len(schedule[d][2]) - 1 if schedule[d][2] else 0
            )
            # The oracle's next layer-0 input is the last token's seed (body +
            # scale), which the device left in the stage with a zero pad.
            if plan[-1][1] is not None:
                last = plan[-1][1] * REC_BYTES
                x_d = toks_out[last : last + D].copy()
                sc_d = float(
                    np.frombuffer(
                        toks_out[last + D : last + D + 4].tobytes(), np.float32
                    )[0]
                )
                seed_left = np.zeros(ACT_BYTES, dtype=np.int8)
                seed_left[: D + 4] = toks_out[last : last + D + 4]
                if not np.array_equal(
                    stage[shape.seed_at : shape.seed_at + ACT_BYTES], seed_left
                ):
                    print(f"  seed carry: FAIL (dispatch {d})", flush=True)
                    carry_fails += 1
            else:
                untouched = not np.any(toks_out)
                print(f"  records: {'untouched' if untouched else 'WRITTEN'}")
                carry_fails += not untouched
        if opts.vs_batch:
            vs = schedule_of(opts.vs_batch)
            ends = []
            for (sched, pl), rs in (
                ((schedule, plans), runs),
                (vs, run_schedule(vs[0])),
            ):
                kv = kv0.copy()
                for _, _, kv_idx, kv_val in rs:
                    kv[kv_idx] = kv_val
                prompt_end = max(d for d, (_, _, b, e) in enumerate(sched) if b and e)
                end = {}
                for name, d in (("prompt's", prompt_end), ("last", len(rs) - 1)):
                    at = shape.final_at + max(len(sched[d][2]) - 1, 0) * RROW
                    end[f"{name} final residual"] = rs[d][1][at : at + D + 4]
                ends.append(
                    end
                    | {
                        "sampled records": np.concatenate(
                            [
                                r[0][k * REC_BYTES : (k + 1) * REC_BYTES]
                                for r, plan in zip(rs, pl)
                                for _, k in plan
                                if k is not None
                            ]
                        ),
                        "KV cache": kv,
                        "seed left": rs[-1][1][
                            shape.seed_at : shape.seed_at + ACT_BYTES
                        ],
                    }
                )
            for name, ours in ends[0].items():
                same = np.array_equal(ours, ends[1][name])
                print(
                    f"  batches of {opts.batch} vs {opts.vs_batch}, {name}: "
                    f"{'same' if same else 'DIFFER'}"
                )
                vs_fails += not same

    n = len(dev_tokens)
    print(
        f"\nchain_persist_grow_mh: {n - tok_fails}/{n} tokens PASS  "
        f"(on-chip feedback + GROWING KV + per-position rope)"
    )
    if opts.split:
        print(
            f"{passes} tokens vs {passes} x 1: {'PASS' if not carry_fails else 'FAIL'}"
        )
    if len(schedule) > 1:
        print(
            f"carry across {len(schedule)} dispatches: "
            f"{'PASS' if not carry_fails else 'FAIL'}"
        )
    if opts.vs_batch:
        print(
            f"prompt in batches of {opts.batch} vs {opts.vs_batch}: "
            f"{'PASS' if not vs_fails else 'FAIL'}"
        )
    if tok_fails or carry_fails or hidden_fails or vs_fails:
        return 1
    if opts.time:
        assert decoder is not None
        time_decode(
            shape,
            decoder.session,
            [
                decoder.w_t,
                iron.tensor(kv0, dtype=np.int32),
                decoder.lmw_t,
                iron.zeros([shape.p_max * REC_BYTES], dtype=np.int8),
                iron.tensor(stage0, dtype=np.int8),
            ],
            shape.params(p0, passes, sample),
            opts,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
