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
    KV_BLOCK_WORDS,
    LMW_GAMMA_BYTES,
    M_MAX,
    REC_BYTES,
    RQ,
    RROW,
    WLM_TOTAL,
    wblob_layout,
)
from gen_llama_data import quant_int8_perchan_absmax
from llama_config import HEAD_D, N_HEADS_KV, VOCAB, D
from llama_decode import Decoder, DecodeShape, tokens_of
from llama_oracle import (
    Layer,
    chain_forward_logits,
    dyn_scale,
    embed_seed_ref,
    greedy_verdict,
    requant,
    topk_sample_reference,
    xoshiro_seed,
)
from ml_dtypes import bfloat16
from numpy_llama_ref import _load_bf16, _load_qw, rope_table

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


def build_fixtures(shape, seed, filled):
    """A random layer-0 residual (int8[D], f32 scale, fp32[D]), and random
    layers whose shape.t-row KV caches hold random rows below `filled`."""
    rng = np.random.default_rng(seed)
    x_fp = rng.uniform(-1.6, 1.6, size=D).astype(np.float32)
    res_scale = dyn_scale(x_fp)
    x_i8 = requant(x_fp, np.float32(1.0) / res_scale)

    layers = []
    for _ in range(shape.n_layers):
        layer = Layer.random(rng, shape.t)
        for cache in (layer.kcache, layer.vcache):
            cache[:, :filled] = rng.integers(
                -32, 33, size=(N_HEADS_KV, filled, HEAD_D), dtype=np.int8
            )
        for slot in (layer.k_slot, layer.v_slot):
            slot[:, :filled] = rng.uniform(0.02, 0.08, size=(N_HEADS_KV, filled))
        layers.append(layer)
    return (x_i8, res_scale, x_fp), layers


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


def replay(shape, records, passes, residual, layers, p0, rope, lm, sampler, rng, plan):
    """Check a dispatch's records against the numpy oracle.

    The oracle runs from the caches the caller set and position p0 on,
    following the device's tokens.

    Args:
        records: The dispatch's packed records.
        passes: Its passes.
        residual: The first token's layer-0 residual.
        layers: The oracle's layers, their caches as the dispatch found them.
        p0: The first token's position.
        rope: The RoPE table.
        lm: (embed_i8, embed_sc, final_norm gamma).
        sampler: The sampler's (temperature, top_k).
        rng: The generation's xoshiro state as the dispatch found it.
        plan: Per token, (the prompt token it inputs, or None to input the
            previous token's sample (the first token: residual); the index of
            the record the lm_head and sampler left for it, or None).

    Returns:
        (failed tokens, the last token's final residual, the xoshiro state
        the dispatch leaves).
    """
    embed_i8, embed_sc, gamma = lm
    temperature, top_k = sampler
    fails = 0
    hidden = None
    cur = residual
    dev_toks = tokens_of(records, passes)
    for t, (prompt, rec_at) in enumerate(plan):
        if prompt is not None:
            cur = embed_seed_ref(embed_i8[prompt], float(embed_sc[prompt]))
        logits, hidden = chain_forward_logits(cur, layers, p0 + t, rope, lm)
        if rec_at is None:
            print(f"  token {t} (pos {p0+t}): prompt, no sample", flush=True)
            continue
        dev_tok = dev_toks[rec_at]
        if temperature > 0:
            ref, rng = topk_sample_reference(
                logits, temperature, top_k, rng, k_set=shape.kset
            )
            ok = dev_tok == ref
            tag = f"sampled, state {int(rng[0]):08x}"
        else:
            ok, ref, tag = greedy_verdict(logits, dev_tok)
        cur = embed_seed_ref(embed_i8[dev_tok], float(embed_sc[dev_tok]))
        rec = records[rec_at * REC_BYTES : (rec_at + 1) * REC_BYTES]
        seed_ok = np.array_equal(rec[:D], embed_i8[dev_tok]) and rec[
            D : D + 4
        ].tobytes() == (np.float32(embed_sc[dev_tok]).tobytes())
        if not seed_ok:
            tag += ", seed DIFF"
        ok = ok and seed_ok
        fails += not ok
        print(
            f"  token {t} (pos {p0+t}): {'PASS' if ok else 'FAIL'}  "
            f"dev={dev_tok} ref={ref}  [{tag}]",
            flush=True,
        )
    return fails, hidden, rng


def check_kv(shape, layers, kv, changed, positions):
    """Print whether the device's KV cache holds the oracle's rows at `positions`.

    Args:
        shape: The build.
        layers: The oracle's layers, after it appended those rows.
        kv: The device's KV cache words after the dispatch.
        changed: The indices of the words the dispatch changed.
        positions: The rows the dispatch appended.

    Returns:
        1 if a row differs or the dispatch changed a word outside them, else 0.
    """
    want = np.concatenate(
        [shape.kv_word_index(L, positions).ravel() for L in range(len(layers))]
    )
    rows = np.concatenate(
        [shape.kv_row_words(layer, positions).ravel() for layer in layers]
    )
    wrong = int(np.count_nonzero(kv[want] != rows))
    wrong += int(np.count_nonzero(~np.isin(changed, want)))
    print(f"  KV cache: {'PASS' if not wrong else f'{wrong} words FAIL'}")
    return int(wrong != 0)


def check_rng(shape, stage, rng):
    """Print whether `stage` holds the xoshiro state `rng`; return 1 if not."""
    ok = shape.rng_state(stage) == list(rng)
    print(f"  sampler state: {'PASS' if ok else 'FAIL'}", flush=True)
    return int(not ok)


def check_final(shape, stage, hidden, row):
    """Print whether `stage` holds `hidden` as final residual row `row`;
    return 1 if not."""
    xo_i8, xo_scale, xo_fp = hidden
    at = stage[shape.final_at + row * RROW : shape.final_at + (row + 1) * RROW]
    ok = (
        np.array_equal(at[:D], xo_i8)
        and at[D : D + 4].tobytes() == np.float32(xo_scale).tobytes()
        and at[RQ:].tobytes() == np.asarray(xo_fp, np.float32).tobytes()
    )
    print(f"  final hidden: {'PASS' if ok else 'FAIL'}", flush=True)
    return int(not ok)


def main(argv=None):
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
    opts = p.parse_args(argv)
    shape = DecodeShape(opts.n_layers, opts.t, opts.kset, opts.p_max)
    if opts.passes is not None and not 1 <= opts.passes <= shape.p_max:
        p.error(f"--passes must be 1 to --p-max ({shape.p_max})")
    fseed = opts.seed
    sample = (opts.sample_temp, opts.sample_topk, opts.sample_seed)
    sampler = sample[:2]
    rng0 = xoshiro_seed(opts.sample_seed)
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

    residual0, layers = build_fixtures(shape, fseed, p0)
    rope = rope_table(shape.t, HEAD_D)
    kv0 = shape.pack_kv(layers)
    stage0 = shape.pack_stage(layers, gamma, residual0, opts.sample_seed)

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
                shape.params(pos, passes, sampler, batch, batch_ends_prompt=ends),
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
        whole = dispatch(stage_t, kv_t, shape.params(p0, passes, sampler))
        kv_t.to("cpu")
        stage_t.to("cpu")
        kv_whole, stage_whole = kv_t.numpy().copy(), stage_t.numpy().copy()
        kv_t = iron.tensor(kv0, dtype=np.int32)
        stage_t = iron.tensor(stage0, dtype=np.int8)
        recs = [
            dispatch(stage_t, kv_t, shape.params(p0 + t, 1, sampler))
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
        tok_fails, hidden, rng = replay(
            shape,
            whole,
            passes,
            residual0,
            layers,
            p0,
            rope,
            lm,
            sampler,
            rng0,
            [(None, t) for t in range(passes)],
        )
        hidden_fails += check_final(shape, stage_whole, hidden, 0)
        carry_fails += check_rng(shape, stage_whole, rng)
        carry_fails += check_kv(
            shape,
            layers,
            kv_whole,
            np.flatnonzero(kv_whole != kv0),
            np.arange(p0, p0 + passes),
        )
    elif opts.repeats > 1:
        first = None
        for r in range(opts.repeats):
            records = dispatch(
                iron.tensor(stage0, dtype=np.int8),
                iron.tensor(kv0, dtype=np.int32),
                shape.params(p0, passes, sampler),
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
        tok_fails, _, _ = replay(
            shape,
            first,
            passes,
            residual0,
            layers,
            p0,
            rope,
            lm,
            sampler,
            rng0,
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
        residual_d = residual0
        rng = rng0
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
            fails, hidden, rng = replay(
                shape,
                toks_out,
                passes,
                residual_d,
                layers,
                pos,
                rope,
                lm,
                sampler,
                rng,
                plans[d],
            )
            tok_fails += fails
            carry_fails += check_rng(shape, stage, rng)
            kv_dev[kv_idx] = kv_val
            carry_fails += check_kv(
                shape, layers, kv_dev, kv_idx, np.arange(pos, pos + len(plan))
            )
            # A batch's rows are rows of its one pass; a pass's token is row 0.
            hidden_fails += check_final(
                shape, stage, hidden, len(schedule[d][2]) - 1 if schedule[d][2] else 0
            )
            # The oracle's next layer-0 input is the residual the device made
            # of the last token's embed row (in its record) and left in the
            # stage.
            if plan[-1][1] is not None:
                last = plan[-1][1] * REC_BYTES
                residual_d = embed_seed_ref(
                    toks_out[last : last + D].copy(),
                    float(
                        np.frombuffer(
                            toks_out[last + D : last + D + 4].tobytes(), np.float32
                        )[0]
                    ),
                )
                seed_left = np.zeros(RROW, dtype=np.int8)
                seed_left[:D] = residual_d[0]
                seed_left[D : D + 4] = np.frombuffer(
                    np.float32(residual_d[1]).tobytes(), np.int8
                )
                seed_left[RQ:] = residual_d[2].view(np.int8)
                if not np.array_equal(
                    stage[shape.seed_at : shape.seed_at + RROW], seed_left
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
                    end[f"{name} final residual"] = rs[d][1][at : at + RROW]
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
                        "seed left": rs[-1][1][shape.seed_at : shape.seed_at + RROW],
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
            shape.params(p0, passes, sampler),
            opts,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
