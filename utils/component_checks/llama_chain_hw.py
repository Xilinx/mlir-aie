#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Build and run the Llama 3.2 1B decode on NPU2.

Does the full N=16 persistent, growing-KV decode still build, match its
numpy oracle and run at the expected rate on real hardware. Compiles once,
at a 128k-row KV cache, with the design's own CLI (aie2_decode.py), then runs
that one full ELF through:
- test_chain_persist_grow_mh.py --contexts: a dispatch of 4 tokens at
  DISPATCH_RATE_AT and at each of CONTEXTS, from one random KV prefix, every
  token, KV row and final residual checked against the oracle, then timed
  (the fastest of 20 after 10 warmups). Attention reads only the live
  prefix, so the one build times every context;
- generate.py --check: a prompt in, tokens out, 5 requests, each generated
  token the oracle's greedy pick; its time to first token runs from before
  tokenization, split into the prefill dispatches and the host's rest;
- generate.py --prompt-tokens, no --check: the same TTFT split, 5 requests
  each, at PROMPT_LENGTHS (batched prefill's own exactness is already
  covered by the --contexts sweep's and test_llama's batched cases; this
  only times it, at prompt lengths long enough to need many batches);
- accuracy_vs_hf.py: the device's arithmetic (llama_oracle) against the
  true bf16 checkpoint, teacher-forced on a small fixed prompt (mean KL,
  top-1 agreement). Refuses (not skips) when the bf16 safetensors are
  missing, the same as for the INT8 weights.
Every run must exit 0 and pass every token. One run, no seed sweep: the
design pins its SA placer seed (aie2_decode.build's aiecc flags).

Needs the real Llama 3.2 1B checkpoint where amd/IRON's examples keep it,
``$IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/`` (default ``/srv``; see
gen_llama_data.WEIGHTS_DIR), or ``--weights``/``--tokenizer``, unless
``data/`` already holds the INT8 weights and the tokenizer: the decode test
and generate.py both load every layer's real weights (run without
--synthetic-weights). Generates ``data/`` once if it isn't complete, and
refuses to measure when that fails.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
any run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/llama_3_2_1b, with

    python3 aie2_decode.py --n-layers 16 --t 131072 --full-elf-path X.elf
    python3 test_chain_persist_grow_mh.py --n-layers 16 --t 131072 \
        --elf X.elf --contexts 64 1024 2048 4096 8192 16384 32768 65536 \
        131068 --time
    python3 generate.py --t 131072 --elf X.elf \
        --prompt "The capital of France is" --check
    python3 generate.py --t 131072 --elf X.elf --prompt-tokens 2048 \
        --max-new 1
"""

import argparse
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

from aiecc_stages import (
    PROFILE_NAME,
    Measured,
    add_pmode_argument,
    compile_rows,
    compile_stats,
    refuse,
    row,
    run_measured,
    stage_rows,
    write,
    write_unfinished,
)

N_LAYERS = 16
P_MAX = 4
T = 131072
# The positions of the timed dispatches' first tokens: 64 for the dispatch
# rate the earlier nightlies charted, then the contexts of the sweep.
DISPATCH_RATE_AT = 64
CONTEXTS = (1024, 2048, 4096, 8192, 16384, 32768, 65536, 131068)
N_ITERS = 20
N_WARMUP = 10
DATA_TIMEOUT = 1800
COMPILE_TIMEOUT = 1800
DECODE_TIMEOUT = 2400
GENERATE_TIMEOUT = 900

_LLAMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml", "llama_3_2_1b"
)

_TOKEN_LINE_RE = re.compile(
    r"^\s*token \d+(?: \(pos \d+\))?: (PASS|FAIL)", re.MULTILINE
)
_SUMMARY_RE = re.compile(
    r"^chain_persist_grow_mh: (\d+)/(\d+) tokens PASS", re.MULTILINE
)
_DECODE_RE = re.compile(
    r"^decode at (\d+): per-dispatch n=(\d+) min=([\d.]+) median=([\d.]+) "
    r"mad=([\d.]+) ms, (\d+) tokens/dispatch: ([\d.]+) ms/token \(([\d.]+) tok/s\)",
    re.MULTILINE,
)
_GENERATE_RE = re.compile(
    r"median of (\d+): TTFT ([\d.]+) ms \(min ([\d.]+); NPU dispatch ([\d.]+), "
    r"host ([\d.]+)\), decode (?:([\d.]+) tok/s|n/a)"
)
_CHECK_RE = re.compile(r"check: (\d+)/(\d+) tokens")
_ORACLE_RE = re.compile(r"^oracle: (\d+)/(\d+) generated tokens pass", re.MULTILINE)
# generate.py's prompt and length, the same every night.
GENERATE_PROMPT = "The capital of France is"
GENERATE_NEW = 32
# Prefill's own TTFT, at prompt lengths long enough to need many batches
# (mlir-air/FastFlowLM report TTFT by prompt length the same way).
PROMPT_LENGTHS = (512, 1024, 2048)
# accuracy_vs_hf.py's own default text, cut short (~28 s, 9.3 GB peak):
# small enough to run every night beside everything else above.
ACCURACY_MAX_TOKENS = 24
_ELF = f"build/chain_persist_grow_N{N_LAYERS}_PT{P_MAX}_T{T}.elf"
_PROJ_DIR = f"build/chain_persist_grow_N{N_LAYERS}_PT{P_MAX}_T{T}.prj"
_PROFILE = os.path.join(_PROJ_DIR, PROFILE_NAME)
# What generate.py and the decode test read from data/.
DATA_FILES = (
    "embed.i8.bin",
    "embed.scales.f32.bin",
    "final_norm.bf16.bin",
    "tokenizer.model",
    *(f"layer_{L:02d}/wd.scales.f32.bin" for L in range(N_LAYERS)),
)

# The design's CompileTime values, which the test script takes too.
_SHAPE_ARGS = [f"--p-max={P_MAX}", "--kset=8", f"--n-layers={N_LAYERS}", f"--t={T}"]


def missing_data() -> list[str]:
    """Return the files in data/ the runs read that it lacks."""
    return [
        f for f in DATA_FILES if not os.path.exists(os.path.join(_LLAMA_DIR, "data", f))
    ]


def missing_bf16(weights: str | None) -> bool:
    """Whether accuracy_vs_hf.py's bf16 reference model.safetensors is absent.

    Args:
        weights: ``--weights``, or None for gen_llama_data.WEIGHTS_DIR's default
            ($IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/, /srv if unset).
    """
    path = (
        Path(weights)
        if weights
        else (Path(os.environ.get("IRON_EXAMPLE_WEIGHTS_DIR", "/srv")) / "llama3.2-1b")
        / "model.safetensors"
    )
    return not path.exists()


def ensure_data(
    weights: str | None, tokenizer: str | None, env: dict
) -> Measured | None:
    """Generate data/ once, if this checkout doesn't already have all of it."""
    if not missing_data():
        return None
    args = [sys.executable, "gen_llama_data.py"]
    if weights:
        args += ["--weights", weights]
    if tokenizer:
        args += ["--tokenizer", tokenizer]
    return run_measured(args, _LLAMA_DIR, env, DATA_TIMEOUT)


def _clean_build_outputs() -> None:
    """Remove this config's full ELF and project dir.

    The design CLI writes into the persistent build/ dir and rebuilds only
    outputs that are out of date, so without this it would not recompile.
    """
    path = os.path.join(_LLAMA_DIR, _ELF)
    if os.path.exists(path):
        os.remove(path)
    proj = os.path.join(_LLAMA_DIR, _PROJ_DIR)
    if os.path.isdir(proj):
        shutil.rmtree(proj)


def compile_cold_warm(env: dict, cache_dir: str) -> dict:
    """Compile twice into the fresh NPU_CACHE_HOME ``cache_dir``: cold, then warm.

    Cold: every cache empty, same as any first-time compile. Warm: only the
    cold compile's kernel objects (NPU_CACHE_HOME/objects) are kept -- the
    whole-build and per-device caches and the outputs are cleared first, so
    the warm compile reruns the full pipeline, which is what an edit to the
    design (not its kernels) costs. The warm compile's build stays in
    ``cache_dir`` for the runs.
    """
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    profile = os.path.join(_LLAMA_DIR, _PROFILE)
    os.makedirs(os.path.join(_LLAMA_DIR, "build"), exist_ok=True)
    build = [
        sys.executable,
        "aie2_decode.py",
        *_SHAPE_ARGS,
        "--full-elf-path",
        _ELF,
    ]
    _clean_build_outputs()
    cold = run_measured(build, _LLAMA_DIR, run_env, COMPILE_TIMEOUT, tree_peak=True)
    stages_cold = stage_rows(
        profile, cold.seconds if cold.process.returncode == 0 else None, ""
    )

    _clean_build_outputs()
    for layer in ("builds", "devices"):
        shutil.rmtree(os.path.join(cache_dir, layer), ignore_errors=True)
    warm = run_measured(build, _LLAMA_DIR, run_env, COMPILE_TIMEOUT, tree_peak=True)
    stages_warm = stage_rows(
        profile, warm.seconds if warm.process.returncode == 0 else None, "_warm"
    )

    return {
        **stages_cold,
        **stages_warm,
        "built": warm.process if warm.process.returncode == 0 else cold.process,
        **compile_stats(cold, warm),
    }


def run_decode(env: dict) -> Measured:
    """Run the checked and timed dispatches at DISPATCH_RATE_AT and CONTEXTS."""
    return run_measured(
        [
            sys.executable,
            "test_chain_persist_grow_mh.py",
            *_SHAPE_ARGS,
            "--elf",
            _ELF,
            "--contexts",
            *(str(at) for at in (DISPATCH_RATE_AT, *CONTEXTS)),
            "--time",
            "--n-iters",
            str(N_ITERS),
            "--n-warmup",
            str(N_WARMUP),
        ],
        _LLAMA_DIR,
        env,
        DECODE_TIMEOUT,
    )


def run_generate(env: dict) -> Measured:
    """Run generate.py --check on GENERATE_PROMPT."""
    return run_measured(
        [
            sys.executable,
            "generate.py",
            f"--t={T}",
            "--elf",
            _ELF,
            "--prompt",
            GENERATE_PROMPT,
            "--max-new",
            str(GENERATE_NEW),
            "--check",
        ],
        _LLAMA_DIR,
        env,
        GENERATE_TIMEOUT,
    )


def run_prefill(env: dict, prompt_tokens: int) -> Measured:
    """Run generate.py --prompt-tokens for its TTFT alone, one generated token."""
    return run_measured(
        [
            sys.executable,
            "generate.py",
            f"--t={T}",
            "--elf",
            _ELF,
            "--prompt-tokens",
            str(prompt_tokens),
            "--max-new",
            "1",
        ],
        _LLAMA_DIR,
        env,
        GENERATE_TIMEOUT,
    )


def parse_prefill(out: str, returncode: int) -> dict:
    """Return a run_prefill run's TTFT, and whether its requests agreed.

    Returns:
        `prefill_passed`: it exited 0 (every repeat generated the same
        token; generate.py's own gate, no oracle here); the TTFT split, as
        `parse_generate`'s.
    """
    timing = _GENERATE_RE.search(out)
    return {
        "prefill_passed": returncode == 0 and bool(timing),
        "ttft_runs": int(timing.group(1)) if timing else None,
        "ttft_ms": float(timing.group(2)) if timing else None,
        "ttft_min_ms": float(timing.group(3)) if timing else None,
        "ttft_npu_dispatch_ms": float(timing.group(4)) if timing else None,
        "ttft_host_ms": float(timing.group(5)) if timing else None,
    }


def run_accuracy_bf16(env: dict, weights: str | None) -> Measured:
    """Run accuracy_vs_hf.py: the device's arithmetic vs the true bf16 model."""
    args = [
        sys.executable,
        "accuracy_vs_hf.py",
        "--max-tokens",
        str(ACCURACY_MAX_TOKENS),
    ]
    if weights:
        args += ["--hf-weights", weights]
    return run_measured(args, _LLAMA_DIR, env, GENERATE_TIMEOUT)


_ACCURACY_TOP1_RE = re.compile(r"next-token top-1 agreement: ([\d.]+)%")
_ACCURACY_KL_RE = re.compile(r"mean KL\(bf16 \|\| int8\)\s*: ([\d.]+)")


def parse_accuracy_bf16(out: str, returncode: int) -> dict:
    """Return accuracy_vs_hf.py's mean KL and top-1 agreement against bf16."""
    top1 = _ACCURACY_TOP1_RE.search(out)
    kl = _ACCURACY_KL_RE.search(out)
    return {
        "accuracy_passed": returncode == 0 and bool(top1) and bool(kl),
        "top1_bf16": float(top1.group(1)) if top1 else None,
        "kl_bf16": float(kl.group(1)) if kl else None,
    }


def parse_generate(out: str, returncode: int) -> dict:
    """Return generate.py --check's numbers and whether its oracle gate passed.

    Args:
        out: Its output.
        returncode: Its exit status.

    Returns:
        Over its requests, the median time to first token, the fastest, and
        the median one's NPU dispatch and host parts (ms), and the median
        decode rate (None when its run printed none: a stop as the first
        token leaves no decode rate); its agreement with numpy_llama_ref, the
        oracle's count of passing generated tokens, and `generate_passed`: it
        exited 0 and the oracle passed every one of at least one generated
        token.
    """
    timing = _GENERATE_RE.search(out)
    check = _CHECK_RE.search(out)
    oracle = _ORACLE_RE.search(out)
    oracle_passed = int(oracle.group(1)) if oracle else 0
    oracle_total = int(oracle.group(2)) if oracle else 0
    return {
        "generate_passed": returncode == 0 and 0 < oracle_passed == oracle_total,
        "oracle_passed": oracle_passed,
        "oracle_total": oracle_total,
        "ttft_runs": int(timing.group(1)) if timing else None,
        "ttft_ms": float(timing.group(2)) if timing else None,
        "ttft_min_ms": float(timing.group(3)) if timing else None,
        "ttft_npu_dispatch_ms": float(timing.group(4)) if timing else None,
        "ttft_host_ms": float(timing.group(5)) if timing else None,
        "decode_tok_s": (
            float(timing.group(6)) if timing and timing.group(6) else None
        ),
        "greedy_agree": int(check.group(1)) if check else None,
        "greedy_total": int(check.group(2)) if check else None,
    }


def parse_correctness(out: str, returncode: int) -> dict:
    tokens = _TOKEN_LINE_RE.findall(out)
    summary = _SUMMARY_RE.search(out)
    passed_n = (
        int(summary.group(1)) if summary else sum(1 for t in tokens if t == "PASS")
    )
    total_n = int(summary.group(2)) if summary else len(tokens)
    return {
        "passed": returncode == 0 and bool(summary) and 0 < passed_n == total_n,
        "tokens_passed": passed_n,
        "tokens_total": total_n,
    }


def artifact_stats() -> dict:
    """Sizes of the files this build actually produced.

    The per-core ELFs are linked into the PDI and not kept on disk, so the
    PDI's size stands in for them.
    """
    paths = {
        "elf_bytes": os.path.join(_LLAMA_DIR, _ELF),
        "pdi_bytes": os.path.join(_LLAMA_DIR, _PROJ_DIR, "main.pdi"),
    }
    return {
        k: os.path.getsize(p) if os.path.exists(p) else None for k, p in paths.items()
    }


def parse_timing(out: str, returncode: int) -> dict:
    """Return the timed dispatches' statistics by position, none unless it exited 0.

    Returns:
        Per position of a dispatch's first token: its iterations, fastest,
        median and median absolute deviation (ms), tokens per dispatch, and
        the fastest per token (ms) and its rate (tok/s).
    """
    if returncode != 0:
        return {}
    return {
        int(m[0]): {
            "n_iters": int(m[1]),
            "min_ms": float(m[2]),
            "median_ms": float(m[3]),
            "mad_ms": float(m[4]),
            "tokens_per_dispatch": int(m[5]),
            "ms_per_token": float(m[6]),
            "tok_s": float(m[7]),
        }
        for m in _DECODE_RE.findall(out)
    }


def timing_sane(timing: dict) -> bool:
    """Return whether the timed dispatches' numbers are ones a working clock gives.

    Args:
        timing: parse_timing's.

    Returns:
        True when every timed position ran every requested iteration, with
        positive times, the fastest no slower than the median.
    """
    return all(
        t["n_iters"] == N_ITERS and 0 < t["min_ms"] <= t["median_ms"]
        for t in timing.values()
    )


def aggregate(
    correctness: dict,
    timing: dict,
    generated: dict,
    prefill: dict,
    accuracy: dict,
    artifacts: dict,
    compile_stats: dict,
) -> list[dict]:
    case = f"llama/persist_grow/N={N_LAYERS}"
    out = [
        row(
            case,
            "tokens_passed",
            correctness["tokens_passed"],
            "tokens",
            f"of {correctness['tokens_total']}",
        )
    ]
    for at, t in timing.items():
        note = f"fastest of {t['n_iters']}, {t['tokens_per_dispatch']} tokens/dispatch"
        if at == DISPATCH_RATE_AT:
            out.append(row(case, "ms_per_token", t["ms_per_token"], "ms", note))
            out.append(row(case, "tok_s", t["tok_s"], "tok/s", note))
        elif at in CONTEXTS:
            ctx = f"{case}/ctx={at}"
            out.append(row(ctx, "ms_per_token", t["ms_per_token"], "ms", note))
            out.append(row(ctx, "tok_s", t["tok_s"], "tok/s", note))
    for metric, unit in (
        ("ttft_ms", "ms"),
        ("ttft_min_ms", "ms"),
        ("ttft_npu_dispatch_ms", "ms"),
        ("ttft_host_ms", "ms"),
        ("decode_tok_s", "tok/s"),
    ):
        if generated["generate_passed"] and generated[metric] is not None:
            stat = "fastest" if metric == "ttft_min_ms" else "median"
            note = f"{stat} of {generated['ttft_runs']}"
            out.append(row(case, metric, generated[metric], unit, note))
    if generated["generate_passed"] and generated["greedy_agree"] is not None:
        out.append(
            row(
                case,
                "greedy_agree",
                generated["greedy_agree"],
                "tokens",
                f"of {generated['greedy_total']} vs numpy_llama_ref",
            )
        )
    for length, p in prefill.items():
        if not p["prefill_passed"]:
            continue
        prompt = f"{case}/prompt={length}"
        for metric, unit in (
            ("ttft_ms", "ms"),
            ("ttft_min_ms", "ms"),
            ("ttft_npu_dispatch_ms", "ms"),
            ("ttft_host_ms", "ms"),
        ):
            stat = "fastest" if metric == "ttft_min_ms" else "median"
            out.append(
                row(prompt, metric, p[metric], unit, f"{stat} of {p['ttft_runs']}")
            )
    if accuracy.get("accuracy_passed"):
        note = f"{ACCURACY_MAX_TOKENS} tokens, teacher-forced"
        out.append(row(case, "kl_bf16", accuracy["kl_bf16"], "nats", note))
        out.append(row(case, "top1_bf16", accuracy["top1_bf16"], "%", note))
    for metric, unit in (
        ("elf_bytes", "bytes"),
        ("pdi_bytes", "bytes"),
    ):
        if artifacts.get(metric) is not None:
            out.append(row(case, metric, artifacts[metric], unit))
    return out + compile_rows(case, compile_stats)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--weights",
        help="path to model.safetensors (default: $IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/)",
    )
    parser.add_argument(
        "--tokenizer",
        help="path to tokenizer.model (default: beside the weights)",
    )
    add_pmode_argument(parser)
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)

    # Deferred so that the tests can import this module's parsers without an NPU runtime.
    from aie.utils.benchmark import preflight, provenance

    pre = preflight()
    meta = {
        "preflight": dict(vars(pre)),
        "provenance": provenance(device=pre.device, pmode=pre.pmode),
        "n_layers": N_LAYERS,
        "pt": P_MAX,
        "t": T,
        "contexts": [DISPATCH_RATE_AT, *CONTEXTS],
        "prompt_lengths": list(PROMPT_LENGTHS),
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        return refuse(
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'",
            meta,
            args.out,
            args.meta,
        )

    case = f"llama/persist_grow/N={N_LAYERS}"
    write_unfinished(meta, case, args.out, args.meta)
    env = dict(os.environ)
    gen = ensure_data(args.weights, args.tokenizer, env)
    if gen is not None and gen.process.returncode != 0:
        reason = (gen.process.stdout.strip().splitlines() or [""])[-1]
        return refuse(
            f"gen_llama_data.py exited {gen.process.returncode}: {reason}",
            meta,
            args.out,
            args.meta,
        )
    if missing_data():
        return refuse(
            f"data/ lacks {', '.join(missing_data())} after gen_llama_data.py",
            meta,
            args.out,
            args.meta,
        )
    if missing_bf16(args.weights):
        return refuse(
            "accuracy_vs_hf.py's bf16 reference model.safetensors is missing "
            "(pass --weights or set IRON_EXAMPLE_WEIGHTS_DIR)",
            meta,
            args.out,
            args.meta,
        )
    cache_dir = tempfile.mkdtemp(prefix="component-checks-llama-chain-")
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    correctness = parse_correctness("", 1)
    timing: dict = {}
    generated = parse_generate("", 1)
    prefill: dict = {}
    accuracy = parse_accuracy_bf16("", 1)
    artifacts: dict = {}
    try:
        compiled = compile_cold_warm(env, cache_dir)
        if compiled["built"].returncode == 0:
            artifacts = artifact_stats()
            ran = run_decode(run_env).process
            correctness = parse_correctness(ran.stdout, ran.returncode)
            timing = parse_timing(ran.stdout, ran.returncode)
            if correctness["passed"]:
                gen_ran = run_generate(run_env).process
                generated = parse_generate(gen_ran.stdout, gen_ran.returncode)
                for length in PROMPT_LENGTHS:
                    pre_ran = run_prefill(run_env, length).process
                    prefill[length] = parse_prefill(pre_ran.stdout, pre_ran.returncode)
                acc_ran = run_accuracy_bf16(env, args.weights).process
                accuracy = parse_accuracy_bf16(acc_ran.stdout, acc_ran.returncode)
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    timed = set(timing) == {DISPATCH_RATE_AT, *CONTEXTS}
    prefill_timed = set(prefill) == set(PROMPT_LENGTHS) and all(
        p["prefill_passed"] for p in prefill.values()
    )
    passed = (
        correctness["passed"]
        and timed
        and generated["generate_passed"]
        and prefill_timed
        and accuracy["accuracy_passed"]
    )
    failed = [] if passed else [case]

    write(
        args.out,
        aggregate(
            correctness, timing, generated, prefill, accuracy, artifacts, compiled
        ),
    )
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": timing_sane(timing),
            "detail": [
                {
                    "case": case,
                    **correctness,
                    "timing": {str(at): t for at, t in timing.items()},
                    **generated,
                    "prefill": {str(length): p for length, p in prefill.items()},
                    **accuracy,
                    **artifacts,
                    **{k: v for k, v in compiled.items() if k != "built"},
                }
            ],
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
