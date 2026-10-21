#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Build and run the Llama 3.2 1B decode on NPU2.

Does the full N=16 persistent, growing-KV decode still build, match its
numpy oracle and run at the expected rate on real hardware. Compiles once
with the design's own CLI (aie2_decode.py), then runs that full ELF through
test_chain_persist_grow_mh.py (every token, KV row and final residual
against the oracle, then ``--time``) and generate.py ``--check`` (a prompt
in, tokens out, each generated token the oracle's greedy pick). Every run
must exit 0, and generate.py's ``oracle:`` line must pass every token. One
run, no seed sweep: the design pins its SA placer seed (aie2_decode.build's
aiecc flags).

Needs the real Llama 3.2 1B checkpoint where amd/IRON's examples keep it,
``$IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/`` (default ``/srv``; see
gen_llama_data.WEIGHTS_DIR), or ``--weights``/``--tokenizer``, unless
``data/`` already holds the INT8 weights and the tokenizer: the decode test
loads the real embed table and final_norm gamma even for its timing (its
transformer layers are a random fixture), and generate.py loads every
layer. Generates ``data/`` once if it isn't complete, and refuses to measure
when that fails.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
any run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/llama_3_2_1b, with

    python3 test_chain_persist_grow_mh.py --n-layers 16 --time
    python3 generate.py --prompt "The capital of France is" --check
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

from aiecc_stages import (
    PROFILE_NAME,
    add_pmode_argument,
    compile_rows,
    compile_stats,
    refuse,
    row,
    run_measured,
    stage_rows,
    write,
)

N_LAYERS = 16
P_MAX = 4
N_ITERS = 20
N_WARMUP = 10

_LLAMA_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml", "llama_3_2_1b"
)

_TOKEN_LINE_RE = re.compile(
    r"^\s*token \d+(?: \(pos \d+\))?: (PASS|FAIL)", re.MULTILINE
)
_SUMMARY_RE = re.compile(
    r"^chain_persist_grow_mh: (\d+)/(\d+) tokens PASS", re.MULTILINE
)
_PER_TOKEN_RE = re.compile(r"per-token:\s+([\d.]+) ms\s+\(([\d.]+) tok/s\)")
_DISPATCH_RE = re.compile(r"per-dispatch: n=(\d+) median=([\d.]+) ms min=([\d.]+)")
_RATIO_RE = re.compile(r"ratio:\s+([\d.]+)x\s+->\s+(\S+)")
_GENERATE_RE = re.compile(
    r"TTFT ([\d.]+) ms \(([\d.]+) prompt tok/s\), decode (?:([\d.]+) tok/s|n/a)"
)
_CHECK_RE = re.compile(r"check: (\d+)/(\d+) tokens")
_ORACLE_RE = re.compile(r"^oracle: (\d+)/(\d+) generated tokens pass", re.MULTILINE)
# generate.py's prompt and length, the same every night.
GENERATE_PROMPT = "The capital of France is"
GENERATE_NEW = 32
_ELF = f"build/chain_persist_grow_N{N_LAYERS}_PT{P_MAX}.elf"
_PROJ_DIR = f"build/chain_persist_grow_N{N_LAYERS}_PT{P_MAX}.prj"
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
_SHAPE_ARGS = [f"--p-max={P_MAX}", "--kset=8", f"--n-layers={N_LAYERS}", "--t=128"]


def _run(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=_LLAMA_DIR, env=env, capture_output=True, text=True)


def missing_data() -> list[str]:
    """Return the files in data/ the runs read that it lacks."""
    return [
        f
        for f in DATA_FILES
        if not os.path.exists(os.path.join(_LLAMA_DIR, "data", f))
    ]


def ensure_data(
    weights: str | None, tokenizer: str | None, env: dict
) -> subprocess.CompletedProcess | None:
    """Generate data/ once, if this checkout doesn't already have all of it."""
    if not missing_data():
        return None
    args = [sys.executable, "gen_llama_data.py"]
    if weights:
        args += ["--weights", weights]
    if tokenizer:
        args += ["--tokenizer", tokenizer]
    return _run(args, env)


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


def compile_cold_warm(env: dict) -> dict:
    """Compile twice into one fresh NPU_CACHE_HOME: cold, then warm.

    Cold: every cache empty, same as any first-time compile. Warm: only the
    cold compile's kernel objects (NPU_CACHE_HOME/objects) are kept -- the
    whole-build and per-device caches and the outputs are cleared first, so
    the warm compile reruns the full pipeline, which is what an edit to the
    design (not its kernels) costs.
    """
    cache_dir = tempfile.mkdtemp(prefix="component-checks-llama-chain-")
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    profile = os.path.join(_LLAMA_DIR, _PROFILE)
    try:
        os.makedirs(os.path.join(_LLAMA_DIR, "build"), exist_ok=True)
        build = [
            sys.executable,
            "aie2_decode.py",
            *_SHAPE_ARGS,
            "--full-elf-path",
            _ELF,
        ]
        _clean_build_outputs()
        cold = run_measured(build, _LLAMA_DIR, run_env)
        stages_cold = stage_rows(
            profile, cold[1] if cold[0].returncode == 0 else None, ""
        )

        _clean_build_outputs()
        for layer in ("builds", "devices"):
            shutil.rmtree(os.path.join(cache_dir, layer), ignore_errors=True)
        warm = run_measured(build, _LLAMA_DIR, run_env)
        stages_warm = stage_rows(
            profile, warm[1] if warm[0].returncode == 0 else None, "_warm"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return {
        **stages_cold,
        **stages_warm,
        "built": warm[0] if warm[0].returncode == 0 else cold[0],
        **compile_stats(cold, warm),
        "log": cold[0].stdout + warm[0].stdout,
    }


def run_correctness(env: dict) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "test_chain_persist_grow_mh.py",
            *_SHAPE_ARGS,
            "--elf",
            _ELF,
        ],
        env,
    )


def run_timed(env: dict) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "test_chain_persist_grow_mh.py",
            *_SHAPE_ARGS,
            "--elf",
            _ELF,
            "--time",
            "--n-iters",
            str(N_ITERS),
            "--n-warmup",
            str(N_WARMUP),
        ],
        env,
    )


def run_generate(env: dict) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "generate.py",
            "--elf",
            _ELF,
            "--prompt",
            GENERATE_PROMPT,
            "--max-new",
            str(GENERATE_NEW),
            "--check",
        ],
        env,
    )


def parse_generate(out: str, returncode: int) -> dict:
    """Return generate.py --check's numbers and whether its oracle gate passed.

    Args:
        out: Its output.
        returncode: Its exit status.

    Returns:
        TTFT, the prompt and decode rates (None when its run printed none:
        a stop as the first token leaves no decode rate), its agreement with
        numpy_llama_ref, the oracle's count of passing generated tokens, and
        `generate_passed`: it exited 0 and the oracle passed every one of at
        least one generated token.
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
        "ttft_ms": float(timing.group(1)) if timing else None,
        "prompt_tok_s": float(timing.group(2)) if timing else None,
        "decode_tok_s": (
            float(timing.group(3)) if timing and timing.group(3) else None
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
    """Return the timed decode's numbers, none unless it exited 0."""
    per_token = _PER_TOKEN_RE.search(out) if returncode == 0 else None
    dispatch = _DISPATCH_RE.search(out) if returncode == 0 else None
    ratio = _RATIO_RE.search(out) if returncode == 0 else None
    return {
        "ms_per_token": float(per_token.group(1)) if per_token else None,
        "tok_s": float(per_token.group(2)) if per_token else None,
        "n_iters": int(dispatch.group(1)) if dispatch else None,
        "median_ms": float(dispatch.group(2)) if dispatch else None,
        "min_ms": float(dispatch.group(3)) if dispatch else None,
        "ratio": float(ratio.group(1)) if ratio else None,
        "bound": ratio.group(2) if ratio else None,
    }


def timing_sane(timing: dict) -> bool:
    """Return whether the timed decode's numbers are ones a working clock gives.

    Args:
        timing: parse_timing's.

    Returns:
        True when nothing was timed, or when every requested iteration was
        timed, with positive times, the minimum no more than the median.
    """
    if timing["ms_per_token"] is None:
        return True
    return (
        timing["n_iters"] == N_ITERS
        and timing["min_ms"] is not None
        and 0 < timing["min_ms"] <= timing["median_ms"]
        and timing["ms_per_token"] > 0
    )


def aggregate(
    correctness: dict,
    timing: dict,
    generated: dict,
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
    if timing["tok_s"] is not None:
        out.append(row(case, "tok_s", timing["tok_s"], "tok/s"))
    if timing["ms_per_token"] is not None:
        out.append(row(case, "ms_per_token", timing["ms_per_token"], "ms"))
    if timing["ratio"] is not None:
        out.append(row(case, "dma_floor_ratio", timing["ratio"], "x", timing["bound"]))
    for metric, unit in (
        ("ttft_ms", "ms"),
        ("prompt_tok_s", "tok/s"),
        ("decode_tok_s", "tok/s"),
    ):
        if generated["generate_passed"] and generated[metric] is not None:
            out.append(row(case, metric, generated[metric], unit))
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
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        return refuse(
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'",
            meta,
            args.out,
            args.meta,
        )

    env = dict(os.environ)
    gen = ensure_data(args.weights, args.tokenizer, env)
    log = (gen.stdout + gen.stderr) if gen else ""
    if gen is not None and gen.returncode != 0:
        print(log)
        reason = (gen.stderr.strip().splitlines() or [""])[-1]
        return refuse(
            f"gen_llama_data.py exited {gen.returncode}: {reason}",
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
    compiled = compile_cold_warm(env)
    built = compiled["built"]
    correctness = parse_correctness("", 1)
    timing = parse_timing("", 1)
    generated = parse_generate("", 1)
    artifacts: dict = {}
    log += compiled["log"]
    timed_ok = True

    if built.returncode == 0:
        artifacts = artifact_stats()
        ran = run_correctness(env)
        correctness = parse_correctness(ran.stdout + ran.stderr, ran.returncode)
        log += ran.stdout + ran.stderr
        if correctness["passed"]:
            timed = run_timed(env)
            timing = parse_timing(timed.stdout + timed.stderr, timed.returncode)
            timed_ok = timed.returncode == 0 and timing["ms_per_token"] is not None
            log += timed.stdout + timed.stderr
            gen_ran = run_generate(env)
            generated = parse_generate(
                gen_ran.stdout + gen_ran.stderr, gen_ran.returncode
            )
            log += gen_ran.stdout + gen_ran.stderr

    print(log)
    case = f"llama/persist_grow/N={N_LAYERS}"
    passed = correctness["passed"] and timed_ok and generated["generate_passed"]
    failed = [] if passed else [case]

    write(args.out, aggregate(correctness, timing, generated, artifacts, compiled))
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": timing_sane(timing),
            "detail": [
                {
                    "case": case,
                    **correctness,
                    **timing,
                    **generated,
                    **artifacts,
                    **{k: v for k, v in compiled.items() if k not in ("built", "log")},
                }
            ],
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
