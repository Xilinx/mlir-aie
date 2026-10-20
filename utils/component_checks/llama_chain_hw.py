#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Build + run the Llama 3.2 1B persistent-decode capstone on NPU2.

Complements the kernel-level checks (each of RMSNorm/SiLU/RoPE/FlowKV/the
top-k sampler has its own isolated unit test) with the thing that actually
matters end to end: does the full N=16 persistent, growing-KV decode loop
still build, verify (device token == numpy-fixture argmax, same oracle
test_chain_persist_grow_mh.py already uses), and run at the expected
throughput on real hardware. Shells out to the existing
test_chain_persist_grow_mh.py CLI (no new Python API), compiling once with the
design's own CLI (aie2_chain_dynscale_mh.py) and then invoking the test script
with that full ELF for both the correctness run and the ``--time`` run. One run, no seed sweep: unlike mobilenet this design has no SA-
placer-seed dimension (Phase 7 unplaced it onto the plain default
sequential placer, which has no seed concept).

Needs the real Llama 3.2 1B checkpoint where amd/IRON's examples keep it,
``$IRON_EXAMPLE_WEIGHTS_DIR/llama3.2-1b/`` (default ``/srv``; see
gen_llama_data.WEIGHTS_DIR), or ``--weights``/``--tokenizer``: the test loads
the real embed table + final_norm
gamma even for a pure timing run (the transformer layers themselves use a
random fixture; only the embed/lm_head weights are real). Generates
``data/`` once if not already there.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
the run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/llama_3_2_1b, with

    LLAMA_CHAIN_N=16 python3 test_chain_persist_grow_mh.py --time
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from aiecc_stages import PROFILE_NAME, STAGE_METRICS, stage_rows

N_LAYERS = 16
PT = 4
N_ITERS = 3
N_WARMUP = 1

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
_RATIO_RE = re.compile(r"ratio:\s+([\d.]+)x\s+->\s+(\S+)")
_ELF = f"build/chain_persist_grow_N{N_LAYERS}_PT{PT}.elf"
_PROJ_DIR = f"build/chain_persist_grow_N{N_LAYERS}_PT{PT}.prj"
_PROFILE = os.path.join(_PROJ_DIR, PROFILE_NAME)
_CORE_RE = re.compile(r"\baie\.core\(")

# The design's CompileTime values; the test script reads the same
# configuration from _CHAIN_ENV at import time.
_BUILD_ARGS = [
    "--sample=1",
    "--onestream=1",
    "--persist=1",
    "--persist-grow=1",
    f"--pt={PT}",
    "--onestream-kset=8",
    f"--n-layers={N_LAYERS}",
    "--t=128",
]
_CHAIN_ENV = {
    "LLAMA_CHAIN_PT": str(PT),
    "LLAMA_CHAIN_ONESTREAM_KSET": "8",
    "LLAMA_CHAIN_N": str(N_LAYERS),
    "LLAMA_CHAIN_T": "128",
}


def _run(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=_LLAMA_DIR, env=env, capture_output=True, text=True)


def ensure_data(
    weights: str | None, tokenizer: str | None, env: dict
) -> subprocess.CompletedProcess | None:
    """Generate data/ once, if this checkout doesn't already have it."""
    if os.path.exists(os.path.join(_LLAMA_DIR, "data", "embed.i8.bin")):
        return None
    args = [sys.executable, "gen_llama_data.py"]
    if weights:
        args += ["--weights", weights]
    if tokenizer:
        args += ["--tokenizer", tokenizer]
    return _run(args, env)


def _build_rss(env: dict) -> tuple[subprocess.CompletedProcess, float]:
    """Like ``build``, plus the process's own peak RSS in MB.

    Uses os.wait4 (not resource.getrusage(RUSAGE_CHILDREN), a running max
    across every reaped child), so this call's peak RSS is its own -- see
    sa_placer_hw.py's _mobilenet_rss, which this mirrors.
    """
    os.makedirs(os.path.join(_LLAMA_DIR, "build"), exist_ok=True)
    proc = subprocess.Popen(
        [
            sys.executable,
            "aie2_chain_dynscale_mh.py",
            *_BUILD_ARGS,
            "--full-elf-path",
            _ELF,
        ],
        cwd=_LLAMA_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    out = proc.stdout.read() if proc.stdout else ""
    _, status, ru = os.wait4(proc.pid, 0)
    result = subprocess.CompletedProcess(
        proc.args, os.waitstatus_to_exitcode(status), out, ""
    )
    return result, ru.ru_maxrss / 1024.0


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
        _clean_build_outputs()
        start = time.perf_counter()
        built_cold, rss_cold = _build_rss(run_env)
        compile_s = time.perf_counter() - start
        stages_cold = stage_rows(
            profile, compile_s if built_cold.returncode == 0 else None, ""
        )

        _clean_build_outputs()
        for layer in ("builds", "devices"):
            shutil.rmtree(os.path.join(cache_dir, layer), ignore_errors=True)
        start = time.perf_counter()
        built_warm, rss_warm = _build_rss(run_env)
        compile_s_warm = time.perf_counter() - start
        stages_warm = stage_rows(
            profile, compile_s_warm if built_warm.returncode == 0 else None, "_warm"
        )
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return {
        **stages_cold,
        **stages_warm,
        "built": built_warm if built_warm.returncode == 0 else built_cold,
        "compile_s": round(compile_s, 1) if built_cold.returncode == 0 else None,
        "compile_peak_rss_mb": (
            round(rss_cold, 1) if built_cold.returncode == 0 else None
        ),
        "compile_s_warm": (
            round(compile_s_warm, 1) if built_warm.returncode == 0 else None
        ),
        "compile_peak_rss_mb_warm": (
            round(rss_warm, 1) if built_warm.returncode == 0 else None
        ),
        "log": built_cold.stdout
        + built_cold.stderr
        + built_warm.stdout
        + built_warm.stderr,
    }


def run_correctness(env: dict) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "test_chain_persist_grow_mh.py",
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


def parse_correctness(out: str, returncode: int) -> dict:
    tokens = _TOKEN_LINE_RE.findall(out)
    summary = _SUMMARY_RE.search(out)
    passed_n = (
        int(summary.group(1)) if summary else sum(1 for t in tokens if t == "PASS")
    )
    total_n = int(summary.group(2)) if summary else len(tokens)
    return {
        "passed": returncode == 0 and bool(summary) and passed_n == total_n,
        "tokens_passed": passed_n,
        "tokens_total": total_n,
    }


def artifact_stats() -> dict:
    """Sizes/counts of the files this build actually produced.

    The per-core ELFs are linked into the PDI and not kept on disk, so the
    PDI's size stands in for them, and the placed MLIR gives the core count.
    """
    paths = {
        "elf_bytes": os.path.join(_LLAMA_DIR, _ELF),
        "pdi_bytes": os.path.join(_LLAMA_DIR, _PROJ_DIR, "main.pdi"),
    }
    out: dict = {
        k: os.path.getsize(p) if os.path.exists(p) else None for k, p in paths.items()
    }
    placed = os.path.join(_LLAMA_DIR, _PROJ_DIR, "input_with_addresses.mlir")
    if os.path.exists(placed):
        with open(placed) as f:
            out["n_cores"] = len(_CORE_RE.findall(f.read()))
    else:
        out["n_cores"] = None
    return out


def parse_timing(out: str) -> dict:
    per_token = _PER_TOKEN_RE.search(out)
    ratio = _RATIO_RE.search(out)
    return {
        "ms_per_token": float(per_token.group(1)) if per_token else None,
        "tok_s": float(per_token.group(2)) if per_token else None,
        "ratio": float(ratio.group(1)) if ratio else None,
        "bound": ratio.group(2) if ratio else None,
    }


def aggregate(
    correctness: dict, timing: dict, artifacts: dict, compile_stats: dict
) -> list[dict]:
    case = f"llama/persist_grow/N={N_LAYERS}"
    out = [
        {
            "name": f"{case}/tokens_passed",
            "unit": "tokens",
            "value": correctness["tokens_passed"],
            "range": f"of {correctness['tokens_total']}",
        }
    ]
    if timing["tok_s"] is not None:
        out.append({"name": f"{case}/tok_s", "unit": "tok/s", "value": timing["tok_s"]})
    if timing["ms_per_token"] is not None:
        out.append(
            {
                "name": f"{case}/ms_per_token",
                "unit": "ms",
                "value": timing["ms_per_token"],
            }
        )
    if timing["ratio"] is not None:
        out.append(
            {
                "name": f"{case}/dma_floor_ratio",
                "unit": "x",
                "value": timing["ratio"],
                "range": timing["bound"],
            }
        )
    for metric, unit in (
        ("elf_bytes", "bytes"),
        ("pdi_bytes", "bytes"),
        ("n_cores", "cores"),
    ):
        if artifacts.get(metric) is not None:
            out.append(
                {"name": f"{case}/{metric}", "unit": unit, "value": artifacts[metric]}
            )
    # Reuses sa_placer_hw.py's metric names (compile_s, compile_peak_rss_mb,
    # ...) so the dashboard's existing labels/tooltips/thresholds for them
    # apply here with no new Layer-3 vocabulary.
    for metric in (
        "compile_s",
        "compile_peak_rss_mb",
        "compile_s_warm",
        "compile_peak_rss_mb_warm",
        *STAGE_METRICS,
        *(f"{m}_warm" for m in STAGE_METRICS),
    ):
        if compile_stats.get(metric) is not None:
            unit = "MB" if "_rss_" in metric else "s"
            out.append(
                {
                    "name": f"{case}/{metric}",
                    "unit": unit,
                    "value": compile_stats[metric],
                }
            )
    return out


def write(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=1)


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
    parser.add_argument(
        "--pmode",
        default="any",
        help="refuse to measure unless the NPU is in this power mode (default: any)",
    )
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)

    from aie.utils.benchmark import preflight, provenance

    pre = preflight()
    meta = {
        "preflight": dict(vars(pre)),
        "provenance": provenance(device=pre.device, pmode=pre.pmode),
        "n_layers": N_LAYERS,
        "pt": PT,
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        meta["refused"] = (
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'"
        )
        print(f"refusing to measure: {meta['refused']}")
        write(args.out, [])
        write(args.meta, {**meta, "failed": [], "exitstatus": 1})
        return 1

    env = dict(os.environ, **_CHAIN_ENV)
    gen = ensure_data(args.weights, args.tokenizer, env)
    compiled = compile_cold_warm(env)
    built = compiled["built"]
    correctness = parse_correctness("", 1)
    timing = parse_timing("")
    artifacts: dict = {}
    log = (gen.stdout + gen.stderr) if gen else ""
    log += compiled["log"]

    if built.returncode == 0:
        artifacts = artifact_stats()
        ran = run_correctness(env)
        correctness = parse_correctness(ran.stdout + ran.stderr, ran.returncode)
        log += ran.stdout + ran.stderr
        if correctness["passed"]:
            timed = run_timed(env)
            timing = parse_timing(timed.stdout + timed.stderr)
            log += timed.stdout + timed.stderr

    print(log)
    case = f"llama/persist_grow/N={N_LAYERS}"
    failed = [] if correctness["passed"] else [case]

    write(args.out, aggregate(correctness, timing, artifacts, compiled))
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": True,
            "detail": [
                {
                    "case": case,
                    **correctness,
                    **timing,
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
