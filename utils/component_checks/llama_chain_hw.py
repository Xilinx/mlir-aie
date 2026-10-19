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
test_chain_persist_grow_mh.py CLI (no new Python API), building once via the
example's own Makefile and then invoking the test script directly for both
the correctness run and the ``--time`` run, so both share one compiled
xclbin. One run, no seed sweep: unlike mobilenet this design has no SA-
placer-seed dimension (Phase 7 unplaced it onto the plain default
sequential placer, which has no seed concept).

Needs the real Llama 3.2 1B weights staged where gen_llama_data.py can read
them (``--weights``/``--tokenizer``, or $LLAMA_3_2_1B_SAFETENSORS /
$LLAMA_3_2_1B_TOKENIZER): the test loads the real embed table + final_norm
gamma even for a pure timing run (the transformer layers themselves use a
random fixture; only the embed/lm_head weights are real). Generates
``data/`` once if not already there.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
the run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/llama_3_2_1b, with

    make CHAIN_MH_N=16 chain_persist_grow_mh
    python3 test_chain_persist_grow_mh.py \\
        --xclbin build/final_chain_persist_grow_N16_PT4.xclbin \\
        --instr build/insts_chain_persist_grow_N16_PT4.bin -k MLIR_AIE --time
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

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
_XCLBIN = f"build/final_chain_persist_grow_N{N_LAYERS}_PT{PT}.xclbin"
_INSTR = f"build/insts_chain_persist_grow_N{N_LAYERS}_PT{PT}.bin"
_PROJ_DIR = f"build/aie_chain_persist_grow_N{N_LAYERS}_PT{PT}.mlir.prj"

# The env this design's own Makefile sets for this target (see its
# run_chain_persist_grow_mh recipe); the build and the test script both read
# these at import time, so both subprocess calls need the same dict.
_CHAIN_ENV = {
    "LLAMA_CHAIN_SAMPLE": "1",
    "LLAMA_CHAIN_ONESTREAM": "1",
    "LLAMA_CHAIN_PERSIST": "1",
    "LLAMA_CHAIN_PERSIST_GROW": "1",
    "LLAMA_CHAIN_PT": str(PT),
    "LLAMA_CHAIN_ONESTREAM_KSET": "8",
    "LLAMA_SAMPLE_TEMP": "0.0",
    "LLAMA_SAMPLE_TOPK": "0",
    "LLAMA_SAMPLE_SEED": "0",
    "LLAMA_CHAIN_N": str(N_LAYERS),
    "LLAMA_CHAIN_T": "128",
}


def _env(weights: str | None, tokenizer: str | None) -> dict:
    env = dict(os.environ, **_CHAIN_ENV)
    if weights:
        env["LLAMA_3_2_1B_SAFETENSORS"] = weights
    if tokenizer:
        env["LLAMA_3_2_1B_TOKENIZER"] = tokenizer
    return env


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
    proc = subprocess.Popen(
        ["make", f"CHAIN_MH_N={N_LAYERS}", "chain_persist_grow_mh"],
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
    """Remove this config's generated MLIR/project/xclbin/instr.

    Unlike mobilenet's CLI (which takes a fresh --xclbin-path/--insts-path
    per run), this design's Makefile always writes into the persistent
    build/ dir, so a fresh NPU_CACHE_HOME alone wouldn't force Make to
    recompile -- it would just see the existing xclbin as up to date.
    """
    for rel in (
        _XCLBIN,
        _INSTR,
        f"build/aie_chain_persist_grow_N{N_LAYERS}_PT{PT}.mlir",
    ):
        path = os.path.join(_LLAMA_DIR, rel)
        if os.path.exists(path):
            os.remove(path)
    proj = os.path.join(_LLAMA_DIR, _PROJ_DIR)
    if os.path.isdir(proj):
        shutil.rmtree(proj)


def compile_cold_warm(env: dict) -> dict:
    """Compile twice into one fresh NPU_CACHE_HOME: cold, then warm.

    Cold: kernel objects built from scratch, same as any first-time compile.
    Warm: immediately after, same cache_dir, so it reuses the cold compile's
    kernel objects -- the Makefile's own output is cleaned before each
    compile so both runs go through the full pipeline (placement + xclbin
    assembly), not just a Make no-op. See sa_placer_hw.py's run_one for the
    mobilenet analogue.
    """
    cache_dir = tempfile.mkdtemp(prefix="component-checks-llama-chain-")
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    try:
        _clean_build_outputs()
        start = time.perf_counter()
        built_cold, rss_cold = _build_rss(run_env)
        compile_s = time.perf_counter() - start

        _clean_build_outputs()
        start = time.perf_counter()
        built_warm, rss_warm = _build_rss(run_env)
        compile_s_warm = time.perf_counter() - start
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    return {
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
            "--xclbin",
            _XCLBIN,
            "--instr",
            _INSTR,
            "-k",
            "MLIR_AIE",
        ],
        env,
    )


def run_timed(env: dict) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "test_chain_persist_grow_mh.py",
            "--xclbin",
            _XCLBIN,
            "--instr",
            _INSTR,
            "-k",
            "MLIR_AIE",
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

    Per-core ELFs are counted from the project dir rather than the xclbin
    (which packs them compressed), so `core_elfs_bytes` reflects what Peano
    + the chess backend generated, not the shipped payload size.
    """
    xclbin = os.path.join(_LLAMA_DIR, _XCLBIN)
    instr = os.path.join(_LLAMA_DIR, _INSTR)
    elfs = glob.glob(os.path.join(_LLAMA_DIR, _PROJ_DIR, "**", "*.elf"), recursive=True)
    return {
        "xclbin_bytes": os.path.getsize(xclbin) if os.path.exists(xclbin) else None,
        "instr_bytes": os.path.getsize(instr) if os.path.exists(instr) else None,
        "n_core_elfs": len(elfs) if elfs else None,
        "core_elfs_bytes": sum(os.path.getsize(f) for f in elfs) if elfs else None,
    }


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
    if artifacts.get("xclbin_bytes") is not None:
        out.append(
            {
                "name": f"{case}/xclbin_bytes",
                "unit": "bytes",
                "value": artifacts["xclbin_bytes"],
            }
        )
    if artifacts.get("instr_bytes") is not None:
        out.append(
            {
                "name": f"{case}/instr_bytes",
                "unit": "bytes",
                "value": artifacts["instr_bytes"],
            }
        )
    if artifacts.get("n_core_elfs") is not None:
        out.append(
            {
                "name": f"{case}/n_core_elfs",
                "unit": "elfs",
                "value": artifacts["n_core_elfs"],
            }
        )
    if artifacts.get("core_elfs_bytes") is not None:
        out.append(
            {
                "name": f"{case}/core_elfs_bytes",
                "unit": "bytes",
                "value": artifacts["core_elfs_bytes"],
                "range": f"across {artifacts['n_core_elfs']} ELFs",
            }
        )
    # Reuses sa_placer_hw.py's metric names (compile_s, compile_peak_rss_mb,
    # ...) so the dashboard's existing labels/tooltips/thresholds for them
    # apply here with no new Layer-3 vocabulary.
    for metric in (
        "compile_s",
        "compile_peak_rss_mb",
        "compile_s_warm",
        "compile_peak_rss_mb_warm",
    ):
        if compile_stats.get(metric) is not None:
            unit = "s" if metric.startswith("compile_s") else "MB"
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
        help="path to model.safetensors (overrides $LLAMA_3_2_1B_SAFETENSORS)",
    )
    parser.add_argument(
        "--tokenizer",
        help="path to tokenizer.model (overrides $LLAMA_3_2_1B_TOKENIZER)",
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

    env = _env(args.weights, args.tokenizer)
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
