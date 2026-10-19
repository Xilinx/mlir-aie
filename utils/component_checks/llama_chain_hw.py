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
import json
import os
import re
import subprocess
import sys

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


def build(env: dict) -> subprocess.CompletedProcess:
    return _run(["make", f"CHAIN_MH_N={N_LAYERS}", "chain_persist_grow_mh"], env)


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


def parse_timing(out: str) -> dict:
    per_token = _PER_TOKEN_RE.search(out)
    ratio = _RATIO_RE.search(out)
    return {
        "ms_per_token": float(per_token.group(1)) if per_token else None,
        "tok_s": float(per_token.group(2)) if per_token else None,
        "ratio": float(ratio.group(1)) if ratio else None,
        "bound": ratio.group(2) if ratio else None,
    }


def aggregate(correctness: dict, timing: dict) -> list[dict]:
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
    built = build(env)
    correctness = parse_correctness("", 1)
    timing = parse_timing("")
    log = (gen.stdout + gen.stderr) if gen else ""
    log += built.stdout + built.stderr

    if built.returncode == 0:
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

    write(args.out, aggregate(correctness, timing))
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": True,
            "detail": [{"case": case, **correctness, **timing}],
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
