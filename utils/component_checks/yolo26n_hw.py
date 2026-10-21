#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Build + run the YOLO26n 640x640 detector on NPU2 over real COCO images.

The whole detector end to end: does it still build, does every image's
detection match the numpy oracle (and, as information, onnxruntime), and how
long one image takes at batch 1 and at batch 8: the fastest of ``N_ITERS``
calls after ``N_WARMUP``, host wall time of the whole ``yolo26n(...)`` call,
with the median and MAD beside it. Shells out to the example's
own yolo26n.py CLI (no new Python API): ``--compile-only`` twice into one
fresh NPU_CACHE_HOME, cold then warm, keeping only the cold compile's kernel
objects for the warm one, as llama_chain_hw.py does; then a ``--time`` run
from that cache and one at ``--batch 8``, each checking every image, their
timing kept only when every image matched in both.

Needs the quantized model and the COCO128 images where amd/IRON's examples
keep them, ``$IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/`` (default ``/srv``), or
``--weights-dir``: ``yolo26n_xint8.onnx`` and ``coco128/images/train2017``.
Generates ``data/`` once if not already there.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
the run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/yolo26n, with

    python yolo26n.py --images $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/coco128/images/train2017 --n 8
"""

import argparse
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

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

CASE = "yolo26n/640"
N_ITERS = 20
N_WARMUP = 10
BATCH = 8

_YOLO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml", "yolo26n"
)
_PARAMS = os.path.join(_YOLO_DIR, "data", "yolo26n_params.npz")

_IMAGE_LINE_RE = re.compile(
    r"^image (.+): (PASS|FAIL) \((\d+)/(\d+) differ\)$", re.MULTILINE
)
_SUMMARY_RE = re.compile(r"^yolo26n: (\d+)/(\d+) images PASS", re.MULTILINE)
_ORT_RE = re.compile(r"^ort: (\d+)/(\d+) images match onnxruntime", re.MULTILINE)
_PER_IMAGE_RE = re.compile(
    r"^per-image: (?P<us>[\d.]+) us \((?P<fps>[\d.]+) fps\) at batch (?P<batch>\d+), "
    r"fastest of (?P<n>\d+) after \d+ warmups; "
    r"median (?P<median>[\d.]+) us, MAD (?P<mad>[\d.]+) us$",
    re.MULTILINE,
)


def _run(cmd: list[str], env: dict) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=_YOLO_DIR, env=env, capture_output=True, text=True)


def ensure_data(onnx: Path, env: dict) -> subprocess.CompletedProcess | None:
    """Generate data/ once, if this checkout doesn't already have it."""
    if os.path.exists(_PARAMS):
        return None
    return _run([sys.executable, "gen_yolo_data.py", "--onnx", str(onnx)], env)


def newest_build(cache_dir: str) -> str | None:
    """Return the JIT build directory under ``cache_dir`` aiecc last profiled.

    Args:
        cache_dir: an NPU_CACHE_HOME; each build is ``<cache_dir>/<hash>/``.

    Returns:
        The directory holding the newest ``aiecc_profile.json``, or None when
        no build wrote one.
    """
    profiles = glob.glob(os.path.join(cache_dir, "*", PROFILE_NAME))
    if not profiles:
        return None
    return os.path.dirname(max(profiles, key=os.path.getmtime))


def compile_cold_warm(images: Path, onnx: Path, env: dict, cache_dir: str) -> dict:
    """Compile twice into the fresh NPU_CACHE_HOME ``cache_dir``: cold, then warm.

    Cold: every cache empty, same as any first-time compile. Warm: only the
    cold compile's kernel objects (``objects/``) are kept, so the warm
    compile reruns the full pipeline, which is what an edit to the design
    (not its kernels) costs.
    """
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    compile_only = [
        sys.executable,
        "yolo26n.py",
        "--compile-only",
        "--images",
        str(images),
        "--onnx",
        str(onnx),
    ]
    cold = run_measured(compile_only, _YOLO_DIR, run_env)
    build = newest_build(cache_dir)
    stages_cold = stage_rows(
        os.path.join(build or cache_dir, PROFILE_NAME),
        cold[1] if cold[0].returncode == 0 else None,
        "",
    )

    for entry in os.listdir(cache_dir):
        if entry != "objects":
            shutil.rmtree(os.path.join(cache_dir, entry), ignore_errors=True)
    warm = run_measured(compile_only, _YOLO_DIR, run_env)
    build = newest_build(cache_dir)
    stages_warm = stage_rows(
        os.path.join(build or cache_dir, PROFILE_NAME),
        warm[1] if warm[0].returncode == 0 else None,
        "_warm",
    )

    return {
        **stages_cold,
        **stages_warm,
        "built": warm[0] if warm[0].returncode == 0 else cold[0],
        "build_dir": build,
        **compile_stats(cold, warm),
        "log": cold[0].stdout + warm[0].stdout,
    }


def run_timed(
    images: Path, onnx: Path, n_images: int, batch: int, env: dict
) -> subprocess.CompletedProcess:
    return _run(
        [
            sys.executable,
            "yolo26n.py",
            "--images",
            str(images),
            "--onnx",
            str(onnx),
            "--n",
            str(n_images),
            "--batch",
            str(batch),
            "--time",
            "--n-iters",
            str(N_ITERS),
            "--n-warmup",
            str(N_WARMUP),
        ],
        env,
    )


def parse_correctness(out: str, returncode: int) -> dict:
    """Read a yolo26n.py run: how many images matched the oracle and onnxruntime.

    Args:
        out: the run's stdout and stderr.
        returncode: its exit code.

    Returns:
        ``passed`` (every image matched, by the run's own summary line),
        ``images_passed``, ``images_total``, ``ort_matches`` (None when the
        run printed no onnxruntime line) and ``failed_images``, by name.
    """
    images = _IMAGE_LINE_RE.findall(out)
    summary = _SUMMARY_RE.search(out)
    ort = _ORT_RE.search(out)
    passed_n = (
        int(summary.group(1))
        if summary
        else sum(1 for _, verdict, _, _ in images if verdict == "PASS")
    )
    total_n = int(summary.group(2)) if summary else len(images)
    return {
        "passed": returncode == 0
        and bool(summary)
        and total_n > 0
        and passed_n == total_n,
        "images_passed": passed_n,
        "images_total": total_n,
        "ort_matches": int(ort.group(1)) if ort else None,
        "failed_images": [name for name, verdict, _, _ in images if verdict == "FAIL"],
    }


def parse_timing(out: str) -> dict:
    """Read a ``--time`` run: per batch, one image's time and the frame rate.

    Args:
        out: the run's stdout and stderr.

    Returns:
        For each batch ``B`` the run timed, in the order it printed them:
        ``us_per_image_b<B>`` (the fastest call, per image), ``fps_b<B>``
        (from it), ``timed_calls_b<B>``, and the per-image
        ``median_us_per_image_b<B>`` and ``mad_us_per_image_b<B>``. Empty
        when it timed none.
    """
    timing: dict = {}
    for m in _PER_IMAGE_RE.finditer(out):
        batch = m["batch"]
        timing[f"us_per_image_b{batch}"] = float(m["us"])
        timing[f"fps_b{batch}"] = float(m["fps"])
        timing[f"timed_calls_b{batch}"] = int(m["n"])
        timing[f"median_us_per_image_b{batch}"] = float(m["median"])
        timing[f"mad_us_per_image_b{batch}"] = float(m["mad"])
    return timing


def measurement_sane(correctness: dict, timing: dict, batches: list[int]) -> bool:
    """Whether a run's numbers may stand as the published baseline.

    Args:
        correctness: The final ``parse_correctness`` result.
        timing: The merged ``parse_timing`` results.
        batches: The batches the check timed.

    Returns:
        True when every image matched and each batch was timed over
        ``N_ITERS`` calls, its fastest call positive and no slower than the
        median.
    """
    if not correctness["passed"]:
        return False
    for batch in batches:
        fastest = timing.get(f"us_per_image_b{batch}")
        median = timing.get(f"median_us_per_image_b{batch}")
        if timing.get(f"timed_calls_b{batch}") != N_ITERS or fastest is None:
            return False
        if median is None or not 0 < fastest <= median:
            return False
    return True


def artifact_stats(build_dir: str | None) -> dict:
    """Sizes of the files the warm build actually produced.

    A full-ELF build's ``design.elf`` carries every core and the runtime
    sequence; otherwise the xclbin and instruction stream stand in for it.
    """
    if build_dir is None:
        return {}
    out: dict = {}
    elf = os.path.join(build_dir, "design.elf")
    if os.path.exists(elf):
        out["elf_bytes"] = os.path.getsize(elf)
    else:
        for metric, name in (
            ("xclbin_bytes", "final.xclbin"),
            ("instr_bytes", "insts.bin"),
        ):
            path = os.path.join(build_dir, name)
            if os.path.exists(path):
                out[metric] = os.path.getsize(path)
    return out


def aggregate(
    correctness: dict, timing: dict, artifacts: dict, compile_stats: dict
) -> list[dict]:
    of = f"of {correctness['images_total']}"
    out = [row(CASE, "images_passed", correctness["images_passed"], "images", of)]
    if correctness["ort_matches"] is not None:
        out.append(row(CASE, "ort_matches", correctness["ort_matches"], "images", of))
    for metric, value in timing.items():
        if metric.startswith("us_per_image_b"):
            batch = metric.removeprefix("us_per_image_b")
            spread = (
                f"fastest of {timing[f'timed_calls_b{batch}']}; median "
                f"{timing[f'median_us_per_image_b{batch}']:.1f}, MAD "
                f"{timing[f'mad_us_per_image_b{batch}']:.1f}"
            )
            out.append(row(CASE, metric, value, "us", spread))
        elif metric.startswith("fps_b"):
            out.append(row(CASE, metric, value, "fps"))
    for metric, unit in (
        ("elf_bytes", "bytes"),
        ("xclbin_bytes", "bytes"),
        ("instr_bytes", "bytes"),
    ):
        if artifacts.get(metric) is not None:
            out.append(row(CASE, metric, artifacts[metric], unit))
    return out + compile_rows(CASE, compile_stats)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument(
        "--weights-dir",
        type=Path,
        default=Path(os.environ.get("IRON_EXAMPLE_WEIGHTS_DIR", "/srv")) / "yolo26n",
        help="holds yolo26n_xint8.onnx and coco128/ (default: $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n)",
    )
    parser.add_argument(
        "--n-images",
        type=int,
        default=8,
        help="COCO images to run, a multiple of 8 (default: 8)",
    )
    add_pmode_argument(parser)
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)
    if args.n_images < BATCH or args.n_images % BATCH:
        parser.error(f"--n-images {args.n_images} is not a positive multiple of {BATCH}")

    # Deferred so that the tests can import this module's parsers without an NPU runtime.
    from aie.utils.benchmark import preflight, provenance

    onnx = args.weights_dir / "yolo26n_xint8.onnx"
    images = args.weights_dir / "coco128" / "images" / "train2017"
    pre = preflight()
    meta = {
        "preflight": dict(vars(pre)),
        "provenance": provenance(device=pre.device, pmode=pre.pmode),
        "weights_dir": str(args.weights_dir),
        "n_images": args.n_images,
    }
    if args.pmode != "any" and pre.pmode != args.pmode:
        return refuse(
            f"power mode is {pre.pmode or 'unreadable'}, required '{args.pmode}'",
            meta,
            args.out,
            args.meta,
        )
    if not onnx.is_file() or not images.is_dir():
        missing = onnx if not onnx.is_file() else images
        return refuse(
            f"{missing} is missing; see --weights-dir", meta, args.out, args.meta
        )

    env = dict(os.environ)
    gen = ensure_data(onnx, env)
    if gen is not None and gen.returncode != 0:
        print(gen.stdout + gen.stderr)
        return refuse(
            f"gen_yolo_data.py exited {gen.returncode}", meta, args.out, args.meta
        )
    cache_dir = tempfile.mkdtemp(prefix="component-checks-yolo26n-")
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    correctness = parse_correctness("", 1)
    timing = parse_timing("")
    artifacts: dict = {}
    log = (gen.stdout + gen.stderr) if gen else ""
    try:
        compiled = compile_cold_warm(images, onnx, env, cache_dir)
        log += compiled["log"]
        if compiled["built"].returncode == 0:
            artifacts = artifact_stats(compiled["build_dir"])
            ran = run_timed(images, onnx, args.n_images, 1, run_env)
            correctness = parse_correctness(ran.stdout + ran.stderr, ran.returncode)
            log += ran.stdout + ran.stderr
            if correctness["passed"]:
                batched = run_timed(images, onnx, args.n_images, BATCH, run_env)
                log += batched.stdout + batched.stderr
                in_batches = parse_correctness(
                    batched.stdout + batched.stderr, batched.returncode
                )
                if in_batches["passed"]:
                    timing = parse_timing(ran.stdout + ran.stderr) | parse_timing(
                        batched.stdout + batched.stderr
                    )
                else:
                    correctness = in_batches
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

    print(log)
    failed = [] if correctness["passed"] else [CASE]

    write(args.out, aggregate(correctness, timing, artifacts, compiled))
    write(
        args.meta,
        {
            **meta,
            "measurement_sane": measurement_sane(correctness, timing, [1, BATCH]),
            "detail": [
                {
                    "case": CASE,
                    **correctness,
                    **timing,
                    **artifacts,
                    **{
                        k: v
                        for k, v in compiled.items()
                        if k not in ("built", "build_dir", "log")
                    },
                }
            ],
            "failed": failed,
            "exitstatus": 1 if failed else 0,
        },
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
