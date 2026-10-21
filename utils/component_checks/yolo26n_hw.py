#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Build + run the YOLO26n 640x640 detector on NPU2 over real COCO images.

The whole detector end to end: does it still build, does every image's
detection match the numpy oracle and onnxruntime (all but ``--ort-mismatches``
images, the ones known to land off a softmax rounding boundary), and how
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
Extracts the parameters from the model into ``--params-cache``, under a name
keyed by the sha256 of the model and the extractor, unless already there.

Writes rows and a meta file in the kernel checks' format
(utils/kernel_checks/publish.py), with the device's power mode and the
toolchain; ``--pmode`` refuses to measure in any other mode. Exits 1 when
the run fails or the check refuses, after writing both. Reproduced, from
programming_examples/ml/yolo26n, with

    python yolo26n.py --images $IRON_EXAMPLE_WEIGHTS_DIR/yolo26n/coco128/images/train2017 --n 8
"""

import argparse
import glob
import hashlib
import os
import re
import shutil
import sys
import tempfile
from importlib.metadata import version
from pathlib import Path

from aie.utils.compile.utils import AIECC_PROFILE_NAME
from aiecc_stages import (
    Measured,
    add_pmode_argument,
    clear_for_warm,
    compile_rows,
    compile_stats,
    refuse,
    row,
    run_measured,
    stage_rows,
    write,
    write_unfinished,
)

CASE = "yolo26n/640"
N_ITERS = 20
N_WARMUP = 10
BATCH = 8
# Seconds each command may take before the check kills it and fails: the
# parameter extraction, one compile, and one checked and timed run.
EXTRACT_TIMEOUT = 1800
COMPILE_TIMEOUT = 1800
RUN_TIMEOUT = 900

_YOLO_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "programming_examples", "ml", "yolo26n"
)
# What gen_yolo_data.py's output depends on besides the ONNX: its extractor
# and the spec it binds, and the packages that read and evaluate the graph.
EXTRACTOR_SOURCES = ("gen_yolo_data.py", "yolo_params.py", "yolo_spec.py")
EXTRACTOR_PACKAGES = ("onnx", "onnxruntime")

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


def params_key(onnx: Path, sources: list[Path], packages: dict[str, str]) -> str:
    """The sha256 that names the parameters extracted from ``onnx``.

    Args:
        onnx: The quantized model.
        sources: The extractor's source files.
        packages: The extractor's packages, name to version.

    Returns:
        The hex digest of the model, each source (by name and contents) and
        each package version, so a change to any of them names a new file.
    """
    digest = hashlib.sha256()
    for name, data in [
        ("model", onnx.read_bytes()),
        *((s.name, s.read_bytes()) for s in sources),
        *((p, v.encode()) for p, v in sorted(packages.items())),
    ]:
        digest.update(f"{name}\0{len(data)}\0".encode())
        digest.update(data)
    return digest.hexdigest()


def ensure_params(onnx: Path, cache: Path, env: dict) -> tuple[Path, Measured | None]:
    """The parameter file for ``onnx``, generated into ``cache`` unless already there.

    Args:
        onnx: The quantized model.
        cache: The directory the keyed parameter files live in.
        env: gen_yolo_data.py's environment.

    Returns:
        The file, named by ``params_key``, and gen_yolo_data.py's run, or
        None when the file was already there.
    """
    key = params_key(
        onnx,
        [Path(_YOLO_DIR) / name for name in EXTRACTOR_SOURCES],
        {name: version(name) for name in EXTRACTOR_PACKAGES},
    )
    path = cache / f"yolo26n_params-{key}.npz"
    if path.exists():
        return path, None
    cache.mkdir(parents=True, exist_ok=True)
    # Renamed into place, so a failed or concurrent run never leaves a partial file.
    partial = cache / f"yolo26n_params-{key}.{os.getpid()}.npz"
    gen = run_measured(
        [
            sys.executable,
            "gen_yolo_data.py",
            "--onnx",
            str(onnx),
            "--out",
            str(partial),
        ],
        _YOLO_DIR,
        env,
        EXTRACT_TIMEOUT,
    )
    if gen.process.returncode == 0:
        os.replace(partial, path)
    return path, gen


def newest_build(cache_dir: str) -> str | None:
    """Return the JIT build directory under ``cache_dir`` aiecc last profiled.

    Args:
        cache_dir: an NPU_CACHE_HOME; each build is ``<cache_dir>/<hash>/``.

    Returns:
        The directory holding the newest ``aiecc_profile.json``, or None when
        no build wrote one.
    """
    profiles = glob.glob(os.path.join(cache_dir, "*", AIECC_PROFILE_NAME))
    if not profiles:
        return None
    return os.path.dirname(max(profiles, key=os.path.getmtime))


def compile_cold_warm(
    images: Path, onnx: Path, params: Path, env: dict, cache_dir: str
) -> dict:
    """Compile twice into the fresh NPU_CACHE_HOME ``cache_dir``: cold, then warm.

    Cold: every cache empty, same as any first-time compile. Warm: only
    what ``clear_for_warm`` keeps (the kernel objects and precompiled
    headers), so the warm compile reruns the full pipeline, which is what an
    edit to the design (not its kernels) costs.
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
        "--params",
        str(params),
    ]
    cold = run_measured(compile_only, _YOLO_DIR, run_env, COMPILE_TIMEOUT)
    build = newest_build(cache_dir)
    stages_cold = stage_rows(
        os.path.join(build or cache_dir, AIECC_PROFILE_NAME),
        cold.seconds if cold.process.returncode == 0 else None,
        "",
    )

    clear_for_warm(cache_dir)
    warm = run_measured(compile_only, _YOLO_DIR, run_env, COMPILE_TIMEOUT)
    build = newest_build(cache_dir)
    stages_warm = stage_rows(
        os.path.join(build or cache_dir, AIECC_PROFILE_NAME),
        warm.seconds if warm.process.returncode == 0 else None,
        "_warm",
    )

    return {
        **stages_cold,
        **stages_warm,
        "built": warm.process if warm.process.returncode == 0 else cold.process,
        "build_dir": build,
        **compile_stats(cold, warm),
    }


def run_timed(
    images: Path, onnx: Path, params: Path, n_images: int, batch: int, env: dict
) -> Measured:
    """Run yolo26n.py over ``n_images`` at ``batch``, checking every image, then timing it."""
    return run_measured(
        [
            sys.executable,
            "yolo26n.py",
            "--images",
            str(images),
            "--onnx",
            str(onnx),
            "--params",
            str(params),
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
        _YOLO_DIR,
        env,
        RUN_TIMEOUT,
    )


def parse_correctness(out: str, returncode: int, ort_mismatches: int = 0) -> dict:
    """Read a yolo26n.py run: how many images matched the oracle and onnxruntime.

    Args:
        out: the run's stdout and stderr.
        returncode: its exit code.
        ort_mismatches: how many of the run's images are known to differ from
            onnxruntime, where its float32 softmax rounds off a boundary
            (verify_oracle.py attributes each).

    Returns:
        ``passed`` (every image matched the oracle, by the run's own summary
        line, and ``ort_passed``), ``ort_passed`` (onnxruntime matched all
        but ``ort_mismatches`` images; False when the run printed no
        onnxruntime line), ``images_passed``, ``images_total``,
        ``ort_matches`` (None without that line) and ``failed_images``, by
        name.
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
    ort_matches = int(ort.group(1)) if ort else None
    ort_passed = ort_matches is not None and ort_matches >= total_n - ort_mismatches
    return {
        "passed": returncode == 0
        and bool(summary)
        and total_n > 0
        and passed_n == total_n
        and ort_passed,
        "ort_passed": ort_passed,
        "images_passed": passed_n,
        "images_total": total_n,
        "ort_matches": ort_matches,
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
    parser.add_argument(
        "--ort-mismatches",
        type=int,
        default=0,
        help="of those, how many are known to differ from onnxruntime where its "
        "float32 softmax rounds off a boundary, as verify_oracle.py attributes "
        "them; fewer matches fail the check (default: 0)",
    )
    parser.add_argument(
        "--params-cache",
        type=Path,
        default=Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "mlir-aie"
        / "yolo26n",
        help="where the parameters extracted from the ONNX are kept, named by the "
        "sha256 of the model and the extractor (default: ~/.cache/mlir-aie/yolo26n)",
    )
    add_pmode_argument(parser)
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)
    if args.n_images < BATCH or args.n_images % BATCH:
        parser.error(f"--n-images {args.n_images} is not a positive multiple of {BATCH}")
    if not 0 <= args.ort_mismatches <= args.n_images:
        parser.error(f"--ort-mismatches {args.ort_mismatches} is not in 0..--n-images")

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
        "ort_mismatches": args.ort_mismatches,
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

    write_unfinished(meta, CASE, args.out, args.meta)
    env = dict(os.environ)
    params, gen = ensure_params(onnx, args.params_cache, env)
    meta["params"] = str(params)
    if gen is not None and gen.process.returncode != 0:
        return refuse(
            f"gen_yolo_data.py exited {gen.process.returncode}",
            meta,
            args.out,
            args.meta,
        )
    cache_dir = tempfile.mkdtemp(prefix="component-checks-yolo26n-")
    run_env = dict(env, NPU_CACHE_HOME=cache_dir)
    correctness = parse_correctness("", 1, args.ort_mismatches)
    timing = parse_timing("")
    artifacts: dict = {}
    try:
        compiled = compile_cold_warm(images, onnx, params, env, cache_dir)
        if compiled["built"].returncode == 0:
            artifacts = artifact_stats(compiled["build_dir"])
            ran = run_timed(images, onnx, params, args.n_images, 1, run_env).process
            correctness = parse_correctness(
                ran.stdout, ran.returncode, args.ort_mismatches
            )
            if correctness["passed"]:
                batched = run_timed(
                    images, onnx, params, args.n_images, BATCH, run_env
                ).process
                in_batches = parse_correctness(
                    batched.stdout, batched.returncode, args.ort_mismatches
                )
                if in_batches["passed"]:
                    timing = parse_timing(ran.stdout) | parse_timing(batched.stdout)
                else:
                    correctness = in_batches
    finally:
        shutil.rmtree(cache_dir, ignore_errors=True)

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
                        if k not in ("built", "build_dir")
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
