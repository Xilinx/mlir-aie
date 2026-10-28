#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Preview the maintainer dashboard locally against published data.

Copies ``kernel-checks/`` and ``component-checks/`` from the publication
branch (``origin/gh-pages`` by default) into a scratch directory, migrates a
component check still in github-action-benchmark's format as its nightly's
next publish will, and serves it as the nightlies publish it (the page at ``dashboard/``, a redirect
at ``kernel-checks/``), except that the page itself is read from this
working tree on every request: edit ``utils/kernel_checks/index.html``,
reload the browser. It reads synthetic nights through the checks' own code,
so it runs where mlir_aie imports (``source utils/env_setup.sh``).

    python3 utils/kernel_checks/preview.py                 # real data
    python3 utils/kernel_checks/preview.py --fetch         # git fetch it first
    python3 utils/kernel_checks/preview.py --site components

Nothing is written back to the branch. Until the nightlies publish batch
rows and per-seed detail, a few synthetic nights come first, so the
component cards and charts have something to show; the page's own failure
states (a correctness failure, a timing failure, a refused run, a
regression) are each covered directly by test/test_kernel_checks_page.py,
against real recorded data rather than a run against the live branch.
"""

import argparse
import datetime
import functools
import hashlib
import http.server
import json
import random
import subprocess
import sys
import tarfile
import tempfile
from io import BytesIO
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAGE = HERE / "index.html"

sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "utils" / "component_checks"))
import aiecc_stages  # noqa: E402
import ci_health  # noqa: E402
import llama_chain_hw  # noqa: E402
import publish  # noqa: E402
import sa_placer  # noqa: E402
import sa_placer_hw  # noqa: E402
import yolo26n_hw  # noqa: E402

# Per hardware seed: its batch 1 launch and each further image's time in us,
# and its cost.
HW_SEEDS = {3: (316.0, 65.0, 290), 2: (309.0, 65.0, 295), 7: (313.0, 72.0, 295)}
# A cold and a warm compile's aiecc profiles, for the stage times.
STAGE_PROFILES = (
    ROOT / "test/fixtures/aiecc_profile_mobilenet.json",
    ROOT / "test/fixtures/aiecc_profile_shim.json",
)
# YOLO26n's own, from a cold then warm compile of yolo26n.py --synthetic 7 on
# Strix: 27.89 s and 6.01 s.
YOLO_STAGE_PROFILES = (
    ROOT / "test/fixtures/aiecc_profile_yolo26n_cold.json",
    ROOT / "test/fixtures/aiecc_profile_yolo26n_warm.json",
)
HW_DETAIL = (
    "seed",
    "batch",
    "effort_label",
    "passed",
    *sa_placer_hw.TIMED,
    "fps",
    "dispatch_overhead_us",
    "placement",
)


def extract(ref: str, into: Path) -> Path:
    """Extract the checks' directories of ``ref`` into ``into``; return the kernels'.

    ``component-checks/`` is optional (a branch from before it); each of its
    checks is migrated and redirected as the nightly publishes it.
    """
    for path in ("kernel-checks", "component-checks", "ci-health"):
        archive = subprocess.run(
            ["git", "-C", str(ROOT), "archive", ref, path],
            check=path == "kernel-checks",
            capture_output=True,
        )
        if archive.returncode:
            continue
        with tarfile.open(fileobj=BytesIO(archive.stdout)) as tar:
            tar.extractall(into, filter="data")
    components = into / "component-checks"
    publish.redirect(components, "../dashboard/#view=placement")
    for check in publish.COMPONENTS:
        out = components / check
        if publish.migrate(out) or (out / "runs").exists():
            publish.rebuild(out)
            publish.redirect(out, f"../../dashboard/#view={publish.SECTION_OF[check]}")
    publish.redirect(into / "kernel-checks", "../dashboard/#view=night")
    return into / "kernel-checks"


def synthetic_ci_health(out: Path, days: int = 20) -> None:
    """Write a made-up ci-health/ in ci_health.py's format, if there is none.

    It has each workflow's runs, the critical ones' figures and a history,
    with a few things wrong, so the CI health tab has something to show.
    """
    if (out / "status.json").exists():
        return
    rng = random.Random(3)
    now = publish.now_utc()
    config = ci_health.load_workflows()

    def stamp(hours):
        return publish.iso(now - datetime.timedelta(hours=hours))

    def run(i, hours, conclusion, status="completed", took=None):
        took = took if took is not None else rng.uniform(20, 70)
        waited = rng.uniform(0.2, 6)
        return {
            "id": i,
            "url": "",
            "status": status,
            "conclusion": conclusion,
            "created_at": stamp(hours + waited / 60),
            "run_started_at": stamp(hours),
            "updated_at": stamp(hours - took / 60),
            "head_sha": f"{i:040x}",
            "title": "Preview: a synthetic run",
            "attempt": 1,
        }

    workflows = []
    for n, wf in enumerate(config["workflows"]):
        every = wf["every"]
        first = rng.uniform(0.5, every)
        runs = [
            run(
                n * 100 + i,
                first + every * i,
                "failure" if rng.random() < 0.15 else "success",
            )
            for i in range(ci_health.RUNS_KEPT)
        ]
        jobs = None
        if wf["file"] == "nightlyKernelChecks.yml":
            # Still going: one leg never found a runner.
            runs[0].update(status="in_progress", conclusion=None)
            jobs = [
                {
                    "name": "Kernel checks (aie2-4col)",
                    "url": "",
                    "status": "completed",
                    "conclusion": "failure",
                    "created_at": stamp(first),
                    "started_at": stamp(first),
                },
                {
                    "name": "Kernel checks (aie2p-8col)",
                    "url": "",
                    "status": "queued",
                    "conclusion": None,
                    "created_at": stamp(first + 7),
                    "started_at": None,
                },
            ]
        elif wf.get("legs"):
            jobs = [
                {
                    "name": "build",
                    "url": "",
                    "status": "completed",
                    "conclusion": runs[0]["conclusion"],
                    "created_at": stamp(first),
                    "started_at": stamp(first),
                }
            ]
        workflows.append({**wf, "runs": runs, "jobs": jobs})
    critical = []
    for n, wf in enumerate(config["critical"]):
        minutes = rng.uniform(8, 120)
        main = rng.choice([0.95, 1.0, 0.82])
        critical.append(
            {
                **wf,
                "runs": rng.randint(60, 300),
                "main": {
                    "passed": round(40 * main),
                    "failed": 40 - round(40 * main),
                    "rate": main,
                },
                "latest_main": run(
                    9000 + n, rng.uniform(1, 5), "failure" if n == 0 else "success"
                ),
                "latest_main_jobs": (
                    [
                        {
                            "name": "build-and-test-from-source (aie2p-8col)",
                            "url": "",
                            "status": "completed",
                            "conclusion": "failure",
                            "created_at": stamp(3),
                            "started_at": stamp(3),
                        }
                    ]
                    if n == 0
                    else None
                ),
                "pull_requests": {"passed": 120, "failed": 45, "rate": 0.727},
                "merge_queue": {"passed": 30, "failed": 1, "rate": 0.968},
                "flaky": {
                    "commits": 150,
                    "flaky": [4, 21, 2, 1, 0][n % 5],
                    "rate": round([4, 21, 2, 1, 0][n % 5] / 150, 3),
                },
                "minutes_median": round(minutes, 1),
                "minutes_p90": round(minutes * 1.6, 1),
                "queued_median": round(rng.uniform(0.2, 25), 1),
                # Runs on main, newest first, about two a day.
                "main_runs": [
                    run(
                        9100 + 50 * n + i,
                        1 + 11 * i,
                        (
                            "failure"
                            if (i == 0 and n == 0) or rng.random() < 0.08
                            else "success"
                        ),
                        took=minutes * rng.uniform(0.85, 1.2),
                    )
                    for i in range(28)
                ],
            }
        )
    status = {
        "schema": 1,
        "generated_at": publish.iso(now - datetime.timedelta(minutes=20)),
        "repo": "Xilinx/mlir-aie",
        "window_days": ci_health.WINDOW_DAYS,
        "critical": critical,
        "merge_time": {
            "merged": 37,
            "hours_median": 30.2,
            "hours_p90": 212.0,
            "open": 58,
            "open_drafts": 14,
            "open_over_30_days": 17,
        },
        "workflows": workflows,
    }
    history = None
    for d in range(days, -1, -1):
        point = ci_health.history_point(
            {**status, "generated_at": publish.iso(now - datetime.timedelta(days=d))}
        )
        point["merge"]["hours_median"] = round(30 + 8 * rng.uniform(-1, 1), 1)
        for c in point["critical"].values():
            c["minutes_median"] = round(c["minutes_median"] * rng.uniform(0.9, 1.1), 1)
        history = ci_health.update_history(history, point, now)
    out.mkdir(parents=True, exist_ok=True)
    (out / "status.json").write_text(json.dumps(status, indent=1))
    (out / "history.json").write_text(json.dumps(history, indent=1))
    print("synthetic ci-health: no published one yet")


def _hw_night(rng: random.Random) -> tuple[list, dict]:
    """One synthetic hardware nightly at each SA effort: its rows and meta."""
    runs, compiles, placements = [], [], []
    for seed, (one, streaming, cost) in HW_SEEDS.items():
        for effort_label in sa_placer_hw.EFFORTS:
            digest = hashlib.sha256(f"{seed}/{effort_label}".encode()).hexdigest()[:12]
            compiles.append(
                {
                    "seed": seed,
                    "effort_label": effort_label,
                    "passed": True,
                    "compile_s": round(rng.uniform(39, 41), 1),
                    "placement": digest,
                }
            )
            for batch in (1, 4, 16, 64):
                launch = (one + streaming * (batch - 1)) * rng.uniform(0.99, 1.02)
                overhead = 47 * rng.uniform(0.99, 1.02)
                printed = "".join(
                    f"{label:<18}(avg/min/max us): {t * 1.01:.1f} / {t:.1f} / "
                    f"{t * 1.1:.1f}   [median {t * 1.02:.1f}, MAD {t * 0.005:.1f}, "
                    f"p95 {t * 1.08:.1f}]\n"
                    for label, t in (
                        ("Dispatch", launch),
                        ("Dispatch overhead", overhead),
                        ("NPU", launch - overhead),
                        ("End-to-end", launch + 75),
                    )
                )
                runs.append(
                    {
                        "seed": seed,
                        "batch": batch,
                        "effort_label": effort_label,
                        **sa_placer_hw.parse_run(printed + "PASS!\n", 0, batch),
                        "placement": digest,
                    }
                )
            placements.append(
                {
                    "seed": seed,
                    "effort_label": effort_label,
                    "placement": digest,
                    "placement_cost": cost,
                }
            )
    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "provenance": "dispatch dynamic",
        "efforts": sa_placer_hw.EFFORTS,
        "measurement_sane": True,
        "compiles": compiles,
        "placements": placements,
        "detail": [{k: r[k] for k in HW_DETAIL} for r in runs],
        "failed": [],
        "exitstatus": 0,
    }
    return sa_placer_hw.aggregate(runs, compiles, placements), meta


def _sweep_night(rng: random.Random) -> tuple[list, dict]:
    """One synthetic sweep nightly: a placement's cost is the seed's, its CPU time noise."""
    costs = random.Random(0)
    rows = [
        {
            "fixture": sa_placer.case_of(fixture),
            "seed": seed,
            "passed": True,
            "final_cost": base + costs.randint(0, base // 20),
            "cpu_ms": cpu_ms * rng.uniform(0.9, 1.15),
            "peak_rss_mb": rss + rng.uniform(0, 1),
            "timed_out": False,
            "output": "",
        }
        for fixture, (base, cpu_ms, rss) in zip(
            sa_placer.FIXTURES, ((12, 780, 63), (290, 25000, 67))
        )
        for seed in range(1, 21)
    ]
    return sa_placer.aggregate(rows), sa_placer.meta(rows, sa_placer.FIXTURES, 20, 1.0)


def _compile_night(
    rng: random.Random, cold_s: float, warm_s: float, profiles: tuple[Path, Path]
) -> dict:
    """Return a cold and a warm compile's times and memory, staged as ``profiles`` were.

    Args:
        rng: The night's random source.
        cold_s: The cold compile's typical wall time.
        warm_s: The warm compile's.
        profiles: The cold and the warm compile's captured aiecc profiles;
            each must span less than its compile.

    Returns:
        compile_stats' rows and stage_rows' for both compiles.
    """
    cold, warm = cold_s * rng.uniform(0.97, 1.05), warm_s * rng.uniform(0.97, 1.03)
    rss, rss_warm = rng.uniform(2000, 2200), rng.uniform(1700, 1900)
    return {
        "compile_s": round(cold, 1),
        "compile_peak_mib": round(2 * rss, 1),
        "compile_peak_rss_mb": round(rss, 1),
        "compile_s_warm": round(warm, 1),
        "compile_peak_mib_warm": round(1.2 * rss_warm, 1),
        "compile_peak_rss_mb_warm": round(rss_warm, 1),
        **aiecc_stages.stage_rows(str(profiles[0]), cold, ""),
        **aiecc_stages.stage_rows(str(profiles[1]), warm, "_warm"),
    }


def _llama_night(rng: random.Random) -> tuple[list, dict]:
    """One synthetic llama decode nightly, read through the check's own parsers."""
    n = llama_chain_hw.N_LAYERS
    drift = rng.uniform(0.98, 1.03)
    ttft = 125.9 * rng.uniform(0.98, 1.03)
    ran = "".join(f"  token {i} (pos {64 + i}): PASS\n" for i in range(4))
    ran += "decode: 4/4 tokens PASS\n"
    for at in (llama_chain_hw.DISPATCH_RATE_AT, *llama_chain_hw.CONTEXTS):
        ms = (45.1 + 0.0033 * at) * drift
        ran += (
            f"decode at {at}: per-dispatch n={llama_chain_hw.N_ITERS} "
            f"min={4 * ms:.2f} median={4 * ms + 0.5:.2f} mad=0.32 ms, "
            f"4 tokens/dispatch: {ms:.2f} ms/token ({1000 / ms:.2f} tok/s)\n"
        )
    generated = (
        "prompt 6 tokens, generated 32, median of 5: "
        f"TTFT {ttft:.1f} ms (min {ttft - 2:.1f}; NPU dispatch {ttft - 12:.1f}, "
        f"host 12.0), decode {970 / 45.1 / drift:.2f} tok/s\n"
        "check: 29/32 tokens are numpy_llama_ref's argmax\n"
        "oracle: 32/32 generated tokens pass\n"
    )
    correctness = llama_chain_hw.parse_correctness(ran, 0)
    timing = llama_chain_hw.parse_timing(ran, 0)
    gen = llama_chain_hw.parse_generate(generated, 0)
    prefill = {}
    for length in llama_chain_hw.PROMPT_LENGTHS:
        pre_ttft = (15.6 * length + 180) * drift
        pre = (
            f"prompt {length + 1} tokens, generated 1, median of 5: TTFT "
            f"{pre_ttft:.1f} ms (min {pre_ttft - 20:.1f}; NPU dispatch "
            f"{pre_ttft - 30:.1f}, host 30.0), decode n/a\n"
        )
        prefill[length] = llama_chain_hw.parse_prefill(pre, 0)
    accuracy_out = (
        f"next-token top-1 agreement: {91.7 * rng.uniform(0.99, 1.01):.1f}%  (22/24)\n"
        f"mean KL(bf16 || int8)     : {0.0250 * rng.uniform(0.95, 1.05):.4f}\n"
    )
    accuracy = llama_chain_hw.parse_accuracy_bf16(accuracy_out, 0)
    artifacts = {"elf_bytes": 1_843_200, "pdi_bytes": 412_160}
    compiled = _compile_night(rng, 19.8, 0.5, STAGE_PROFILES)
    case = f"llama/persist_grow/N={n}"
    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "n_layers": n,
        "pt": llama_chain_hw.P_MAX,
        "t": llama_chain_hw.T,
        "contexts": [llama_chain_hw.DISPATCH_RATE_AT, *llama_chain_hw.CONTEXTS],
        "prompt_lengths": list(llama_chain_hw.PROMPT_LENGTHS),
        "measurement_sane": True,
        "detail": [
            {
                "case": case,
                **correctness,
                "timing": {str(at): t for at, t in timing.items()},
                **gen,
                "prefill": {str(length): p for length, p in prefill.items()},
                **accuracy,
                **artifacts,
                **compiled,
            }
        ],
        "failed": [],
        "exitstatus": 0,
    }
    return (
        llama_chain_hw.aggregate(
            correctness, timing, gen, prefill, accuracy, artifacts, compiled
        ),
        meta,
    )


def _yolo_night(rng: random.Random) -> tuple[list, dict]:
    """One synthetic YOLO26n nightly, read through the check's own parsers."""
    images = [f"{i:012d}.jpg" for i in (9, 25, 30, 34, 36, 42, 49, 61)]
    ran = "".join(f"image {name}: PASS (0/8400 differ)\n" for name in images)
    ran += f"yolo26n: {len(images)}/{len(images)} images PASS\n"
    ran += f"ort: {len(images)}/{len(images)} images match onnxruntime\n"
    one = 12700 * rng.uniform(0.98, 1.03)
    batched = 6150 * rng.uniform(0.98, 1.03)
    # One run per batch, as the check makes them: the dispatch ~157 us under
    # the call and the empty dispatch ~88 us, as on Strix.
    timing = {}
    for us, b in ((one, 1), (batched, yolo26n_hw.BATCH)):
        call, dispatch = us * b, us * b - 157 * rng.uniform(0.95, 1.05)
        empty = 88 * rng.uniform(0.95, 1.05)
        timing |= yolo26n_hw.parse_timing(
            "".join(
                f"{label:<18}(avg/min/max us): {t * 1.01:.1f} / {t:.1f} / "
                f"{t * 1.1:.1f}   [median {t * 1.02:.1f}, MAD {t * 0.005:.1f}, "
                f"p95 {t * 1.08:.1f}]\n"
                for label, t in (
                    ("Dispatch", dispatch),
                    ("Dispatch overhead", empty),
                    ("NPU", dispatch - empty),
                    ("End-to-end", call),
                )
            )
            + f"per-image: {us:.1f} us ({1e6 / us:.1f} fps) at batch {b}, fastest of "
            f"{yolo26n_hw.N_ITERS} after {yolo26n_hw.N_WARMUP} warmups; "
            f"median {us * 1.02:.1f} us, MAD {us * 0.01:.1f} us\n"
        )
    correctness = yolo26n_hw.parse_correctness(ran, 0)
    map50 = 0.586 * rng.uniform(0.99, 1.01)
    timing |= yolo26n_hw.parse_map50(
        f"map50: {map50:.4f} over {len(images)} images (58 labelled objects)\n"
    )
    artifacts = {"elf_bytes": 2_621_440}
    compiled = _compile_night(rng, 27.9, 6.0, YOLO_STAGE_PROFILES)
    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "measurement_sane": True,
        "detail": [
            {
                "case": yolo26n_hw.CASE,
                **correctness,
                **timing,
                **artifacts,
                **compiled,
            }
        ],
        "failed": [],
        "exitstatus": 0,
    }
    return yolo26n_hw.aggregate(correctness, timing, artifacts, compiled), meta


def synthetic_components(site: Path, nights: int = 5) -> None:
    """Append ``nights`` daily nightlies to a check that publishes no per-seed detail yet."""
    rng = random.Random(1)
    now = publish.now_utc()
    for check, night in (
        ("sa-placer", _sweep_night),
        ("sa-placer-hw", _hw_night),
        ("llama-chain-hw", _llama_night),
        ("yolo26n-hw", _yolo_night),
    ):
        out = site / check
        latest = out / "latest.json"
        if latest.exists() and "detail" in json.loads(latest.read_text()):
            continue
        for i in range(nights):
            rows, meta = night(rng)
            with tempfile.TemporaryDirectory() as d:
                (Path(d) / "perf.json").write_text(json.dumps(rows))
                (Path(d) / "meta.json").write_text(json.dumps(meta))
                date = now - datetime.timedelta(days=nights - 1 - i, hours=1)
                run = {
                    "id": f"preview-{i}",
                    "url": "",
                    "date": publish.iso(date),
                    "commit": publish.commit_info(
                        "0" * 40, "Preview: a synthetic night", publish.iso(date)
                    ),
                    "event": "schedule",
                }
                record = publish.record_perf(Path(d), target=check, run=run)
            (out / "runs").mkdir(parents=True, exist_ok=True)
            (out / "runs" / f"{run['id']}.json").write_text(
                json.dumps(record, indent=1)
            )
        publish.rebuild(out)
        print(f"synthetic nights on {check}: {nights}")


class Handler(http.server.SimpleHTTPRequestHandler):
    """Serve the scratch copy, but the page from the working tree."""

    def do_GET(self):
        if self.path.split("?", 1)[0] in ("/", "/dashboard"):
            self.send_response(302)
            self.send_header("Location", "/dashboard/")
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean in ("/dashboard/", "/dashboard/index.html"):
            return str(PAGE)
        return super().translate_path(path)

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format, *args):
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--ref", default="origin/gh-pages")
    parser.add_argument(
        "--fetch", action="store_true", help="git fetch origin gh-pages first"
    )
    parser.add_argument(
        "--site",
        choices=["kernels", "components"],
        default="kernels",
        help="which page the printed link opens",
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.fetch:
        subprocess.run(
            ["git", "-C", str(ROOT), "fetch", "origin", "gh-pages"], check=True
        )
    scratch = Path(tempfile.mkdtemp(prefix="kernel-checks-"))
    extract(args.ref, scratch)
    components = scratch / "component-checks"
    synthetic_components(components)
    synthetic_ci_health(scratch / "ci-health")
    view = "#view=placement" if args.site == "components" else ""
    handler = functools.partial(Handler, directory=str(scratch))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {args.ref} with the working-tree page:")
        print(f"  http://127.0.0.1:{args.port}/dashboard/{view}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
