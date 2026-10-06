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
reload the browser.

    python3 utils/kernel_checks/preview.py                 # real data
    python3 utils/kernel_checks/preview.py --fetch         # git fetch it first
    python3 utils/kernel_checks/preview.py --scenario bad-night
    python3 utils/kernel_checks/preview.py --site components --scenario refused-night

A scenario appends a synthetic run on top of the real data so the page's
failure states can be seen; nothing is written back to the branch. On the
kernels (``--site kernels``, the default), ``bad-night`` is an npu1 run with
regressions and improvements in cycles and object size, a failing case, a
timing failure, a truncated trace and a Peano bump, and ``refused-night`` an
npu1 run that refused to measure in the wrong power mode. On the components,
``bad-night`` fails sweep seeds 4 and 11 and the hardware run of seed 7 at
batch 64, and ``refused-night`` is a hardware run that refused. Until the
nightlies publish batch rows and per-seed detail, a few synthetic nights of
both come first, so the component cards and charts have something to show.
"""

import argparse
import copy
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
import ci_health  # noqa: E402
import publish  # noqa: E402
import sa_placer  # noqa: E402
import sa_placer_hw  # noqa: E402

# Per hardware seed: its batch 1 launch and streaming time in us, and its cost.
HW_SEEDS = {3: (316.0, 65.0, 290), 2: (309.0, 65.0, 295), 7: (313.0, 72.0, 295)}
HW_DETAIL = ("seed", "batch", "passed", "us_per_image", "e2e_us_per_image")


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


def _synthetic(base: dict, run_id: str, message: str) -> dict:
    """Copy the record ``base`` as a nightly run of now."""
    run = copy.deepcopy(base)
    run["id"] = run_id
    run["url"] = ""
    run["date"] = publish.iso(publish.now_utc())
    run["event"] = "schedule"
    run["commit"] = {
        **run.get("commit", {}),
        "id": "0" * 40,
        "url": "",
        "message": message,
    }
    return run


def _store(out: Path, run: dict) -> None:
    (out / "runs" / f"{run['id']}.json").write_text(json.dumps(run, indent=1))
    publish.rebuild(out, publish.now_utc() + datetime.timedelta(seconds=1))


def refused_night(out: Path) -> None:
    """Append a run to ``out`` that refused to measure in the wrong power mode."""
    run = _synthetic(
        json.loads((out / "latest.json").read_text()),
        "preview-refused-night",
        "Preview: a synthetic refused night",
    )
    run.update(
        pmode="default",
        sane=None,
        published=False,
        n_rows=0,
        exitstatus=1,
        failed=[],
        truncated=[],
        rows={},
        refused="power mode is default, required 'turbo'",
    )
    _store(out, run)
    print(f"refused-night on {out.name}: {run['refused']}")


def _hw_night(rng: random.Random) -> tuple[list, dict]:
    """One synthetic hardware nightly: its rows and meta."""
    runs, placements = [], []
    for seed, (one, streaming, cost) in HW_SEEDS.items():
        digest = hashlib.sha256(str(seed).encode()).hexdigest()[:12]
        for batch in (1, 4, 16, 64):
            launch = (one + streaming * (batch - 1)) * rng.uniform(0.99, 1.02)
            per, e2e = launch / batch, (launch + 75) / batch
            runs.append(
                {
                    "seed": seed,
                    "batch": batch,
                    "passed": True,
                    "launch_us": round(launch, 1),
                    "us_per_image": round(per, 2),
                    "us_per_image_range": f"median {per * 1.02:.2f}, MAD {per * 0.005:.2f}, p95 {per * 1.08:.2f}",
                    "e2e_us_per_image": round(e2e, 2),
                    "e2e_range": f"median {e2e * 1.05:.2f}, MAD {e2e * 0.01:.2f}, p95 {e2e * 1.15:.2f}",
                    "compile_s": round(rng.uniform(39, 41), 1),
                    "placement": digest,
                }
            )
        placements.append({"seed": seed, "placement": digest, "placement_cost": cost})
    meta = {
        "preflight": {"pmode": "turbo", "device": "NPU Strix Halo"},
        "measurement_sane": True,
        "detail": [
            {k: r[k] for k in HW_DETAIL + ("compile_s", "placement")} for r in runs
        ],
        "failed": [],
        "exitstatus": 0,
    }
    return sa_placer_hw.aggregate(runs, placements), meta


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
            "stderr": "",
        }
        for fixture, (base, cpu_ms, rss) in zip(
            sa_placer.FIXTURES, ((12, 780, 63), (290, 25000, 67))
        )
        for seed in range(1, 21)
    ]
    return sa_placer.aggregate(rows), sa_placer.meta(rows, sa_placer.FIXTURES, 20, 1.0)


def synthetic_components(site: Path, nights: int = 5) -> None:
    """Append ``nights`` daily nightlies to a check that publishes no per-seed detail yet."""
    rng = random.Random(1)
    now = publish.now_utc()
    for check, night in (("sa-placer", _sweep_night), ("sa-placer-hw", _hw_night)):
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


def bad_components(site: Path) -> None:
    """Fail sweep seeds 4 and 11 and hardware seed 7 at batch 64, the sweep's costs up 8%."""
    out = site / "sa-placer"
    run = _synthetic(
        json.loads((out / "latest.json").read_text()),
        "preview-bad-night",
        "Preview: a synthetic bad night",
    )
    detail = run["detail"]
    for d in detail:
        d["final_cost"] = round(d["final_cost"] * 1.08) if d["final_cost"] else None
        if d["fixture"] == "mobilenet" and d["seed"] in (4, 11):
            d.update(passed=False, final_cost=None)
    run["rows"] = publish.rows_by_case(sa_placer.aggregate(detail))
    run["failed"] = [
        f"{d['fixture']}/seed{d['seed']}" for d in detail if not d["passed"]
    ]
    _store_failed(out, run)

    out = site / "sa-placer-hw"
    run = _synthetic(
        json.loads((out / "latest.json").read_text()),
        "preview-bad-night",
        "Preview: a synthetic bad night",
    )
    failed = "mobilenet/seed=7/batch=64"
    run["rows"].pop(failed, None)
    run["rows"]["mobilenet"]["failed_seeds"]["value"] += 1
    for d in run.get("detail", []):
        if (d["seed"], d["batch"]) == (7, 64):
            d["passed"] = False
    run["failed"] = [failed]
    _store_failed(out, run)


def _store_failed(out: Path, run: dict) -> None:
    run["exitstatus"] = 1
    run["n_rows"] = sum(len(c) for c in run["rows"].values())
    _store(out, run)
    print(f"bad-night on {out.name}: failed {', '.join(run['failed'])}")


def bad_night(site: Path, npu: str = "npu1", seed: int = 7) -> None:
    """Append a synthetic run to ``npu`` with something wrong in every way."""
    out = site / npu
    latest = json.loads((out / "latest.json").read_text())
    catalogue = json.loads((out / "catalogue.json").read_text())
    rng = random.Random(seed)
    run = copy.deepcopy(latest)
    run["id"] = "preview-bad-night"
    run["url"] = ""
    now = publish.now_utc()
    run["date"] = publish.iso(now)
    run["commit"] = {
        **run["commit"],
        "id": "0" * 40,
        "url": "",
        "message": "Preview: a synthetic bad night",
    }
    run["provenance"] = {**run["provenance"], "peano": "22.0.0+preview1"}
    cases = sorted(run["rows"])
    picked = rng.sample(cases, 12)
    slower, faster, bigger = picked[:6], picked[6:9], picked[9:10]
    failing, timing = picked[10], picked[11]
    for i, case in enumerate(slower):
        cell = run["rows"][case]["cycles"]
        cell["value"] = round(cell["value"] * (1.03 + 0.05 * i))
    for case in faster:
        cell = run["rows"][case]["cycles"]
        cell["value"] = round(cell["value"] * 0.9)
    for case in bigger:
        cell = run["rows"][case].get("kernel_object_bytes")
        if cell:
            cell["value"] = round(cell["value"] * 1.12)
    truncated = slower[0]
    cyc = run["rows"][truncated]["cycles"]
    spread = cyc.get("range") or f"median {cyc['value']} max {cyc['value']} n=16"
    cyc["range"] = spread + "; truncated"
    run["truncated"] = [truncated]
    for case in (failing, timing):
        run["rows"].pop(case, None)
    run["failed"] = [f"test_kernels_perf.py::test_kernel_perf[{timing}]"]
    run["n_rows"] = sum(len(c) for c in run["rows"].values())
    for k in catalogue["kernels"]:
        if k["factory"] == failing.split("/")[0]:
            k["failed"] = sorted({*k.get("failed", []), failing})
            k["passed"] = max(0, k.get("passed", 0) - 1)
            k["timed"] = max(0, k.get("timed", 0) - 1)
        if k["factory"] == timing.split("/")[0]:
            k["timing_failed"] = sorted({*k.get("timing_failed", []), timing})
            k["timed"] = max(0, k.get("timed", 0) - 1)
    catalogue["date"] = run["date"]
    run.update(publish._summary_of_catalogue(catalogue))
    (out / "catalogue.json").write_text(json.dumps(catalogue, indent=1))
    (out / "runs" / f"{run['id']}.json").write_text(json.dumps(run, indent=1))
    publish.rebuild(out, now + datetime.timedelta(seconds=1))
    print(
        f"bad-night on {npu}: slower {len(slower)}, faster {len(faster)}, "
        f"bigger {len(bigger)}, failing {failing}, timing failed {timing}"
    )


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
        help="which checks the scenario changes and the printed link opens",
    )
    parser.add_argument(
        "--scenario", choices=["real", "bad-night", "refused-night"], default="real"
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.fetch:
        subprocess.run(
            ["git", "-C", str(ROOT), "fetch", "origin", "gh-pages"], check=True
        )
    scratch = Path(tempfile.mkdtemp(prefix="kernel-checks-"))
    site = extract(args.ref, scratch)
    components = scratch / "component-checks"
    synthetic_components(components)
    synthetic_ci_health(scratch / "ci-health")
    if args.site == "kernels" and args.scenario == "bad-night":
        bad_night(site)
    elif args.site == "kernels" and args.scenario == "refused-night":
        refused_night(site / "npu1")
    elif args.scenario == "bad-night":
        bad_components(components)
    elif args.scenario == "refused-night":
        refused_night(components / "sa-placer-hw")
    view = "#view=placement" if args.site == "components" else ""
    handler = functools.partial(Handler, directory=str(scratch))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {args.ref} ({args.scenario}) with the working-tree page:")
        print(f"  http://127.0.0.1:{args.port}/dashboard/{view}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
