#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Preview the Nightly Kernel Checks page locally against published data.

Copies ``kernel-checks/`` and ``component-checks/`` from the publication
branch (``origin/gh-pages`` by default) into a scratch directory, migrates a
component check still in github-action-benchmark's format as its nightly's
next publish will, and serves it, except that the page itself is read from
this working tree on every request: edit ``utils/kernel_checks/index.html``,
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
``bad-night`` fails two sweep seeds and a hardware seed, and
``refused-night`` is a hardware run that refused.
"""

import argparse
import copy
import datetime
import functools
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
import publish  # noqa: E402


def extract(ref: str, into: Path) -> Path:
    """Extract the checks' directories of ``ref`` into ``into``; return the kernels'.

    ``component-checks/`` is optional (a branch from before it); each of its
    checks is migrated and redirected as the nightly publishes it.
    """
    for path in ("kernel-checks", "component-checks"):
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
    publish.redirect(components, "../kernel-checks/#view=components")
    for check in publish.COMPONENTS:
        out = components / check
        if publish.migrate(out) or (out / "runs").exists():
            publish.rebuild(out)
            publish.redirect(out, "../../kernel-checks/#view=components")
    return into / "kernel-checks"


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


def bad_components(site: Path) -> None:
    """Fail two sweep seeds and a hardware seed, the sweep's costs up 8%."""
    for check, seeds in (("sa-placer", (4, 11)), ("sa-placer-hw", (7,))):
        out = site / check
        run = _synthetic(
            json.loads((out / "latest.json").read_text()),
            "preview-bad-night",
            "Preview: a synthetic bad night",
        )
        rows = run["rows"]
        fixture = next(c for c in rows if "failed_seeds" in rows[c])
        failed = [f"{fixture}/seed{n}" for n in seeds]
        for cells in rows.values():
            for metric, cell in cells.items():
                if metric.startswith("final_cost"):
                    cell["value"] = round(cell["value"] * 1.08, 1)
        rows[fixture]["failed_seeds"]["value"] += len(failed)
        for name in failed:
            rows.pop(name, None)
        run["failed"] = failed
        run["exitstatus"] = 1
        run["n_rows"] = sum(len(c) for c in rows.values())
        _store(out, run)
        print(f"bad-night on {check}: failed {', '.join(failed)}")


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
        if self.path.split("?", 1)[0] in ("/", "/kernel-checks"):
            self.send_response(302)
            self.send_header("Location", "/kernel-checks/")
            self.end_headers()
            return
        super().do_GET()

    def translate_path(self, path):
        clean = path.split("?", 1)[0].split("#", 1)[0]
        if clean in ("/kernel-checks/", "/kernel-checks/index.html"):
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
    if args.site == "kernels" and args.scenario == "bad-night":
        bad_night(site)
    elif args.site == "kernels" and args.scenario == "refused-night":
        refused_night(site / "npu1")
    elif args.scenario == "bad-night":
        bad_components(components)
    elif args.scenario == "refused-night":
        refused_night(components / "sa-placer-hw")
    view = "#view=components" if args.site == "components" else ""
    handler = functools.partial(Handler, directory=str(scratch))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {args.ref} ({args.scenario}) with the working-tree page:")
        print(f"  http://127.0.0.1:{args.port}/kernel-checks/{view}")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
