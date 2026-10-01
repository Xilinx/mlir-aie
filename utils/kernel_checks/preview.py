#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Preview the Nightly Kernel Checks page locally against published data.

Copies ``kernel-checks/`` from the publication branch (``origin/gh-pages`` by
default) into a scratch directory and serves it, except that the page itself
is read from this working tree on every request: edit
``utils/kernel_checks/index.html``, reload the browser.

    python3 utils/kernel_checks/preview.py                 # real data
    python3 utils/kernel_checks/preview.py --fetch         # git fetch it first
    python3 utils/kernel_checks/preview.py --scenario bad-night

``--scenario bad-night`` appends a synthetic npu1 run on top of the real data
(regressions and improvements in cycles and object size, a failing case, a
timing failure, a truncated trace and a Peano bump) so the page's failure
states can be seen; nothing is written back to the branch.
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
    """Extract ``kernel-checks/`` of ``ref`` into ``into``; return the directory."""
    blob = subprocess.run(
        ["git", "-C", str(ROOT), "archive", ref, "kernel-checks"],
        check=True,
        capture_output=True,
    ).stdout
    with tarfile.open(fileobj=BytesIO(blob)) as tar:
        tar.extractall(into, filter="data")
    return into / "kernel-checks"


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

    def log_message(self, *args):
        pass


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--ref", default="origin/gh-pages")
    parser.add_argument(
        "--fetch", action="store_true", help="git fetch origin gh-pages first"
    )
    parser.add_argument("--scenario", choices=["real", "bad-night"], default="real")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    if args.fetch:
        subprocess.run(
            ["git", "-C", str(ROOT), "fetch", "origin", "gh-pages"], check=True
        )
    scratch = Path(tempfile.mkdtemp(prefix="kernel-checks-"))
    site = extract(args.ref, scratch)
    if args.scenario == "bad-night":
        bad_night(site)
    handler = functools.partial(Handler, directory=str(scratch))
    with http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler) as httpd:
        print(f"Serving {args.ref} ({args.scenario}) with the working-tree page:")
        print(f"  http://127.0.0.1:{args.port}/kernel-checks/")
        httpd.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
