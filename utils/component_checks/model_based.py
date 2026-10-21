#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Turn the model-based tests' reports into a component check's results.

The allocator's property test (test/assign-buffer-addresses/allocator_properties.py)
and the router's (test/create-packet-flows/nightly/router_properties.py and
utils/router_testing/router_mutation.py) each write a JSON report with
``--report``. modelBasedTests.yml runs them nightly; this reads the reports
of one check and writes ``perf.json`` and ``meta.json`` in the kernel checks'
format (utils/kernel_checks/publish.py), exiting 1 when anything failed.

    model_based.py allocator --report allocator.json --out perf.json --meta meta.json
    model_based.py router --report npu2-properties.json --report npu2-hops-on.json \
        ... --out perf.json --meta meta.json

The allocator's rows are one case, ``allocator``; the router's a case per
device, its properties and its mutation runs (both hop settings) together.
Every metric is smaller-is-better. A failed id names what to run again: an
allocator property by name; a router property as
``<device>/properties/seed=<first>+<count>/<property>``; a mutation failure,
the first seed of each kind, as
``<device>/hops-<on|off>/seed=<seed>+1/<kind> <name> <outcome>``; a test of
the router's nightly lit suite, whose results ``--lit`` reads from lit's
``-o`` JSON, as ``lit/<test path>``. A report that is missing (its script
crashed before writing it) fails as ``<file stem>: no report``.
"""

import argparse
import json
import sys
from pathlib import Path

from aiecc_stages import row
from sa_placer import commit

ALLOCATOR_COUNTS = {
    "unsolved": "designs",
    "illegal": "designs",
    "avoidable_crossings": "crossings",
    "bank_pair_sharing": "fraction",
    "wall_s": "s",
}
LIT_FAILURES = {"FAIL", "XPASS", "UNRESOLVED", "TIMEOUT"}
ROUTER_COUNTS = {
    "unrouted": "routings",
    "illegal": "routings",
    "hop_ratio": "×",
    "amsels_per_design": "amsels",
    "ms_per_design": "ms",
}


def allocator(reports: dict[str, dict | None]) -> tuple[list, list, list]:
    """Return the allocator's rows, failed ids and detail."""
    rows, failed, detail = [], [], []
    for stem, report in reports.items():
        if report is None:
            failed.append(f"{stem}: no report")
            continue
        bad = [p["name"] for p in report["properties"] if not p["ok"]]
        failed += bad
        rows.append(row("allocator", "failed_properties", len(bad), "properties"))
        rows += [
            row("allocator", metric, report["counts"][metric], unit)
            for metric, unit in ALLOCATOR_COUNTS.items()
        ]
        detail += report["properties"]
    return rows, failed, detail


def router(
    reports: dict[str, dict | None], lits: dict[str, dict | None]
) -> tuple[list, list, list]:
    """Return the router's rows, failed ids and detail, a case per device."""
    cases: dict[str, dict] = {}
    failed, detail = [], []
    for stem, lit in lits.items():
        if lit is None:
            failed.append(f"{stem}: no report")
            continue
        bad = [
            t["name"].split(" :: ", 1)[-1]
            for t in lit["tests"]
            if t["code"] in LIT_FAILURES
        ]
        failed += [f"lit/{name}" for name in bad]
        detail.append({"run": "lit", "tests": len(lit["tests"]), "failed": bad})
    for stem, report in reports.items():
        if report is None:
            failed.append(f"{stem}: no report")
            continue
        device, first = report["device"], report["first_seed"]
        cells = cases.setdefault(device, {})
        if "properties" in report:
            bad = [p["name"] for p in report["properties"] if not p["ok"]]
            span = f"seed={first}+{report['seeds']}"
            failed += [f"{device}/properties/{span}/{name}" for name in bad]
            cells["failed_properties"] = (len(bad), "properties")
            for metric, unit in ROUTER_COUNTS.items():
                cells[metric] = (report["counts"][metric], unit)
            detail.append(
                {
                    "device": device,
                    "run": "properties",
                    "first_seed": first,
                    "seeds": report["seeds"],
                    "failed": bad,
                }
            )
            continue
        part = f"hops-{report['hops']}"
        seen = set()
        for f in report["failures"]:
            what = " ".join(x for x in (f["kind"], f["name"], f["outcome"]) if x)
            if what not in seen:
                seen.add(what)
                failed.append(f"{device}/{part}/seed={f['seed']}+1/{what}")
        failures = cells.get("failed_mutations", (0, ""))[0] + len(report["failures"])
        survived = cells.get("survived_corruptions", (0, ""))[0] + report["survived"]
        cells["failed_mutations"] = (failures, "cases")
        cells["survived_corruptions"] = (survived, "corruptions")
        detail.append(
            {
                "device": device,
                "run": part,
                "first_seed": first,
                "seeds": report["seeds"],
                "killed": report["killed"],
                "survived": report["survived"],
                "failures": len(report["failures"]),
            }
        )
    rows = [
        row(device, metric, value, unit)
        for device, cells in cases.items()
        for metric, (value, unit) in cells.items()
    ]
    return rows, failed, detail


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("check", choices=["allocator", "router"])
    parser.add_argument(
        "--report", action="append", required=True, type=Path, help="repeatable"
    )
    parser.add_argument(
        "--lit",
        action="append",
        default=[],
        type=Path,
        help="lit -o results of the router's nightly lit suite; repeatable",
    )
    parser.add_argument("--out", required=True, help="perf.json to write")
    parser.add_argument("--meta", required=True, help="meta.json to write")
    args = parser.parse_args(argv)

    reports = {path.stem: read(path) for path in args.report}
    if args.check == "allocator":
        rows, failed, detail = allocator(reports)
    else:
        rows, failed, detail = router(reports, {p.stem: read(p) for p in args.lit})
    for line in failed:
        print("FAILED:", line)
    with open(args.out, "w") as f:
        json.dump(rows, f, indent=1)
    meta = {
        "preflight": {},
        "provenance": f"commit {commit()}",
        "measurement_sane": True,
        "detail": detail,
        "failed": failed,
        "exitstatus": 1 if failed else 0,
    }
    with open(args.meta, "w") as f:
        json.dump(meta, f, indent=1)
    return meta["exitstatus"]


if __name__ == "__main__":
    sys.exit(main())
