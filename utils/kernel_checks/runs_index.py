#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record one NPU's run on the publication branch, beside its series.

benchmark-action keeps one value per row and a text ``extra``; what a run
was (which Actions run, in which power mode, on which host, whether its
sanity check passed, which cases failed or were truncated) lives in the
timing step's ``meta.json``, which was only ever an artifact. This installs
it as ``<out-dir>/latest.json`` and appends a summary of it to
``<out-dir>/runs.json``, which the results page reads for its dashboard.
Standard library only: publishKernelResults.yml runs it from a copy on the
publication branch, where the package is not installed.

    runs_index.py --npu npu1 --meta results/npu1/meta.json
        --catalogue results/npu1/catalogue.json --run-id 123
        --run-url https://github.com/.../actions/runs/123 --out-dir kernel-checks/npu1
"""

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

MAX_RUNS = 400  # as many as the series keep (max-items-in-chart)


def provenance_fields(line: str) -> dict[str, str]:
    """Split ``"commit abc | peano 22.0.0+e1 | host bench-3"`` into its fields."""
    fields = {}
    for part in (line or "").split(" | "):
        key, _, value = part.strip().partition(" ")
        if key and value:
            fields[key] = value
    return fields


def summarize(meta: dict, catalogue: dict | None = None) -> dict:
    """Return the runs.json entry for one ``meta.json`` (and the catalogue, if any)."""
    preflight = meta.get("preflight") or {}
    cases = meta.get("cases") or {}
    entry = {
        "pmode": preflight.get("pmode"),
        "device": preflight.get("device"),
        "provenance": provenance_fields(meta.get("provenance", "")),
        "sane": meta.get("measurement_sane") is True,
        "published": meta.get("measurement_sane") is True and bool(meta.get("n_rows")),
        "n_rows": meta.get("n_rows", 0),
        "exitstatus": meta.get("exitstatus"),
        "failed": list(meta.get("failed", [])),
        "truncated": sorted(
            name
            for name, d in cases.items()
            if (d.get("cycles") or {}).get("truncated")
        ),
    }
    if "correctness_error" in meta:
        entry["correctness_error"] = meta["correctness_error"]
    if catalogue:
        kernels = catalogue.get("kernels", [])
        entry["cases"] = {
            "passed": sum(k.get("passed", 0) for k in kernels),
            "failed": sum(len(k.get("failed", [])) for k in kernels),
            "timed": sum(k.get("timed", 0) for k in kernels),
            "timing_failed": sum(len(k.get("timing_failed", [])) for k in kernels),
            "untimed": sum(len(k.get("untimed", [])) for k in kernels),
        }
        entry["kernels"] = {
            "offered": sum(1 for k in kernels if k.get("builds")),
            "checked": sum(1 for k in kernels if k.get("passed") or k.get("failed")),
        }
    return entry


def record(
    index_path: Path,
    entry: dict,
    *,
    npu: str,
    run_id: str,
    run_url: str,
    commit: str,
    date: str,
) -> dict:
    """Append ``entry`` to the index at ``index_path`` (replacing a rerun's) and return it."""
    if index_path.exists():
        index = json.loads(index_path.read_text())
    else:
        index = {"npu": npu, "runs": []}
    entry = {"id": run_id, "url": run_url, "date": date, "commit": commit, **entry}
    runs = [r for r in index.get("runs", []) if r.get("id") != run_id]
    runs.append(entry)
    index["npu"] = npu
    index["runs"] = runs[-MAX_RUNS:]
    index_path.write_text(json.dumps(index, indent=1))
    return index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--npu", required=True)
    parser.add_argument("--meta", required=True, type=Path)
    parser.add_argument("--catalogue", type=Path)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    meta = json.loads(args.meta.read_text())
    catalogue = json.loads(args.catalogue.read_text()) if args.catalogue else None
    commit = os.environ.get("GITHUB_SHA", "")
    date = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "latest.json").write_text(
        json.dumps(
            {
                "id": args.run_id,
                "url": args.run_url,
                "date": date,
                "commit": commit,
                **meta,
            },
            indent=1,
        )
    )
    record(
        args.out_dir / "runs.json",
        summarize(meta, catalogue),
        npu=args.npu,
        run_id=args.run_id,
        run_url=args.run_url,
        commit=commit,
        date=date,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
