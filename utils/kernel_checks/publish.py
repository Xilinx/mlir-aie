#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Publish one NPU's kernel check results on the publication branch.

Each NPU the hardware checks ran on (``npu1``, ``npu2``) has a directory on
the branch, ``kernel-checks/<npu>/``, holding:

    runs/<id>.json         one record per run: the Actions run, commit, power
                           mode, provenance, sanity result, failures, and
                           every row as rows[case][metric] = {value, unit, range}
    runs.json              the records without their rows, oldest first
    latest.json            the newest record that published rows; the PR
                           report's baseline
    history/<metric>.json  one series per case for that metric, a value per
                           run, for the charts

Standard library only: publishKernelResults.yml runs it from a copy beside
the page, where the package is not installed.

    publish.py perf --target npu1 --results results/npu1 --run-id 42
        --run-url https://github.com/.../actions/runs/42 --out kernel-checks/npu1
    publish.py migrate --out kernel-checks/npu1
    publish.py rebuild --out kernel-checks/npu1 [--drop-pmode default]

``perf`` reads the timing step's ``perf.json`` and ``meta.json`` and the
catalogue. Every publish first migrates a directory that still holds a
github-action-benchmark ``data.js`` (one record per entry, then the file is
removed), rebuilds ``runs.json``, ``latest.json`` and the history from the
run records, and prunes: every run of the last ``KEEP_DAYS`` is kept, older
ones one per ISO week, ``MAX_RUNS`` at most. ``rebuild --drop-pmode MODE``
also deletes the records of every run measured in ``MODE``, for retiring a
power mode nobody should compare against.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

KEEP_DAYS = 90
MAX_RUNS = 400
# The format of every file this writes. A reader that knows an older one
# refuses the file rather than misreading it; bump it with any change a
# reader of the old format would get wrong.
SCHEMA = 1
REPO = "https://github.com/Xilinx/mlir-aie"
TARGETS = ("npu1", "npu2")
# Provenance fields worth a column in the history (the rest stay in the record).
HISTORY_PROVENANCE = ("peano", "host", "xrt", "xdna", "kernels", "device")
# The part a runtime's device name means, first match wins. XRT names the
# same NPU differently across drivers ("RyzenAI-npu1", "NPU Phoenix"); the
# raw name stays in the record as ``device_raw``.
PARTS = (
    ("Strix Halo", ("strix halo", "npu5")),
    ("Strix", ("strix", "npu4")),
    ("Krackan", ("krackan", "npu6")),
    ("Gorgon Point", ("gorgon point",)),
    ("Phoenix", ("phoenix", "npu1")),
)


def device_part(raw: str | None) -> str | None:
    """Return the NPU part a runtime's device name means, or the name itself."""
    if not raw:
        return raw
    lowered = raw.lower()
    for part, needles in PARTS:
        if any(n in lowered for n in needles):
            return part
    return raw


class NewerSchema(Exception):
    """A file on the branch was written by a newer publish.py."""


def _check_schema(data: dict, where: Path) -> dict:
    if data.get("schema", 1) > SCHEMA:
        raise NewerSchema(
            f"{where} is schema {data['schema']}; this publish.py writes {SCHEMA}"
        )
    return data


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def parse_date(text: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def provenance_fields(line: str) -> dict[str, str]:
    """Split ``"commit abc | peano 22.0.0+e1 | host bench-3"`` into its fields."""
    fields = {}
    for part in (line or "").split(" | "):
        key, _, value = part.strip().partition(" ")
        if key and value:
            fields[key] = value
    return fields


def rows_by_case(rows: list[dict]) -> dict[str, dict[str, dict]]:
    """Group benchmark-style rows (``name`` = ``<case>/<metric>``) by case and metric."""
    out: dict[str, dict[str, dict]] = {}
    for row in rows:
        case, _, metric = row["name"].rpartition("/")
        cell = {"value": row["value"], "unit": row.get("unit", "")}
        if row.get("range"):
            cell["range"] = row["range"]
        out.setdefault(case, {})[metric] = cell
    return out


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def commit_info(sha: str, message: str = "", timestamp: str = "") -> dict:
    """Return the commit as the page shows it; message and date from git when not given."""
    if sha and not (message and timestamp):
        try:
            out = subprocess.run(
                ["git", "log", "-1", "--format=%s%n%cI", sha],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.splitlines()
            message = message or (out[0] if out else "")
            timestamp = timestamp or (out[1] if len(out) > 1 else "")
        except (OSError, subprocess.CalledProcessError):
            pass
    return {
        "id": sha,
        "url": f"{REPO}/commit/{sha}" if sha else "",
        "message": message,
        "timestamp": timestamp,
    }


def _summary_of_catalogue(catalogue: dict) -> dict:
    kernels = catalogue.get("kernels", [])
    return {
        "cases": {
            "passed": sum(k.get("passed", 0) for k in kernels),
            "failed": sum(len(k.get("failed", [])) for k in kernels),
            "timed": sum(k.get("timed", 0) for k in kernels),
            "timing_failed": sum(len(k.get("timing_failed", [])) for k in kernels),
            "untimed": sum(len(k.get("untimed", [])) for k in kernels),
        },
        "kernels": {
            "offered": sum(1 for k in kernels if k.get("builds")),
            "checked": sum(1 for k in kernels if k.get("passed") or k.get("failed")),
        },
    }


def record_perf(results: Path, *, target: str, run: dict) -> dict:
    """Return the record of one hardware run from its ``results`` directory."""
    meta = _load(results / "meta.json", {})
    rows = _load(results / "perf.json", [])
    catalogue = _load(results / "catalogue.json", None)
    preflight = meta.get("preflight") or {}
    cases = meta.get("cases") or {}
    record = {
        "schema": SCHEMA,
        "target": target,
        **run,
        "pmode": preflight.get("pmode"),
        "device": device_part(preflight.get("device")),
        "device_raw": preflight.get("device"),
        "provenance": provenance_fields(meta.get("provenance", "")),
        "sane": meta.get("measurement_sane"),
        "published": meta.get("measurement_sane") is True and bool(rows),
        "n_rows": len(rows),
        "exitstatus": meta.get("exitstatus"),
        "failed": list(meta.get("failed", [])),
        "truncated": sorted(
            name
            for name, d in cases.items()
            if (d.get("cycles") or {}).get("truncated")
        ),
        "rows": rows_by_case(rows) if meta.get("measurement_sane") is True else {},
    }
    for key in ("correctness_error", "refused"):
        if key in meta:
            record[key] = meta[key]
    if catalogue:
        record.update(_summary_of_catalogue(catalogue))
    return record


def summary(record: dict) -> dict:
    """Return the record without its rows, for ``runs.json``."""
    return {k: v for k, v in record.items() if k != "rows"}


def migrate(out: Path) -> int:
    """Turn a github-action-benchmark ``data.js`` into run records; return how many.

    Each entry of each suite ("aie_kernels (<npu>, <mode>)") becomes ``runs/bench-<date>.json`` with the mode from the
    suite name and the provenance from the rows' ``extra``. The file and
    the action's own page are removed afterwards. Nothing happens when
    there is no ``data.js``.
    """
    source = out / "data.js"
    if not source.exists():
        return 0
    text = re.sub(r"^\s*window\.BENCHMARK_DATA\s*=\s*", "", source.read_text())
    data = json.loads(re.sub(r";\s*$", "", text))
    target = out.name
    count = 0
    (out / "runs").mkdir(parents=True, exist_ok=True)
    for suite, entries in data.get("entries", {}).items():
        mode = re.match(r"^aie_kernels \(\w+, (.+)\)$", suite)
        for entry in entries:
            benches = entry.get("benches", [])
            commit = entry.get("commit", {})
            date = datetime.datetime.fromtimestamp(
                entry["date"] / 1000, datetime.timezone.utc
            )
            run_id = f"bench-{entry['date']}"
            raw = (
                provenance_fields(benches[0].get("extra", "")).get("device")
                if benches
                else None
            )
            record = {
                "schema": SCHEMA,
                "target": target,
                "id": run_id,
                "url": "",
                "date": iso(date),
                "commit": {
                    "id": commit.get("id", ""),
                    "url": commit.get("url", ""),
                    "message": commit.get("message", ""),
                    "timestamp": commit.get("timestamp", ""),
                },
                "pmode": mode[1] if mode else None,
                "device": device_part(raw),
                "device_raw": raw,
                "provenance": (
                    provenance_fields(benches[0].get("extra", "")) if benches else {}
                ),
                "sane": True,
                "published": bool(benches),
                "n_rows": len(benches),
                "exitstatus": None,
                "failed": [],
                "truncated": sorted(
                    b["name"].rpartition("/")[0]
                    for b in benches
                    if b["name"].endswith("/cycles")
                    and (b.get("range") or "").endswith("truncated")
                ),
                "migrated_from": suite,
                "rows": rows_by_case(benches),
            }
            path = out / "runs" / f"{run_id}.json"
            if not path.exists():
                path.write_text(json.dumps(record, indent=1))
                count += 1
    source.unlink()
    page = out / "index.html"
    if page.exists() and "BENCHMARK_DATA" in page.read_text():
        page.unlink()
    return count


def prune(records: list[dict], now: datetime.datetime) -> list[dict]:
    """Every run of the last ``KEEP_DAYS``; before that the newest of each ISO week."""
    records = sorted(records, key=lambda r: r["date"])
    recent, weekly = [], {}
    for r in records:
        age = now - parse_date(r["date"])
        if age.days < KEEP_DAYS:
            recent.append(r)
        else:
            weekly[parse_date(r["date"]).isocalendar()[:2]] = r
    kept = sorted(weekly.values(), key=lambda r: r["date"]) + recent
    return kept[-MAX_RUNS:]


def rebuild(
    out: Path, now: datetime.datetime | None = None, drop_pmode: str | None = None
) -> dict:
    """Rewrite the derived files from ``runs/*.json``, pruning old runs.

    With ``drop_pmode``, the records of runs measured in that power mode are
    deleted first.
    """
    now = now or now_utc()
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _check_schema(json.loads(p.read_text()), p) for p in runs_dir.glob("*.json")
    ]
    if drop_pmode:
        for r in records:
            if r.get("pmode") == drop_pmode:
                (runs_dir / f"{r['id']}.json").unlink()
        records = [r for r in records if r.get("pmode") != drop_pmode]
    kept = prune(records, now)
    keep_ids = {r["id"] for r in kept}
    for r in records:
        if r["id"] not in keep_ids:
            (runs_dir / f"{r['id']}.json").unlink(missing_ok=True)
    target = kept[-1]["target"] if kept else out.name
    (out / "runs.json").write_text(
        json.dumps(
            {"schema": SCHEMA, "target": target, "runs": [summary(r) for r in kept]},
            indent=1,
        )
    )
    published = [r for r in kept if r.get("published") and r.get("rows")]
    latest = out / "latest.json"
    if published:
        latest.write_text(json.dumps(published[-1], indent=1))
    elif latest.exists():
        latest.unlink()

    history_dir = out / "history"
    history_dir.mkdir(exist_ok=True)
    metrics: dict[str, dict] = {}
    for i, r in enumerate(published):
        for case, cells in r.get("rows", {}).items():
            for metric, cell in cells.items():
                h = metrics.setdefault(
                    metric, {"unit": cell.get("unit", ""), "series": {}}
                )
                s = h["series"].setdefault(
                    case, {"values": [None] * len(published), "ranges": None}
                )
                s["values"][i] = cell["value"]
                if cell.get("range"):
                    if s["ranges"] is None:
                        s["ranges"] = [None] * len(published)
                    s["ranges"][i] = cell["range"]
    runs_column = [
        {
            "id": r["id"],
            "date": r["date"],
            "commit": r.get("commit", {}),
            "pmode": r.get("pmode"),
            "provenance": {
                k: v
                for k, v in r.get("provenance", {}).items()
                if k in HISTORY_PROVENANCE
            },
        }
        for r in published
    ]
    for stale in history_dir.glob("*.json"):
        if stale.stem not in metrics:
            stale.unlink()
    for metric, h in metrics.items():
        series = {
            case: ({"values": s["values"]} if s["ranges"] is None else s)
            for case, s in sorted(h["series"].items())
        }
        (history_dir / f"{metric}.json").write_text(
            json.dumps(
                {
                    "schema": SCHEMA,
                    "target": target,
                    "metric": metric,
                    "unit": h["unit"],
                    "runs": runs_column,
                    "series": series,
                },
                separators=(",", ":"),
            )
        )
    return {"runs": len(kept), "published": len(published), "metrics": sorted(metrics)}


def publish(out: Path, results: Path, *, target: str, run: dict, now=None) -> dict:
    """Migrate, record one run, rebuild; return the record."""
    index = out / "runs.json"
    if index.exists():
        _check_schema(json.loads(index.read_text()), index)
    migrate(out)
    record = record_perf(results, target=target, run=run)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "runs" / f"{run['id']}.json").write_text(json.dumps(record, indent=1))
    rebuild(out, now)
    return record


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("perf", help="record one hardware run and rebuild the NPU")
    p.add_argument("--target", required=True, choices=TARGETS)
    p.add_argument("--results", required=True, type=Path)
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-url", default="")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    p.add_argument("--commit-message", default="")
    p.add_argument("--commit-date", default="")
    p.add_argument("--date", default="", help="ISO date of the run (default: now)")
    p = sub.add_parser("migrate", help="turn a data.js into run records, once")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("rebuild", help="rewrite the derived files from the records")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--drop-pmode", help="delete the records of runs in this power mode")
    args = parser.parse_args(argv)

    if args.command == "migrate":
        n = migrate(args.out)
        if n or (args.out / "runs").exists():
            rebuild(args.out)
        print(f"migrated {n} runs into {args.out}")
        return 0
    if args.command == "rebuild":
        print(json.dumps(rebuild(args.out, drop_pmode=args.drop_pmode)))
        return 0
    run = {
        "id": args.run_id,
        "url": args.run_url,
        "date": args.date or iso(now_utc()),
        "commit": commit_info(args.commit, args.commit_message, args.commit_date),
    }
    record = publish(args.out, args.results, target=args.target, run=run)
    print(
        f"{args.command} {args.target}: run {record['id']}, {record['n_rows']} rows, "
        f"published={record['published']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
