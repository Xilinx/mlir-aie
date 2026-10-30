#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Publish one NPU's kernel check results on the publication branch.

Each NPU the hardware checks ran on (``npu1``, ``npu2``) has a directory on
the branch, ``kernel-checks/<npu>/``, and each component check
(``sa-placer`` and ``sa-placer-hw`` from nightlyComponentChecks.yml,
``allocator`` and ``router`` from modelBasedTests.yml) one at
``component-checks/<check>/``, holding:

    runs/<id>.json         one record per run: the Actions run, what started
                           it (``event``), commit, power mode, provenance,
                           sanity result, failures, and every row as
                           rows[case][metric] = {value, unit, range}, a
                           timed case's error beside its times; a
                           component check's adds its per-seed ``detail``
    runs.json              the records without their rows or detail, oldest
                           first, and the metrics there is a history of
    latest.json            the newest nightly record that published rows; the
                           PR report's baseline
    history/<metric>.json  one series per case for that metric, a value per
                           run, for the charts; each run names its part
                           (``device``), which host time is compared within

Standard library only: publishKernelResults.yml runs it from a copy beside
the page, where the package is not installed.

    publish.py perf --target npu1 --results results/npu1 --run-id 42
        --run-url https://github.com/.../actions/runs/42 --out kernel-checks/npu1
        [--event schedule|workflow_dispatch]
    publish.py migrate --out kernel-checks/npu1
    publish.py rebuild --out kernel-checks/npu1 [--drop-pmode default]
    publish.py backfill --out kernel-checks/npu1 --runs runs.jsonl
        [--keep-pmode turbo]
    publish.py redirect --out kernel-checks --to ../dashboard/#view=night

``perf`` reads the timing step's ``perf.json`` and ``meta.json``, the
correctness step's ``error.json`` (``--report-error``) and the catalogue.
Each timed case's error is charted as two more metrics per output dtype,
over its data cases (but the all-zero and all-one edge data) and seeds:
``max_ulp_<dtype>``, the furthest any result fell from the correctly rounded
mathematical reference, and ``not_correctly_rounded_<dtype>``, the
percentage of all those results, pooled, that were not. ``builds`` maps each timed case to its kernel's
``ExternalFunction.recipe_key``, so a design that calls the same kernel finds
its row. Every publish first migrates a directory that still holds a
github-action-benchmark ``data.js`` (one record per entry, then the file is
removed), upgrades any older run record (``upgrade``: a static
``sa-placer-hw`` record's cases, a kernel check's ``measurement``), rebuilds
``runs.json``, ``latest.json`` and the history from
the run records, and prunes: every run of the last ``KEEP_DAYS`` is kept, older
ones one per ISO week plus the selected comparison pair, ``MAX_RUNS`` at most.
``rebuild --drop-pmode MODE``
also deletes the records of every run measured in ``MODE``, for retiring a
power mode nobody should compare against; ``--keep-pmode MODE`` deletes
those of every run that charted numbers in any other mode, keeping a run
that refused to measure, so the page can still say it did.

``redirect`` writes an ``index.html`` that sends a browser on to ``--to``:
the component checks are a view of the kernel checks page, and their old
pages' links still land there.

A record is dated by its run's start (``--date``, from the Actions run's
``run_started_at``), so a run that queued for the NPU is charted when it
was started rather than when it published. ``backfill`` brings older records
up to that: given the workflow's runs, one JSON object per line as
``gh api .../actions/workflows/<file>/runs --paginate --jq
'.workflow_runs[] | {id, event, run_started_at}'`` prints them, it dates each
record of a listed run by its start and fills in a missing ``event``, then
rebuilds. Running it again changes nothing.

Both the scheduled nightly and an unfiltered dispatch on main publish. A
record's ``event`` is the GitHub event that started its run; a run started by
hand (``workflow_dispatch``) is charted like any other, but only a nightly
(``schedule``, or a record older than the field) becomes ``latest.json``, so
PR reports always compare with the last nightly.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Every run of the last KEEP_DAYS; then one a week for KEEP_WEEKS more (a
# year of trend), then none. MAX_RUNS bounds the lot should a check run
# many times a day.
KEEP_DAYS = 90
KEEP_WEEKS = 52
MAX_RUNS = 200
# A case none of the last RETIRED_AFTER published runs measured is retired:
# its series leaves the history rather than trailing nulls for a year.
RETIRED_AFTER = 14
# What a record carries for the page's latest and previous run only (a
# hardware check's per-seed detail, a kernel leg's builds): a record thinned
# to a week's keeps its rows and summary, which are all the history needs.
HEAVY_FIELDS = ("detail", "builds")
# The format of every file this writes. A reader that knows an older one
# refuses the file rather than misreading it; bump it with any change a
# reader of the old format would get wrong.
SCHEMA = 1
REPO = "https://github.com/Xilinx/mlir-aie"
COMPONENTS = (
    "sa-placer",
    "sa-placer-hw",
    "allocator",
    "router",
    "llama-chain-hw",
    "yolo26n-hw",
)
# The dashboard section each component check's page redirects to.
SECTION_OF = {
    "sa-placer": "placement",
    "sa-placer-hw": "performance",
    "allocator": "placement",
    "router": "routing",
    "llama-chain-hw": "performance",
    "yolo26n-hw": "performance",
}
TARGETS = ("npu1", "npu2") + COMPONENTS
# Provenance fields worth a column in the history (the rest stay in the record).
HISTORY_PROVENANCE = (
    "peano",
    "host",
    "runtime",
    "xrt",
    "xdna",
    "kernels",
    "device",
    "fixtures",
    "dispatch",
    "measurement",
)
# The component checks' rows from before they were published here, renamed to
# what they are called now. Their other rows (wall time and RSS over the one
# small fixture) measured something the checks no longer do, and are dropped.
LEGACY_ROWS = (
    (r"sa_placer/fail_count", "test_sa_effort/failed_seeds"),
    (r"sa_placer/(mean|max)_final_cost", r"test_sa_effort/final_cost_\1"),
    (r"sa_placer/hw_fail_count", "mobilenet/failed_seeds"),
    (
        r"sa_placer/hw_seed(\d+)_latency_us",
        r"mobilenet/seed=\1/batch=1/us_per_image",
    ),
)
# An sa-placer-hw case from before the check swept the SA effort and compiled
# a static design per batch. Every one of those runs placed at effort 1.0, the
# page's "full".
STATIC_CASE = re.compile(r"mobilenet/seed=(?P<seed>\d+)(?:/batch=(?P<batch>\d+))?")
# The night #3807's trace decoder, which counts every timer wrap an Event_Sync
# repeat folds in, first measured npu2. Before it, the cases below, whose
# calls run past two wraps, decoded 1.5 to 7.2 times too few cycles; the
# wraps lost can't be recovered, so their points are dropped.
DECODER_FIXED = "2026-10-02"
UNDERSTATED = frozenset(
    {
        "bn_conv2dk1_i8/2352x8/uint8_int8/input_channels=336/input_width=7/output_channels=80",
        "bn_conv2dk1_relu/1792x8/int8_uint8/input_channels=16/input_width=112/output_channels=64",
        "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=40/input_width=28/output_channels=120",
        "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40/skip_dtype=int8",
        "bn_conv2dk1_skip/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40",
        "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=120/input_width=28/output_channels=40",
        "bn_conv2dk1_relu/1120x8/int8_uint8/input_channels=80/input_width=14/output_channels=184",
        "bn_conv2dk1_skip/2576x8/uint8_int8/input_channels=184/input_width=14/output_channels=80/skip_dtype=int8",
        "bn_conv2dk1_i8/3360x8/uint8_int8/input_channels=240/input_width=14/output_channels=80",
        "bn_conv2dk3/1792x8/int8_uint8/input_channels=8/input_width=224/output_channels=16",
    }
)
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


# Edge data whose every result is right or wrong together: pooled with the
# rest, it would swamp the percentage.
TRIVIAL_DATA = ("zeros", "ones")


def error_rows(report: dict) -> dict[str, dict[str, dict]]:
    """Each case's error metrics in a ``--report-error`` report.

    The report is keyed ``<case>/<data>/s<seed>``, and each run lists its
    outputs. Per case and output dtype, ``max_ulp_<dtype>`` is the worst
    over its data cases, seeds and outputs, and ``not_correctly_rounded_
    <dtype>`` is pooled over all of them; ``TRIVIAL_DATA`` is left out. A
    row's range names the reference it was measured against, and says when
    the check itself judges the build by a model of the kernel.
    """
    totals: dict[tuple[str, str], dict] = {}
    for key, run in report.items():
        case, data, _ = key.rsplit("/", 2)
        if data in TRIVIAL_DATA:
            continue
        for entry in run["outputs"]:
            t = totals.setdefault(
                (case, entry["dtype"]),
                {"ulp": 0, "wrong": 0, "n": 0, "references": set(), "modeled": False},
            )
            t["ulp"] = max(t["ulp"], entry["max_ulp"])
            t["wrong"] += entry["not_correctly_rounded"]
            t["n"] += entry["n"]
            t["references"].add(entry["reference"])
            t["modeled"] |= entry["modeled"]
    rows: dict[str, dict[str, dict]] = {}
    for (case, dtype), t in totals.items():
        if not t["n"]:
            continue
        spread = (
            f"vs the {', '.join(sorted(t['references']))} correctly rounded reference"
        )
        if t["modeled"]:
            spread += "; the check judges this build by a model of the kernel"
        rows.setdefault(case, {}).update(
            {
                f"max_ulp_{dtype}": {"value": t["ulp"], "unit": "ulp", "range": spread},
                f"not_correctly_rounded_{dtype}": {
                    "value": round(100 * t["wrong"] / t["n"], 3),
                    "unit": "%",
                    "range": spread,
                },
            }
        )
    return rows


def _load(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default


def commit_info(sha: str, message: str = "", timestamp: str = "") -> dict:
    """Return the commit as the page shows it: its subject line, and its date.

    Both come from git when not given; the page shows only the subject, which
    every history file repeats per run, so the body is not kept.
    """
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
        "message": message.split("\n", 1)[0],
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
            "flaky": sum(len(k.get("flaky", [])) for k in kernels),
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
    errors = error_rows(_load(results / "error.json", {}))
    catalogue = _load(results / "catalogue.json", None)
    preflight = meta.get("preflight") or {}
    provenance = provenance_fields(meta.get("provenance", ""))
    cases = meta.get("cases") or {}
    record = {
        "schema": SCHEMA,
        "target": target,
        **run,
        "pmode": preflight.get("pmode"),
        # A check without an NPU times the CPU it ran on.
        "device": device_part(preflight.get("device")) or provenance.get("cpu"),
        "device_raw": preflight.get("device"),
        "provenance": provenance,
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
        "rows": {},
    }
    if meta.get("measurement_sane") is True:
        record["rows"] = {
            case: {**cells, **errors.get(case, {})}
            for case, cells in rows_by_case(rows).items()
        }
        record["builds"] = {
            case: d["build"]
            for case, d in cases.items()
            if case in record["rows"] and "build" in d
        }
    for key in (
        "correctness_error",
        "refused",
        "reason",
        "efforts",
        "iters",
        "detail",
    ):
        if key in meta:
            record[key] = meta[key]
    if catalogue:
        record.update(_summary_of_catalogue(catalogue))
    return record


MANUAL_EVENTS = ("workflow_dispatch",)


def is_nightly(record: dict) -> bool:
    """Whether ``record`` is a scheduled run; one without ``event`` predates it."""
    return record.get("event") not in MANUAL_EVENTS


def summary(record: dict) -> dict:
    """Return the record without its rows, builds or detail, for ``runs.json``."""
    return {k: v for k, v in record.items() if k not in ("rows", "builds", "detail")}


def _current_name(name: str) -> str | None:
    for pattern, current in LEGACY_ROWS:
        if re.fullmatch(pattern, name):
            return re.sub(pattern, current, name)
    return None if name.startswith("sa_placer/") else name


def upgrade(record: dict) -> bool:
    """Bring an older run record up to the current case ids; return whether it changed.

    An ``sa-placer-hw`` record from before the check swept the SA effort and
    compiled one ``--dynamic`` design for every batch has no ``dispatch`` in
    its provenance; it gets ``dispatch static``. Its placement cost
    (``mobilenet/seed=S``) and failures become ``.../effort=full``, and its
    batch 1 compile time becomes ``mobilenet/seed=S/effort=full``'s, the one
    compile a seed takes now. Its other per-batch compile times go, and its
    per-batch times stay under their own ids (``mobilenet/seed=S/batch=B``):
    a static design's times are not the dynamic one's. Its runs get
    ``effort_label`` ``full``.

    A record with a ``device_raw`` but no ``device`` in its provenance gets
    it there, where the page keeps each part's times apart.

    A kernel check's record gets the ``measurement`` that took its cycles:
    ``1`` before ``DECODER_FIXED``, ``2`` since. An npu2 record from before
    it loses the cycles of the ``UNDERSTATED`` cases. A kernel check's
    ``npu_us`` from before the check published ``dispatch_us`` was submit to
    completion, the dispatch's fixed cost included, so it becomes that
    case's ``dispatch_us``; ``npu_us`` is now the dispatch less an empty
    one's.

    Other records, and records already current, are left as they are.
    """
    before = json.dumps(record, sort_keys=True)
    target = record.get("target")
    provenance = record.get("provenance") or {}
    rows = record.get("rows", {})
    if record.get("device_raw") and "device" not in provenance:
        provenance = record["provenance"] = {
            **provenance,
            "device": record["device_raw"],
        }
    if target == "sa-placer-hw" and "dispatch" not in provenance:
        record["provenance"] = {**provenance, "dispatch": "static"}
        upgraded: dict[str, dict] = {}
        for case, cells in rows.items():
            static = STATIC_CASE.fullmatch(case)
            if not static:
                upgraded[case] = cells
                continue
            seed = f"mobilenet/seed={static['seed']}/effort=full"
            if static["batch"] is None:
                upgraded.setdefault(seed, {}).update(cells)
                continue
            cells = dict(cells)
            compiled = cells.pop("compile_s", None)
            if compiled and static["batch"] == "1":
                upgraded.setdefault(seed, {})["compile_s"] = compiled
            if cells:
                upgraded[case] = cells
        if "rows" in record:
            record["rows"] = upgraded
        record["failed"] = [
            f"{case}/effort=full" if STATIC_CASE.fullmatch(case) else case
            for case in record.get("failed", [])
        ]
        for run in record.get("detail", []):
            run.setdefault("effort_label", "full")
    if target in ("npu1", "npu2") and "measurement" not in provenance:
        decoded_before = record["date"] < DECODER_FIXED
        record["provenance"] = {
            **provenance,
            "measurement": "1" if decoded_before else "2",
        }
        if target == "npu2" and decoded_before:
            for case in UNDERSTATED & rows.keys():
                for metric in ("cycles", "cycles_per_kop"):
                    rows[case].pop(metric, None)
    if target in ("npu1", "npu2"):
        for cells in rows.values():
            if "npu_us" in cells and "dispatch_us" not in cells:
                cells["dispatch_us"] = cells.pop("npu_us")
    return json.dumps(record, sort_keys=True) != before


def migrate(out: Path) -> int:
    """Turn a github-action-benchmark ``data.js`` into run records; return how many.

    Each entry of each suite ("aie_kernels (<npu>, <mode>)") becomes
    ``runs/bench-<date>.json`` with the mode from the suite name and the
    provenance from the rows' ``extra``; a component check's rows are
    renamed by ``LEGACY_ROWS``. The file and the action's own page are
    removed afterwards. Nothing happens when there is no ``data.js``.
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
            benches = [
                {**b, "name": name}
                for b in entry.get("benches", [])
                if (name := _current_name(b["name"]))
            ]
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
                    "message": commit.get("message", "").split("\n", 1)[0],
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


def _stands_for_week(record: dict) -> tuple:
    """How well a run stands for its week once thinned, higher better.

    A published run (one that charted numbers) before one that did not (a
    refused or failed night), a nightly before a run started by hand, then
    the newest: a week's charted point is never lost to a newer run that
    charted nothing.
    """
    published = bool(record.get("published") and record.get("rows"))
    return (published, is_nightly(record), record["date"])


def prune(
    records: list[dict], now: datetime.datetime, max_runs: int = MAX_RUNS
) -> list[dict]:
    """Keep recent runs and weekly representatives within the horizon.

    Keep every run of the last ``KEEP_DAYS``; then, for ``KEEP_WEEKS`` more,
    one run each ISO week (the one that best stands for it); none older,
    and ``max_runs`` at most.
    """
    records = sorted(records, key=lambda r: r["date"])
    recent, weekly = [], {}
    for r in records:
        age = now - parse_date(r["date"])
        if age.days < KEEP_DAYS:
            recent.append(r)
        elif age.days < KEEP_DAYS + 7 * KEEP_WEEKS:
            week = parse_date(r["date"]).isocalendar()[:2]
            if week not in weekly or _stands_for_week(r) >= _stands_for_week(
                weekly[week]
            ):
                weekly[week] = r
    kept = sorted(weekly.values(), key=lambda r: r["date"]) + recent
    return kept[-max_runs:]


def rebuild(
    out: Path,
    now: datetime.datetime | None = None,
    drop_pmode: str | None = None,
    keep_pmode: str | None = None,
    max_runs: int = MAX_RUNS,
) -> dict:
    """Rewrite the derived files from ``runs/*.json``, pruning old runs.

    A record still using older case ids is upgraded in place first. With
    ``drop_pmode``, the records of runs measured in that power mode are
    deleted first; with ``keep_pmode``, those of runs that charted numbers in
    any other mode. At most ``max_runs`` records are kept.
    """
    now = now or now_utc()
    runs_dir = out / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    records = [
        _check_schema(json.loads(p.read_text()), p) for p in runs_dir.glob("*.json")
    ]
    for r in records:
        if upgrade(r):
            (runs_dir / f"{r['id']}.json").write_text(json.dumps(r, indent=1))

    def dropped(r: dict) -> bool:
        if drop_pmode and r.get("pmode") == drop_pmode:
            return True
        return bool(keep_pmode and r.get("rows") and r.get("pmode") != keep_pmode)

    for r in records:
        if dropped(r):
            (runs_dir / f"{r['id']}.json").unlink()
    records = sorted((r for r in records if not dropped(r)), key=lambda r: r["date"])
    eligible = [
        r
        for r in records
        if (now - parse_date(r["date"])).days < KEEP_DAYS + 7 * KEEP_WEEKS
    ]
    published = [r for r in eligible if r.get("published") and r.get("rows")]
    # The newest nightly is the baseline; only manual runs, the newest of those.
    baseline = [r for r in published if is_nightly(r)] or published
    selected = baseline[-1] if baseline else None
    # A record past KEEP_DAYS sheds what only the latest runs are read for;
    # protect the actual latest.json and its index.html previousPublished.
    needed = set()
    if selected:
        needed.add(selected["id"])
        before = [
            r
            for r in reversed(eligible[: eligible.index(selected)])
            if r.get("published")
            and r.get("pmode") == selected.get("pmode")
            and is_nightly(r)
        ]
        previous = next(
            (
                r
                for r in before
                if selected.get("device") and r.get("device") == selected["device"]
            ),
            before[0] if before else None,
        )
        if previous:
            needed.add(previous["id"])
    # Select before weekly thinning: two comparison nights can share a week,
    # and a long tail of refusals or manual runs must not evict their details.
    protected = [r for r in eligible if r["id"] in needed]
    others = [r for r in prune(records, now, max_runs) if r["id"] not in needed]
    slots = max_runs - len(protected)
    kept = sorted(
        protected + (others[-slots:] if slots else []), key=lambda r: r["date"]
    )
    keep_ids = {r["id"] for r in kept}
    for r in records:
        if r["id"] not in keep_ids:
            (runs_dir / f"{r['id']}.json").unlink(missing_ok=True)
    target = kept[-1]["target"] if kept else out.name
    published = [r for r in kept if r.get("published") and r.get("rows")]
    for r in kept:
        old = (now - parse_date(r["date"])).days >= KEEP_DAYS
        if old and r["id"] not in needed and any(f in r for f in HEAVY_FIELDS):
            for f in HEAVY_FIELDS:
                r.pop(f, None)
            (runs_dir / f"{r['id']}.json").write_text(json.dumps(r, indent=1))
    latest = out / "latest.json"
    if selected:
        latest.write_text(json.dumps(selected, indent=1))
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
            "device": r.get("device"),
            **({"event": r["event"]} if r.get("event") else {}),
            "provenance": {
                k: v
                for k, v in r.get("provenance", {}).items()
                if k in HISTORY_PROVENANCE
            },
        }
        for r in published
    ]
    # A case the last RETIRED_AFTER published runs did not measure is retired;
    # a metric left with no case goes with it.
    if len(published) >= RETIRED_AFTER:
        for metric in list(metrics):
            series = metrics[metric]["series"]
            for case in [
                c
                for c, s in series.items()
                if all(v is None for v in s["values"][-RETIRED_AFTER:])
            ]:
                del series[case]
            if not series:
                del metrics[metric]
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
    (out / "runs.json").write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "target": target,
                "metrics": sorted(metrics),
                "runs": [summary(r) for r in kept],
            },
            indent=1,
        )
    )
    return {"runs": len(kept), "published": len(published), "metrics": sorted(metrics)}


def read_runs(path: Path) -> dict[str, dict]:
    """Read the workflow's runs, one JSON object per line, by run id."""
    runs = {}
    for line in path.read_text().splitlines():
        if line.strip():
            run = json.loads(line)
            runs[str(run["id"])] = run
    return runs


def backfill(out: Path, runs: dict[str, dict]) -> int:
    """Date each record of a listed run by its start, fill in its ``event``.

    Returns how many records changed; the rest, including migrated
    ``bench-*`` records, which no run lists, are left as they are.
    """
    changed = 0
    for path in sorted((out / "runs").glob("*.json")):
        record = _check_schema(json.loads(path.read_text()), path)
        run = runs.get(str(record.get("id")))
        if not run:
            continue
        before = dict(record)
        if run.get("run_started_at"):
            record["date"] = iso(parse_date(run["run_started_at"]))
        if run.get("event") and not record.get("event"):
            record["event"] = run["event"]
        if record != before:
            path.write_text(json.dumps(record, indent=1))
            changed += 1
    return changed


def redirect(out: Path, to: str) -> Path:
    """Write ``out/index.html``, sending a browser on to ``to``.

    A link that carries its own ``#...`` keeps it, so an old link to one
    view lands on that view; without one, ``to``'s own hash is used.
    """
    out.mkdir(parents=True, exist_ok=True)
    page = out / "index.html"
    base = to.split("#", 1)[0]
    script = (
        f"<script>location.replace({json.dumps(base)} + "
        f"(location.hash || {json.dumps(to[len(base):])}));</script>\n"
    )
    page.write_text(
        "<!DOCTYPE html>\n"
        '<meta charset="utf-8">\n'
        "<title>Maintainer dashboard</title>\n"
        + script
        + f'<meta http-equiv="refresh" content="0; url={to}">\n'
        f'<link rel="canonical" href="{to}">\n'
        f'<p>This page moved to <a href="{to}">{to}</a>.</p>\n'
    )
    return page


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
    p.add_argument(
        "--date",
        default="",
        help="when the run started, ISO 8601 (default: now)",
    )
    p.add_argument(
        "--event",
        default=os.environ.get("GITHUB_EVENT_NAME", ""),
        help="the GitHub event that started the run (default: $GITHUB_EVENT_NAME)",
    )
    p = sub.add_parser("migrate", help="turn a data.js into run records, once")
    p.add_argument("--out", required=True, type=Path)
    p = sub.add_parser("rebuild", help="rewrite the derived files from the records")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--drop-pmode", help="delete the records of runs in this power mode")
    p = sub.add_parser(
        "backfill", help="date records by their run's start, fill in events, rebuild"
    )
    p.add_argument("--out", required=True, type=Path)
    p.add_argument(
        "--runs",
        required=True,
        type=Path,
        help="the workflow's runs, one {id, event, run_started_at} per line",
    )
    p.add_argument(
        "--keep-pmode",
        help="delete the records of runs that charted numbers in any other mode",
    )
    p = sub.add_parser("redirect", help="write an index.html that sends on to --to")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--to", required=True, help="where to send the browser")
    args = parser.parse_args(argv)

    if args.command == "redirect":
        print(f"wrote {redirect(args.out, args.to)} -> {args.to}")
        return 0
    if args.command == "migrate":
        n = migrate(args.out)
        if n or (args.out / "runs").exists():
            rebuild(args.out)
        print(f"migrated {n} runs into {args.out}")
        return 0
    if args.command == "rebuild":
        print(json.dumps(rebuild(args.out, drop_pmode=args.drop_pmode)))
        return 0
    if args.command == "backfill":
        n = backfill(args.out, read_runs(args.runs))
        built = rebuild(args.out, keep_pmode=args.keep_pmode)
        print(f"backfilled {n} records in {args.out}: {json.dumps(built)}")
        return 0
    run = {
        "id": args.run_id,
        "url": args.run_url,
        "date": iso(parse_date(args.date)) if args.date else iso(now_utc()),
        "commit": commit_info(args.commit, args.commit_message, args.commit_date),
        **({"event": args.event} if args.event else {}),
    }
    record = publish(args.out, args.results, target=args.target, run=run)
    print(
        f"{args.command} {args.target}: run {record['id']}, {record['n_rows']} rows, "
        f"published={record['published']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
