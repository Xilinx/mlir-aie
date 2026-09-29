#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Report a kernel checks run's failures and regressions against the last nightly.

For each NPU that produced results: the cases that failed, and the cases
whose cycles or core ELF size moved against the nightly baseline on main.
nightlyKernelChecks.yml writes it to the job summary and, on a Peano PR,
keeps it as one comment. Standard library only, so it runs on any runner.

    pr_report.py --results results --baselines baselines --out report.md

``results/<npu>/`` holds a leg's artifact (meta.json, perf.json,
correctness.xml); ``baselines/<npu>/latest.json`` is the record publish.py
wrote for the last nightly that published rows, which the publish job
caches. The thresholds are ``thresholds.json``'s, shared with the page.
"""

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

MARKER = "<!-- kernel-checks-report -->"
PAGE = "https://xilinx.github.io/mlir-aie/kernel-checks/"
THRESHOLDS = {
    metric: spec
    for metric, spec in json.loads(
        Path(__file__).with_name("thresholds.json").read_text()
    ).items()
    if not metric.startswith("_")
}
# Stable run to run, so a move is the compiler's: the regressions that count.
GATED = {m: s["pct"] / 100 for m, s in THRESHOLDS.items() if s.get("gated")}
# Derived from another metric, so it would repeat that metric's rows.
DERIVED = {m for m, s in THRESHOLDS.items() if s.get("derived")}
# Everything else (host timing moves with the machine) is listed past this.
OTHER = {
    m: s["pct"] / 100
    for m, s in THRESHOLDS.items()
    if m not in GATED and m not in DERIVED
}
DEFAULT_THRESHOLD = 0.10  # a metric thresholds.json does not name
# A change must also exceed this many MADs, per metric (npu_us).
MAD = {m: s["mad"] for m, s in THRESHOLDS.items() if s.get("mad")}
SCHEMA = 1  # the newest publish.py record format this reads
MAX_ROWS = 50

_EXTENSIVE = re.compile(r"test_kernel_extensive\[(.+)/([^/]+/s\d+)\]")
_MAD = re.compile(r"^\u00b1 ([\d.]+)")
_BRACKETED = re.compile(r"\[(.+)\]$")
_PEANO = re.compile(r"\bpeano (\S+)")


@dataclass
class Failure:
    case: str
    failed: list[str] = field(default_factory=list)
    total: int = 0
    reason: str = ""


@dataclass
class Change:
    case: str
    metric: str
    unit: str
    before: float
    after: float

    @property
    def ratio(self) -> float:
        return (self.after - self.before) / self.before


@dataclass
class Leg:
    npu: str
    measured: bool
    peano: str = ""
    baseline_peano: str = ""
    baseline_commit: dict | None = None
    cases: int = 0
    failures: list[Failure] = field(default_factory=list)
    regressed: list[Change] = field(default_factory=list)
    improved: list[Change] = field(default_factory=list)
    other: list[Change] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)


def _reason(test) -> str:
    """Return the line of a failure that says what went wrong, not where."""
    bad = test.find("failure")
    if bad is None:
        bad = test.find("error")
    text = bad.get("message") or bad.text or ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    line = next((line for line in lines if "error: " in line), None)
    if line is not None:
        line = line.split("error: ", 1)[1]
    else:
        line = lines[0] if lines else "no message"
    return line[:200] + ("\u2026" if len(line) > 200 else "")


def sweep(xml) -> Iterator[tuple[str, str, ET.Element]]:
    """Yield the case, variant and testcase of each input the extensive sweep ran."""
    for test in ET.parse(xml).iter("testcase"):
        match = _EXTENSIVE.fullmatch(test.get("name", ""))
        if match and test.find("skipped") is None:
            yield match[1], match[2], test


def failed(test) -> bool:
    return test.find("failure") is not None or test.find("error") is not None


def failures(directory: Path) -> tuple[list[Failure], set[str]]:
    """Return the failing cases, and every case the extensive sweep ran."""
    by_case: dict[str, Failure] = {}
    swept = set()
    xml = directory / "correctness.xml"
    if xml.exists():
        for case, variant, test in sweep(xml):
            swept.add(case)
            f = by_case.setdefault(case, Failure(case))
            f.total += 1
            if failed(test):
                f.failed.append(variant)
                f.reason = f.reason or _reason(test)
    meta = directory / "meta.json"
    if meta.exists():
        # The timing run checks each case again; its failures reach only meta.
        for nodeid in json.loads(meta.read_text()).get("failed", []):
            if "test_kernel_extensive[" in nodeid:
                continue
            match = _BRACKETED.search(nodeid)
            case = match[1] if match else nodeid
            f = by_case.setdefault(case, Failure(case))
            if not f.failed:
                f.reason = "failed in the timing run; see perf.log"
            f.failed.append("timing run")
    return (
        sorted((f for f in by_case.values() if f.failed), key=lambda f: f.case),
        swept,
    )


def baseline(path: Path, run_id: str = "") -> dict | None:
    """Return the last nightly's record (``latest.json``), or None.

    None without rows, in a newer format than this reads, or from this very
    run: the publish job of a nightly can cache its record before a re-run
    of the report restores it, and a run compared with itself shows nothing.
    """
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("schema", 1) > SCHEMA:
        return None
    if run_id and str(record.get("id", "")) == str(run_id):
        return None
    return record if record.get("rows") else None


def _mad(cell: dict) -> float | None:
    match = _MAD.match(cell.get("range") or "")
    return float(match[1]) if match else None


def moved(metric: str, threshold: float, before: dict, after: dict) -> bool:
    """Whether a change from ``before`` to ``after`` counts for ``metric``.

    At least ``threshold`` of the old value, and for a metric with a MAD
    factor, more than that many of the larger of the two rows' MADs too.
    """
    change = abs(after["value"] - before["value"])
    if change < threshold * abs(before["value"]):
        return False
    factor = MAD.get(metric)
    if factor:
        spread = [m for m in (_mad(before), _mad(after)) if m is not None]
        if spread and change <= factor * max(spread):
            return False
    return True


def _split(name: str) -> tuple[str, str]:
    case, metric = name.rsplit("/", 1)
    return case, metric


def _peano(rows, fallback: str = "") -> str:
    for row in rows:
        if match := _PEANO.search(row.get("extra", "")):
            return match[1]
    match = _PEANO.search(fallback)
    return match[1] if match else ""


def read_leg(npu: str, directory: Path, latest: Path, run_id: str = "") -> Leg:
    meta_path = directory / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    perf_path = directory / "perf.json"
    rows = json.loads(perf_path.read_text()) if perf_path.exists() else []
    result = Leg(npu, measured=bool(rows))
    result.failures, swept = failures(directory)
    failing = {f.case for f in result.failures}
    result.peano = _peano(rows, meta.get("provenance", ""))
    measured = {_split(row["name"])[0] for row in rows}
    result.cases = len(measured | swept)

    record = baseline(latest, run_id)
    if not record or not rows:
        return result
    result.baseline_commit = record.get("commit") or None
    result.baseline_peano = record.get("provenance", {}).get("peano", "")
    before = {
        f"{case}/{metric}": cell
        for case, cells in record["rows"].items()
        for metric, cell in cells.items()
    }
    for row in rows:
        case, metric = _split(row["name"])
        old = before.get(row["name"])
        if metric in DERIVED or not old or not old["value"]:
            continue
        change = Change(case, metric, row["unit"], old["value"], row["value"])
        if metric in GATED:
            if change.ratio >= GATED[metric]:
                result.regressed.append(change)
            elif change.ratio <= -GATED[metric]:
                result.improved.append(change)
        elif moved(metric, OTHER.get(metric, DEFAULT_THRESHOLD), old, row):
            result.other.append(change)
    for changes in (result.regressed, result.other):
        changes.sort(key=lambda c: -c.ratio)
    result.improved.sort(key=lambda c: c.ratio)
    nightly = {_split(name)[0] for name in before}
    result.unmeasured = sorted(nightly - measured - failing)
    result.new = sorted(measured - nightly)
    return result


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def _value(v: float, unit: str) -> str:
    if unit == "bytes":
        return f"{v / 1024:,.1f} KiB"
    if float(v).is_integer():
        return f"{int(v):,} {unit}"
    return f"{v:,.2f} {unit}"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(_cell(c) for c in row) + " |" for row in rows[:MAX_ROWS]]
    if len(rows) > MAX_ROWS:
        out.append(f"\n\u2026 and {len(rows) - MAX_ROWS} more, in the run's artifacts.")
    return out


def _changes(legs: list[Leg], attr: str) -> list[list[str]]:
    return [
        [
            leg.npu,
            f"`{c.case}`",
            c.metric,
            _value(c.before, c.unit),
            _value(c.after, c.unit),
            f"{100 * c.ratio:+.1f}%",
        ]
        for leg in legs
        for c in getattr(leg, attr)
    ]


def _details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body, "", "</details>"]


def render(legs: list[Leg], run_url: str = "") -> str:
    failing = sum(len(leg.failures) for leg in legs)
    regressed = sum(len(leg.regressed) for leg in legs)
    if not legs:
        title = "no NPU produced results"
    elif failing or regressed:
        title = f"{failing} failing, {regressed} regressed"
    else:
        title = "all passed, no regressions"
    gated = " or ".join(f"`{m}` {100 * t:g}%" for m, t in GATED.items())
    links = [f"[run]({run_url})"] if run_url else []
    links.append(f"[nightly history]({PAGE})")
    out = [
        MARKER,
        f"## Kernel checks: {title}",
        "",
        f"Compared with the last nightly on main. Regressed: {gated} or more "
        "worse. Both are stable run to run, so a move is the compiler's. "
        + " \u00b7 ".join(links),
        "",
    ]

    summary = []
    for leg in legs:
        if not leg.measured and not leg.failures:
            summary.append(
                [leg.npu, leg.peano or "?", "\u2014", "no results", "", "", ""]
            )
            continue
        peano = leg.peano or "?"
        if leg.baseline_peano and leg.baseline_peano != leg.peano:
            peano += f" (nightly: {leg.baseline_peano})"
        commit = leg.baseline_commit
        if commit and commit.get("id"):
            base = f"[{commit['id'][:7]}]({commit.get('url', '')})"
        else:
            base = "none cached" if leg.measured else "\u2014"
        summary.append(
            [
                leg.npu,
                peano,
                base,
                str(leg.cases),
                str(len(leg.failures)),
                str(len(leg.regressed)) if commit else "\u2014",
                str(len(leg.improved)) if commit else "\u2014",
            ]
        )
    out += _table(
        ["NPU", "Peano", "Nightly", "Cases", "Failing", "Regressed", "Improved"],
        summary,
    )

    rows = [
        [
            leg.npu,
            f"`{f.case}`",
            f"{len(f.failed)} of {f.total}" if f.total else ", ".join(f.failed),
            f.reason,
        ]
        for leg in legs
        for f in leg.failures
    ]
    if rows:
        out += ["", "### Failing", ""]
        out += _table(["NPU", "Case", "Inputs failed", "Reason"], rows)

    header = ["NPU", "Case", "Metric", "Nightly", "This run", "Change"]
    if rows := _changes(legs, "regressed"):
        out += ["", "### Regressed", ""]
        out += _table(header, rows)
    if rows := _changes(legs, "improved"):
        out += [""] + _details(f"Improved ({len(rows)})", _table(header, rows))
    if rows := _changes(legs, "other"):
        thresholds = ", ".join(
            f"`{m}` {100 * t:g}%"
            + (f" and {MAD[m]:g}\u00d7 its MAD" if m in MAD else "")
            for m, t in OTHER.items()
        )
        note = (
            f"Listed past {thresholds}. `npu_us` is timed on the host and "
            "moves with the machine; the byte counts do not."
        )
        out += [""] + _details(
            f"Other metrics that moved ({len(rows)})",
            [note, "", *_table(header, rows)],
        )
    unmeasured = [[leg.npu, f"`{case}`"] for leg in legs for case in leg.unmeasured]
    if unmeasured:
        out += [""] + _details(
            f"In the nightly but not measured here ({len(unmeasured)})",
            _table(["NPU", "Case"], unmeasured),
        )
    new = [[leg.npu, f"`{case}`"] for leg in legs for case in leg.new]
    if new:
        out += [""] + _details(
            f"Measured here, not in the nightly ({len(new)})",
            _table(["NPU", "Case"], new),
        )
    return "\n".join(out) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--results", required=True, type=Path)
    parser.add_argument("--baselines", required=True, type=Path)
    parser.add_argument("--run-url", default="")
    parser.add_argument(
        "--run-id", default="", help="ignore a baseline this run published itself"
    )
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    legs = (
        [
            read_leg(d.name, d, args.baselines / d.name / "latest.json", args.run_id)
            for d in sorted(args.results.iterdir())
            if d.is_dir()
        ]
        if args.results.is_dir()
        else []
    )
    report = render(legs, args.run_url)
    if args.out:
        args.out.write_text(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
