#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Report a kernel checks run's failures and regressions against the last nightly.

For each NPU that produced results: the cases that failed, and the cases
whose cycles or kernel object size moved against the nightly baseline on main.
nightlyKernelChecks.yml writes it to the job summary and, on a Peano PR,
keeps it as one comment. Standard library only, so it runs on any runner.

    pr_report.py --results results --baselines baselines --out report.md

``results/<npu>/`` holds a leg's artifact (meta.json, perf.json,
correctness.xml); ``baselines/<npu>/latest.json`` is the record publish.py
wrote to gh-pages for the last nightly that published rows. The thresholds are ``thresholds.json``'s, shared with the page.
The report names the part each side ran on; when they differ, only the
metrics thresholds.json marks ``across_parts`` are compared.
"""

import argparse
import json
import math
import re
import sys
import xml.etree.ElementTree as ET
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from publish import device_part, upgrade

MARKER = "<!-- kernel-checks-report -->"
PAGE = "https://xilinx.github.io/mlir-aie/dashboard/#view=night"
REPO = "https://github.com/Xilinx/mlir-aie"
THRESHOLDS = {
    metric: spec
    for metric, spec in json.loads(
        Path(__file__).with_name("thresholds.json").read_text()
    ).items()
    if not metric.startswith("_")
}
# Stable run to run, so a move is the compiler's: the regressions that count.
GATED = {m for m, s in THRESHOLDS.items() if s.get("gated")}
# Derived from another metric, so it would repeat that metric's rows; or
# shown but never judged.
UNJUDGED = {m for m, s in THRESHOLDS.items() if s.get("derived") or s.get("neutral")}
# A metric thresholds.json does not name.
DEFAULT_THRESHOLD = {"pct": 10}
# A change must also exceed this many MADs, per metric (the host-timed ones).
MAD = {m: s["mad"] for m, s in THRESHOLDS.items() if s.get("mad")}
# The same on every part of an NPU generation: compared even across parts.
ACROSS_PARTS = {m for m, s in THRESHOLDS.items() if s.get("across_parts")}
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
        """The change over the old value; from zero, infinite its way."""
        if not self.before:
            return math.copysign(math.inf, self.after - self.before)
        return (self.after - self.before) / self.before

    @property
    def text(self) -> str:
        """The change in percent, or from zero in the metric's unit."""
        if not self.before:
            return f"{self.after - self.before:+g} {self.unit}".rstrip()
        return f"{100 * self.ratio:+.1f}%"


@dataclass
class Leg:
    npu: str
    measured: bool
    peano: str = ""
    baseline_peano: str = ""
    # The part this run and the nightly ran on ("Strix", "Krackan").
    part: str = ""
    baseline_part: str = ""
    # Rows not compared because the parts differ.
    other_part: int = 0
    baseline_commit: dict | None = None
    # Why there is no comparison, when there is none.
    uncompared: str = ""
    # Why the timing run measured nothing, when it refused to.
    refused: str = ""
    cases: int = 0
    # How hard this run looked: the inputs the sweep checked, and how many
    # runs in a row each timed case made before its last check.
    inputs: int = 0
    runs: int = 0
    failures: list[Failure] = field(default_factory=list)
    regressed: list[Change] = field(default_factory=list)
    improved: list[Change] = field(default_factory=list)
    other: list[Change] = field(default_factory=list)
    unmeasured: list[str] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    flaky: list[tuple[str, int]] = field(default_factory=list)


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


def _last_attempts(xml) -> list[ET.Element]:
    """Every test's last attempt: JUnit repeats a retried test, once per attempt."""
    last: dict[tuple[str, str], ET.Element] = {}
    for test in ET.parse(xml).iter("testcase"):
        last[(test.get("classname", ""), test.get("name", ""))] = test
    return list(last.values())


def sweep(xml, *, include_skipped=False) -> Iterator[tuple[str, str, ET.Element]]:
    """Yield generic cases and explicitly attributed dedicated hardware checks."""
    for test in _last_attempts(xml):
        if test.find("skipped") is not None and not include_skipped:
            continue
        match = _EXTENSIVE.fullmatch(test.get("name", ""))
        if match:
            yield match[1], match[2], test
        else:
            factories = {
                factory
                for prop in test.findall("properties/property")
                if prop.get("name") == "kernel_check" and (factory := prop.get("value"))
            }
            for factory in sorted(factories):
                yield f"{factory}/{test.get('name')}", "dedicated", test


def failed(test) -> bool:
    return test.find("failure") is not None or test.find("error") is not None


def reruns(test) -> int:
    """How many times the runner retried ``test`` (its ``reruns`` property)."""
    for prop in test.findall("properties/property"):
        if prop.get("name") == "reruns":
            return int(prop.get("value") or 0)
    return 0


def flaky(directory: Path) -> list[tuple[str, int]]:
    """Return the tests that passed only on a retry, and how many retries each.

    From the sweep's JUnit report and the timing run's meta; a retried test
    that still failed is a failure, not flaky.
    """
    out = {}
    xml = directory / "correctness.xml"
    if xml.exists():
        for test in _last_attempts(xml):
            if (n := reruns(test)) and not failed(test):
                out[f"{test.get('classname', '')}::{test.get('name', '')}"] = n
    meta = directory / "meta.json"
    if meta.exists():
        record = json.loads(meta.read_text())
        bad = set(record.get("failed", []))
        for nodeid, n in record.get("reruns", {}).items():
            if nodeid not in bad:
                out[nodeid] = int(n)
    return sorted(out.items())


def _timing_reason(text: str, case: str) -> str:
    """Trim a timing-run failure's message to what the table does not say."""
    text = text.removeprefix("AssertionError: ").removeprefix(f"{case}: ").strip()
    return text or "failed in the timing run; see perf.log"


def failures(directory: Path) -> tuple[list[Failure], set[str], int]:
    """Return the failing cases, the swept cases and the inputs checked.

    The swept cases are every case the extensive sweep ran; the inputs are how
    many it checked.
    """
    by_case: dict[str, Failure] = {}
    swept = set()
    inputs = 0  # of the sweep; a dedicated test is not an input
    correctness_failures = set()
    xml = directory / "correctness.xml"
    if xml.exists():
        for case, variant, test in sweep(xml):
            swept.add(case)
            inputs += variant != "dedicated"
            f = by_case.setdefault(case, Failure(case))
            f.total += 1
            if failed(test):
                f.failed.append(variant)
                f.reason = f.reason or _reason(test)
                correctness_failures.add(
                    f"{test.get('classname', '')}::{test.get('name', '')}"
                )
    meta = directory / "meta.json"
    if meta.exists():
        # The timing run checks each case again; its failures reach only meta.
        # A run that refused to time fails every timing test for that one
        # reason, which the summary row gives.
        record = json.loads(meta.read_text())
        reasons = record.get("reasons", {})
        for nodeid in [] if record.get("refused") else record.get("failed", []):
            if "test_kernel_extensive[" in nodeid or nodeid in correctness_failures:
                continue
            match = _BRACKETED.search(nodeid)
            case = match[1] if match else nodeid
            f = by_case.setdefault(case, Failure(case))
            if not f.failed:
                f.reason = _timing_reason(reasons.get(nodeid, ""), case)
            f.failed.append("timing run")
    return (
        sorted((f for f in by_case.values() if f.failed), key=lambda f: f.case),
        swept,
        inputs,
    )


def baseline(path: Path, run_id: str = "") -> dict | None:
    """Return the last nightly's record (``latest.json``), or None.

    None without rows, in a newer format than this reads, or from this very
    run: a re-run of a nightly's report reads the record its own publish job
    already wrote, and a run compared with itself shows nothing. An older
    record is brought up to the current rows as publish.py's ``upgrade``
    brings it on its next publish, so a renamed metric is compared under its
    current name.
    """
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("schema", 1) > SCHEMA:
        return None
    upgrade(record)
    if run_id and str(record.get("id", "")) == str(run_id):
        return None
    return record if record.get("rows") else None


def _mad(cell: dict) -> float | None:
    match = _MAD.match(cell.get("range") or "")
    return float(match[1]) if match else None


def moved(metric: str, before: dict, after: dict) -> bool:
    """Whether a change from ``before`` to ``after`` counts for ``metric``.

    By its thresholds.json entry: at least ``pct`` percent of the old value
    (from a zero one, only ``abs`` decides, and without it nothing counts),
    at least ``abs`` in the metric's unit, and for a metric with a MAD
    factor, more than that many of the larger of the two rows' MADs.
    """
    spec = THRESHOLDS.get(metric, DEFAULT_THRESHOLD)
    change = abs(after["value"] - before["value"])
    if not change or change < spec.get("abs", 0):
        return False
    if not before["value"]:
        return "abs" in spec
    if change < spec.get("pct", 0) / 100 * abs(before["value"]):
        return False
    factor = MAD.get(metric)
    if factor:
        spread = [m for m in (_mad(before), _mad(after)) if m is not None]
        if spread and change <= factor * max(spread):
            return False
    return True


def rule(metric: str) -> str:
    """Say what change counts for ``metric``: "15% and 0.5 s", "1 ulp"."""
    spec = THRESHOLDS.get(metric, DEFAULT_THRESHOLD)
    parts = [f"{spec['pct']:g}%"] if "pct" in spec else []
    if "abs" in spec:
        parts.append(f"{spec['abs']:g} in its unit")
    if "mad" in spec:
        parts.append(f"{spec['mad']:g}\u00d7 its MAD")
    return " and ".join(parts)


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
    result.failures, swept, result.inputs = failures(directory)
    result.flaky = flaky(directory)
    result.runs = int(meta.get("runs_per_case") or 0)
    failing = {f.case for f in result.failures}
    result.peano = _peano(rows, meta.get("provenance", ""))
    result.part = device_part((meta.get("preflight") or {}).get("device")) or ""
    measured = {_split(row["name"])[0] for row in rows}
    result.cases = len(measured | swept)

    record = baseline(latest, run_id)
    pmode = (meta.get("preflight") or {}).get("pmode")
    result.refused = str(meta.get("refused") or "")
    if result.refused:
        result.uncompared = f"not timed: {result.refused}"
    elif not rows:
        result.uncompared = "nothing timed"
    elif not record:
        result.uncompared = "no nightly to compare with"
    elif not pmode:
        result.uncompared = "power mode unknown, not compared"
    elif record.get("pmode") != pmode:
        result.uncompared = (
            f"nightly in {record.get('pmode') or 'an unknown'} mode, "
            f"this run in {pmode}: not compared"
        )
    if result.uncompared or not record:
        return result
    result.baseline_commit = record.get("commit") or None
    result.baseline_peano = record.get("provenance", {}).get("peano", "")
    result.baseline_part = record.get("device") or ""
    same_part = bool(result.part) and result.part == result.baseline_part
    before = {
        f"{case}/{metric}": cell
        for case, cells in record["rows"].items()
        for metric, cell in cells.items()
    }
    for row in rows:
        case, metric = _split(row["name"])
        old = before.get(row["name"])
        if metric in UNJUDGED or not old:
            continue
        if not same_part and metric not in ACROSS_PARTS:
            result.other_part += 1
            continue
        if not moved(metric, old, row):
            continue
        change = Change(case, metric, row["unit"], old["value"], row["value"])
        worse = (change.after > change.before) != bool(
            THRESHOLDS.get(metric, {}).get("higher_is_better")
        )
        if metric not in GATED:
            result.other.append(change)
        elif worse:
            result.regressed.append(change)
        else:
            result.improved.append(change)
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
            c.text,
        ]
        for leg in legs
        for c in getattr(leg, attr)
    ]


def _details(summary: str, body: list[str]) -> list[str]:
    return ["<details>", f"<summary>{summary}</summary>", "", *body, "", "</details>"]


def _coverage(legs: list[Leg]) -> str:
    """Say how hard the run looked, and how to look harder."""
    inputs = max((leg.inputs for leg in legs), default=0)
    runs = max((leg.runs for leg in legs), default=0)
    if not inputs and not runs:
        return ""
    parts = []
    if inputs:
        parts.append(f"{inputs} inputs per NPU (random and edge)")
    if runs:
        parts.append(f"each timed case again after {runs} runs in a row")
    return (
        "Checked " + " and ".join(parts) + ". A bug that shows only on rarer "
        "data or after longer runs can still pass: to soak a compiler, "
        "dispatch this workflow with `peano` and larger `seeds` or `iters`."
    )


def render(legs: list[Leg], run_url: str = "", commit: str = "") -> str:
    failing = sum(len(leg.failures) for leg in legs)
    regressed = sum(len(leg.regressed) for leg in legs)
    compared = any(not leg.uncompared for leg in legs)
    if not legs:
        title = "no NPU produced results"
    elif not compared:
        title = f"{failing} failing" if failing else "all passed"
        title += ", not compared with a nightly"
    elif failing or regressed:
        title = f"{failing} failing, {regressed} regressed"
    else:
        title = "all passed, no regressions"
    if refused := [leg.npu for leg in legs if leg.refused]:
        title += f", {' and '.join(refused)} not timed"
    if retried := sum(len(leg.flaky) for leg in legs):
        title += f", {retried} passed on a retry"
    gated = " or ".join(f"`{m}` {rule(m)}" for m in sorted(GATED))
    links = [f"[run]({run_url})"] if run_url else []
    links.append(f"[nightly history]({PAGE})")
    lead = (
        f"Compared with the last nightly on main. Regressed: {gated} or more "
        "worse. Each is stable run to run, so a move is the compiler's. "
        if compared
        else "Not compared with a nightly on main; the table says why. "
    )
    out = [
        MARKER,
        f"## Kernel checks: {title}",
        "",
        lead + " \u00b7 ".join(links),
        "",
    ]
    nightlies = {
        leg.baseline_commit["id"]
        for leg in legs
        if leg.baseline_commit and leg.baseline_commit.get("id")
    }
    if commit and (others := sorted(nightlies - {commit})):
        ranges = ", ".join(
            f"[{n[:7]}...{commit[:7]}]({REPO}/compare/{n}...{commit})" for n in others
        )
        out += [
            f"This run is on a different commit than the nightly ({ranges}): "
            "a change there can move these numbers too, not only the compiler.",
            "",
        ]
    if coverage := _coverage(legs):
        out += [coverage, ""]

    summary = []
    for leg in legs:
        if not leg.measured and not leg.failures and not leg.cases:
            summary.append(
                [
                    leg.npu,
                    leg.part or "?",
                    leg.peano or "?",
                    "\u2014",
                    "no results",
                    "",
                    "",
                    "",
                ]
            )
            continue
        peano = leg.peano or "?"
        if leg.baseline_peano and leg.baseline_peano != leg.peano:
            peano += f" (nightly: {leg.baseline_peano})"
        part = leg.part or "?"
        if leg.baseline_part and leg.baseline_part != leg.part:
            part += f" (nightly: {leg.baseline_part})"
        nightly = leg.baseline_commit
        if nightly and nightly.get("id"):
            base = f"[{nightly['id'][:7]}]({nightly.get('url', '')})"
        else:
            base = leg.uncompared or "\u2014"
        summary.append(
            [
                leg.npu,
                part,
                peano,
                base,
                str(leg.cases),
                str(len(leg.failures)),
                str(len(leg.regressed)) if nightly else "\u2014",
                str(len(leg.improved)) if nightly else "\u2014",
            ]
        )
    out += _table(
        [
            "NPU",
            "Part",
            "Peano",
            "Nightly",
            "Cases",
            "Failing",
            "Regressed",
            "Improved",
        ],
        summary,
    )
    for leg in legs:
        if leg.other_part:
            out += [
                "",
                f"{leg.npu}: this run is on {leg.part or 'an unknown part'}, the "
                f"nightly on {leg.baseline_part or 'an unknown part'}, so "
                f"{leg.other_part} host-timed rows (`dispatch_us`, `npu_us`) are not compared.",
            ]

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

    if rows := [[leg.npu, f"`{t}`", str(n)] for leg in legs for t, n in leg.flaky]:
        out += [""] + _details(
            f"Passed only on a retry ({len(rows)})",
            [
                "The runner retries a failed test once; these failed first. "
                "A kernel that fails now and then is worth a soak.",
                "",
                *_table(["NPU", "Test", "Retries"], rows),
            ],
        )

    header = ["NPU", "Case", "Metric", "Nightly", "This run", "Change"]
    if rows := _changes(legs, "regressed"):
        out += ["", "### Regressed", ""]
        out += _table(header, rows)
    if rows := _changes(legs, "improved"):
        out += [""] + _details(f"Improved ({len(rows)})", _table(header, rows))
    if rows := _changes(legs, "other"):
        # Only the metrics listed: thresholds.json names the component
        # checks' too.
        listed = {row[2] for row in rows}
        thresholds = ", ".join(f"`{m}` {rule(m)}" for m in sorted(listed))
        note = (
            f"Listed past {thresholds}. `dispatch_us` and `npu_us` are timed "
            "on the host and move with the machine; the byte counts do not."
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
        "--commit", default="", help="the mlir-aie commit this run is on"
    )
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
    report = render(legs, args.run_url, args.commit)
    if args.out:
        args.out.write_text(report)
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
