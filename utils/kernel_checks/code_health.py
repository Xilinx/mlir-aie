#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
r"""Record code health for the maintainer dashboard's Code health tab.

nightlyCodeHealth.yml runs four checks of main and this script turns each
one's output into a small summary, then folds the night's summaries into
``status.json`` and ``history.json``, which the workflow installs under
``gh-pages:code-health/``:

* ``sanitizer``: lit's JSON report (``lit -o``) of check-aie built under
  AddressSanitizer with UndefinedBehaviorSanitizer, or under
  ThreadSanitizer, and the sanitizers' own report files (``--logs``). The
  nightly gives each test its own ``log_path`` (``SanitizerLogShTest`` in
  python/aie_lit_utils), so a report a test pipes into FileCheck, or one
  from a tool it expects to fail (``not aie-opt``), is still read and still
  names its test; every test's output is read too, a passing one's as well.
  Each report becomes a finding, keyed by its sanitizer, its kind and the
  first frame in this repository's code, so the same bug reached from forty
  tests is one finding with forty tests. A test that failed with no report
  (usually a time-out: instrumented tools are slower) is listed apart.
* ``tidy``: clang-tidy's output over every C++ file in lib/ and tools/.
  lintAndFormat.yml enforces clang-tidy on the files the ``clang-tidy`` hook
  in .pre-commit-config.yaml lists (enabled): a finding in one of them got
  past the gate. The rest are counted by check, and a file outside the list
  with no finding is ready to join it.
* ``publish``: the night's pieces, the lit suite's coverage (``coverage.lcov``
  from codeCoverage.yml) per file and per area, and, with
  ``--code-scanning``, the repository's open code-scanning alerts (CodeQL's
  Python and C++ analyses), asked of GitHub's API with the workflow's token.
  The previously published status dates each finding (``since``) and says
  which alerts are new; the history gains a point.

    code_health.py sanitizer --name asan-ubsan --logs DIR --out asan.json lit/*.json
    code_health.py tidy --files files.txt --exit-code N --out tidy.json tidy.log
    code_health.py publish --coverage coverage.lcov --piece asan.json ... \
        --code-scanning --previous status.json --history history.json --out DIR
"""

import argparse
import datetime
import hashlib
import json
import os
import posixpath
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ci_health import ApiError, github  # noqa: E402

SCHEMA = 1
# gh-pages stays bounded: status.json is replaced each night and history.json
# keeps a point a night for a year, so neither grows with time; the caps
# below bound status.json however bad a night is (test_code_health.py checks
# STATUS_BUDGET against a flood of findings).
HISTORY_POINTS = 366
KEEP_TESTS = 25
KEEP_FAILED = 100
KEEP_ALERTS = 200
KEEP_FINDINGS = 100
# Every finding's identity and first night, displayed or not, so a finding
# past KEEP_FINDINGS or one a missing night hid keeps its date.
KEEP_KNOWN = 2000
# The report itself for the first few findings of a check; the rest keep
# their kind, place and tests.
EXCERPTS_KEPT = 10
EXCERPT_LINES = 30
# A C++ template frame can run to kilobytes.
LINE_CHARS = 240
STATUS_BUDGET = 1_000_000
HISTORY_BUDGET = 400_000

# This repository's code, as a frame's or a finding's path starts.
OWN = ("lib/", "include/", "tools/", "python/", "runtime_lib/", "aie_runtime_lib/")


def relative(path: str, root: str) -> str:
    """Return a path inside the checkout as the repository names it."""
    # Reports come from Linux runners; keep their separators on any host.
    path = posixpath.normpath(path) if path else path
    if root and path.startswith(root.rstrip("/") + "/"):
        return path[len(root.rstrip("/")) + 1 :]
    # Outside the checkout (a system or prebuilt-MLIR header) stays
    # absolute, so it is never taken for this repository's code.
    return path


def clip(text: str, n: int = LINE_CHARS) -> str:
    return text if len(text) <= n else text[: n - 1] + "…"


# ----------------------------------------------------------------------
# Sanitizers
# ----------------------------------------------------------------------

# The first line of each sanitizer's report, and the kind it names.
ASAN = re.compile(r"ERROR: AddressSanitizer: ([\w-]+)")
LSAN = re.compile(r"ERROR: LeakSanitizer: (detected memory leaks)")
TSAN = re.compile(r"WARNING: ThreadSanitizer: ([^(\n]+?)\s*(?:\(|$)", re.M)
UBSAN = re.compile(r"^(.+?):(\d+):\d+: runtime error: (.+)$", re.M)
# "    #3 0x55d1 in mlir::Foo::bar() /src/lib/X.cpp:12:5" (ASan) or
# "    #3 mlir::Foo::bar() /src/lib/X.cpp:12:5 (aie-opt+0x1234)" (TSan).
FRAME = re.compile(
    r"^\s*#(\d+) (?:0x[0-9a-f]+ in )?(.+?) ((?:/|[A-Za-z]:)[^\s:]+):(\d+)(?::\d+)?",
    re.M,
)
REPORT_END = re.compile(r"^(SUMMARY: \w+Sanitizer.*|==\d+==ABORTING)$", re.M)
# ASan's first line can say only "attempting"; its summary names the bug type.
ASAN_SUMMARY = re.compile(r"^SUMMARY: AddressSanitizer: ([\w-]+)", re.M)


def ubsan_kind(message: str) -> str:
    """UBSan's message without its values: "signed integer overflow"."""
    kind = message.split(":", 1)[0]
    kind = re.sub(r"\b(?:-?\d+|0x[0-9a-f]+)\b", "N", kind)
    kind = re.sub(r"'[^']*'", "T", kind)
    return kind.strip()


def reports(output: str, root: str) -> list[dict]:
    """Extract sanitizer reports from one test's output as findings (no tests)."""
    found = []
    starts = []
    for rx, sanitizer in ((ASAN, "asan"), (LSAN, "lsan"), (TSAN, "tsan")):
        for m in rx.finditer(output):
            starts.append((m.start(), sanitizer, m.group(1).strip()))
    for m in UBSAN.finditer(output):
        starts.append((m.start(), "ubsan", ubsan_kind(m.group(3))))
    starts.sort()
    for i, (at, sanitizer, kind) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(output)
        tail = REPORT_END.search(output, at, end)
        text = output[at : tail.end() if tail else end]
        if sanitizer == "asan" and (summary := ASAN_SUMMARY.search(text)):
            kind = summary.group(1)
        where = {}
        ub = UBSAN.match(output, at) if sanitizer == "ubsan" else None
        if sanitizer == "ubsan":
            assert ub is not None
            where = {"file": relative(ub.group(1), root), "line": int(ub.group(2))}
            # UBSan's own line is the place; a stack, if printed, says who.
            frames = [f for f in FRAME.finditer(text)]
            func = next(
                (
                    f.group(2)
                    for f in frames
                    if relative(f.group(3), root) == where["file"]
                ),
                "",
            )
            if func:
                where["function"] = func
        else:
            for f in FRAME.finditer(text):
                path = relative(f.group(3), root)
                if path.startswith(OWN) and "/third_party/" not in path:
                    where = {
                        "file": path,
                        "line": int(f.group(4)),
                        "function": f.group(2),
                    }
                    break
            if not where:
                first = FRAME.search(text)
                if first:
                    where = {
                        "file": relative(first.group(3), root),
                        "line": int(first.group(4)),
                        "function": first.group(2),
                        "outside": True,
                    }
        lines = text.strip("\n").splitlines()
        # The report's own words for its first occurrence, values and all:
        # "shift exponent 32 is too large for 32-bit type 'uint32_t'".
        message = kind
        if ub is not None:
            message = re.sub(r" \(aka '[^']*'\)", "", ub.group(3)).strip()
        found.append(
            {
                "sanitizer": sanitizer,
                "kind": kind,
                "message": message,
                "where": where,
                "excerpt": "\n".join(
                    clip(line.replace(root.rstrip("/") + "/", "") if root else line)
                    for line in lines[:EXCERPT_LINES]
                ),
            }
        )
    return found


ANONYMOUS = "(anonymous namespace)"


def finding_key(f: dict) -> str:
    """Identify the same bug across nights, ignoring lines that edits move."""
    w = f.get("where") or {}
    # Drop the parameter list, not an unnamed namespace's qualifier.
    func = w.get("function", "").replace(ANONYMOUS, "\0")
    func = re.sub(r"\(.*$", "", func).replace("\0", ANONYMOUS)
    # A template's name can run to kilobytes: such a key carries its digest.
    if len(func) > 160:
        func = func[:80] + "#" + hashlib.sha1(func.encode()).hexdigest()[:12]
    return "|".join([f["sanitizer"], clip(f["kind"], 160), w.get("file", "?"), func])


FAILED = {"FAIL", "TIMEOUT", "UNRESOLVED", "XPASS"}
RAN = {"PASS", "FAIL", "TIMEOUT", "XFAIL", "XPASS", "UNRESOLVED"}


def known_id(key: str) -> str:
    """Digest a finding key for the identities kept across nights."""
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def summarize_sanitizer(
    name: str, lits: list[dict], root: str, logs: dict[str, list[str]] | None = None
) -> dict:
    """Summarize one sanitizer build's reports as one status piece.

    Args:
        name: The check, ``asan-ubsan`` or ``tsan``.
        lits: lit's JSON reports; each test's output is read.
        root: The checkout the reports' paths start with.
        logs: The ``log_path`` files' texts by prefix, the SHA-1 of the lit
            test name that wrote them (``SanitizerLogShTest``).

    Returns:
        The piece, every finding listed with the tests that reach it.
    """
    unclaimed = dict(logs or {})
    findings: dict[str, dict] = {}
    counted: dict[str, set] = {}
    failed, tests, ran = [], 0, 0
    for lit in lits:
        for t in lit.get("tests", []):
            tests += 1
            code = t.get("code", "")
            ran += code in RAN
            texts = unclaimed.pop(hashlib.sha1(t["name"].encode()).hexdigest()[:16], [])
            found = [
                f for text in (t.get("output", ""), *texts) for f in reports(text, root)
            ]
            for f in found:
                key = finding_key(f)
                seen = findings.setdefault(
                    key, {**f, "key": key, "tests": [], "test_count": 0}
                )
                names = counted.setdefault(key, set())
                if t["name"] not in names:
                    names.add(t["name"])
                    seen["test_count"] += 1
                    if len(seen["tests"]) < KEEP_TESTS:
                        seen["tests"].append(t["name"])
            if code in FAILED and not found:
                failed.append({"test": t["name"], "code": code})
    # A log no test in the reports wrote: its finding names no test.
    for text in (text for texts in unclaimed.values() for text in texts):
        for f in reports(text, root):
            key = finding_key(f)
            findings.setdefault(key, {**f, "key": key, "tests": [], "test_count": 0})
    order = sorted(findings.values(), key=lambda f: (-f["test_count"], f["key"]))
    by_sanitizer: dict[str, int] = {}
    for f in order:
        by_sanitizer[f["sanitizer"]] = by_sanitizer.get(f["sanitizer"], 0) + 1
    for f in order:
        f["message"] = clip(f["message"])
        if "function" in f["where"]:
            f["where"]["function"] = clip(f["where"]["function"])
    # Every finding, displayed or not: which are new is only known at publish.
    return {
        "check": name,
        "tests": tests,
        "ran": ran,
        "findings": order[:KEEP_KNOWN],
        "finding_count": len(order),
        "by_sanitizer": by_sanitizer,
        "failed_without_report": failed[:KEEP_FAILED],
        "failed_without_report_count": len(failed),
    }


# ----------------------------------------------------------------------
# clang-tidy
# ----------------------------------------------------------------------

DIAG = re.compile(r"^(.+?):(\d+):(\d+): (warning|error): (.*?) \[([\w.,-]+)\]$", re.M)
# clang's ClangTool::run names the translation unit it could not compile;
# the compiler error itself may point into a header that unit includes.
UNPARSED = re.compile(r"^Error while processing (.+)\.$", re.M)


def enabled_pattern(config: Path) -> re.Pattern:
    """Read the clang-tidy hook's ``files:`` regex from .pre-commit-config.yaml."""
    text = config.read_text(encoding="utf-8")
    m = re.search(r"- id: clang-tidy\n(?:.*\n)*?(\s+)files: \|\n((?:\1\s+.*\n)+)", text)
    if not m:
        raise ValueError(f"no clang-tidy hook with a files: block in {config}")
    return re.compile(m.group(2).strip())


# utils/run_clang_tidy.sh hands its files to xargs: 0 is a clean run and 123
# one in which some clang-tidy reported a finding. Any other status is a run
# that did not finish: the script's own checks failed, or a clang-tidy
# crashed (xargs stops there), could not start, or exited 255.
TIDY_COMPLETED = (0, 123)
TIDY_CRASH = re.compile(r"^(?:Stack dump:|PLEASE submit a bug report)", re.M)


def tidy_incomplete(log: str, exit_code: int) -> str | None:
    """Say why clang-tidy's run did not cover every file, or None if it did."""
    if exit_code not in TIDY_COMPLETED:
        return f"utils/run_clang_tidy.sh exited {exit_code}"
    if TIDY_CRASH.search(log):
        return "clang-tidy crashed"
    return None


def summarize_tidy(log: str, files: list[str], enabled: re.Pattern, root: str) -> dict:
    """clang-tidy's output over ``files`` as one piece of the status."""
    seen = set()
    by_file: dict[str, int] = {}
    by_check: dict[str, int] = {}
    in_enabled = []
    broken = {relative(m.group(1), root) for m in UNPARSED.finditer(log)}
    for m in DIAG.finditer(log):
        path = relative(m.group(1), root)
        checks = [c for c in m.group(6).split(",") if c != "-warnings-as-errors"]
        check = checks[0] if checks else "unknown"
        if (path, m.group(2), check) in seen:
            continue
        seen.add((path, m.group(2), check))
        if check == "clang-diagnostic-error":
            continue
        if not path.startswith(OWN) or "/third_party/" in path:
            continue
        by_file[path] = by_file.get(path, 0) + 1
        if enabled.match(path):
            in_enabled.append(
                {
                    "file": path,
                    "line": int(m.group(2)),
                    "check": check,
                    "message": clip(m.group(5)),
                }
            )
        else:
            by_check[check] = by_check.get(check, 0) + 1
    on = [f for f in files if enabled.match(f)]
    ready = [
        f
        for f in files
        if not enabled.match(f) and f not in by_file and f not in broken
    ]
    outside = {f: n for f, n in by_file.items() if not enabled.match(f)}
    return {
        "check": "clang-tidy",
        "files": len(files),
        "enabled_files": len(on),
        "enabled_findings": in_enabled[:KEEP_FINDINGS],
        "enabled_finding_count": len(in_enabled),
        "outside_count": sum(outside.values()),
        "outside_by_check": dict(
            sorted(by_check.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "outside_by_file": dict(
            sorted(outside.items(), key=lambda kv: (-kv[1], kv[0]))[:60]
        ),
        "ready": sorted(ready),
        "could_not_parse": sorted(broken),
    }


# ----------------------------------------------------------------------
# Coverage
# ----------------------------------------------------------------------


def area_of(path: str) -> str:
    """Find a file's area: lib/Dialect/AIE, lib/Targets, or tools/aiecc."""
    parts = path.split("/")
    if parts[0] == "lib" and len(parts) > 3 and parts[1] in ("Dialect", "Conversion"):
        return "/".join(parts[:3])
    if parts[0] in ("lib", "tools") and len(parts) > 2:
        return "/".join(parts[:2])
    if parts[0] == "include" and len(parts) > 3:
        return "/".join(parts[:4])
    return parts[0]


def read_lcov(text: str, root: str) -> dict:
    """Read lcov as {file: [lines hit, lines, functions hit, functions]}."""
    files: dict[str, list[int]] = {}
    cur = None
    for line in text.splitlines():
        if line.startswith("SF:"):
            cur = relative(line[3:], root)
            files.setdefault(cur, [0, 0, 0, 0])
        elif cur is None:
            continue
        elif line.startswith("LH:"):
            files[cur][0] += int(line[3:])
        elif line.startswith("LF:"):
            files[cur][1] += int(line[3:])
        elif line.startswith("FNH:"):
            files[cur][2] += int(line[4:])
        elif line.startswith("FNF:"):
            files[cur][3] += int(line[4:])
        elif line == "end_of_record":
            cur = None
    return {f: v for f, v in files.items() if v[1]}


def coverage_status(files: dict, previous: dict | None) -> dict:
    total = [sum(v[i] for v in files.values()) for i in range(4)]
    areas: dict[str, list[int]] = {}
    for f, v in files.items():
        a = areas.setdefault(area_of(f), [0, 0, 0, 0])
        for i in range(4):
            a[i] += v[i]
    # A file that lost covered lines since the last night: named, since the
    # page's line for a drop says where.
    lost = []
    before = (previous or {}).get("files") or {}
    for f, v in files.items():
        if f in before and before[f][1]:
            was = before[f][0] / before[f][1]
            now = v[0] / v[1]
            if now < was - 1e-9 and before[f][0] - v[0] > 0:
                lost.append(
                    {
                        "file": f,
                        "before": round(100 * was, 2),
                        "after": round(100 * now, 2),
                        "lines_lost": before[f][0] - v[0],
                    }
                )
    lost.sort(key=lambda x: -x["lines_lost"])
    return {
        "lines": {"hit": total[0], "total": total[1]},
        "functions": {"hit": total[2], "total": total[3]},
        "areas": [
            {"area": a, "lines": [v[0], v[1]], "functions": [v[2], v[3]]}
            for a, v in sorted(areas.items(), key=lambda kv: -(kv[1][1] - kv[1][0]))
        ],
        "files": files,
        "lost": lost[:30],
    }


# ----------------------------------------------------------------------
# Code scanning
# ----------------------------------------------------------------------


def language_of(alert: dict) -> str:
    cat = (alert.get("most_recent_instance") or {}).get("category") or ""
    m = re.search(r"language:([\w-]+)", cat)
    if m:
        return {"c-cpp": "cpp", "cpp": "cpp"}.get(m.group(1), m.group(1))
    path = ((alert.get("most_recent_instance") or {}).get("location") or {}).get(
        "path"
    ) or ""
    return (
        "python"
        if path.endswith(".py")
        else "cpp" if re.search(r"\.(c|cc|cpp|h|hpp)$", path) else "other"
    )


def scanning_status(get, repo: str, previous: dict | None) -> dict:
    alerts, page = [], 1
    while True:
        batch = get(
            f"/repos/{repo}/code-scanning/alerts?state=open&per_page=100&page={page}"
        )
        alerts += batch
        if len(batch) < 100:
            break
        page += 1
    ids = alert_ids(previous)
    known = set(ids or [])
    had = ids is not None
    out = []
    for a in alerts:
        rule = a.get("rule") or {}
        loc = (a.get("most_recent_instance") or {}).get("location") or {}
        out.append(
            {
                "number": a["number"],
                "rule": rule.get("id"),
                "description": clip(rule.get("description") or ""),
                "severity": rule.get("security_severity_level")
                or rule.get("severity")
                or "note",
                "tool": (a.get("tool") or {}).get("name"),
                "language": language_of(a),
                "file": loc.get("path"),
                "line": loc.get("start_line"),
                "url": a.get("html_url"),
                "created_at": a.get("created_at"),
                "new": had and a["number"] not in known,
            }
        )
    rank = {
        "critical": 0,
        "high": 1,
        "error": 1,
        "medium": 2,
        "warning": 2,
        "low": 3,
        "note": 4,
    }
    out.sort(key=lambda a: (not a["new"], rank.get(a["severity"], 5), -a["number"]))
    by: dict[str, int] = {}
    new_by: dict[str, int] = {}
    for a in out:
        by[a["severity"]] = by.get(a["severity"], 0) + 1
        if a["new"]:
            new_by[a["severity"]] = new_by.get(a["severity"], 0) + 1
    langs: dict[str, int] = {}
    for a in out:
        langs[a["language"]] = langs.get(a["language"], 0) + 1
    fixed = len(known - {a["number"] for a in out}) if had else 0
    return {
        "open": len(out),
        "by_severity": by,
        "by_language": langs,
        "alert_ids": sorted(a["number"] for a in out),
        "alerts": out[:KEEP_ALERTS],
        "new": sum(a["new"] for a in out),
        "new_by_severity": new_by,
        "fixed": fixed,
    }


# ----------------------------------------------------------------------
# The night
# ----------------------------------------------------------------------

CHECKS = ("asan-ubsan", "tsan", "clang-tidy")


def known_findings(before: dict | None) -> dict:
    """Return the last published finding identities and their first nights."""
    before = before or {}
    if isinstance(before.get("known"), dict):
        return dict(before["known"])
    return {known_id(f["key"]): f.get("since") for f in before.get("findings") or []}


def date_findings(piece: dict, before: dict | None, today: str) -> dict:
    """Each finding's first night, carried from the last status."""
    since = known_findings(before)
    findings = piece.get("findings", [])
    for f in findings:
        i = known_id(f["key"])
        f["since"] = since.get(i) or today
        f["new"] = i not in since
    findings.sort(key=lambda f: not f["new"])
    for f in findings[EXCERPTS_KEPT:]:
        f["excerpt"] = ""
    piece["new_count"] = sum(f["new"] for f in findings)
    piece["known"] = {known_id(f["key"]): f["since"] for f in findings}
    piece["findings"] = findings[:KEEP_FINDINGS]
    return piece


def alert_ids(before: dict | None) -> list | None:
    """Return the last known open alerts, or None if none were ever published."""
    before = before or {}
    if "alert_ids" in before:
        return before["alert_ids"]
    if "alerts" in before:
        return [a["number"] for a in before["alerts"] or []]
    return None


def run_info(env=os.environ) -> dict:
    server = env.get("GITHUB_SERVER_URL", "https://github.com")
    repo = env.get("GITHUB_REPOSITORY", "Xilinx/mlir-aie")
    rid = env.get("GITHUB_RUN_ID")
    return {
        "id": int(rid) if rid and rid.isdigit() else None,
        "url": f"{server}/{repo}/actions/runs/{rid}" if rid else None,
        "sha": env.get("GITHUB_SHA"),
        "event": env.get("GITHUB_EVENT_NAME"),
    }


def build_status(
    pieces: dict,
    coverage: dict | None,
    scanning,
    previous: dict | None,
    now: datetime.datetime,
    run: dict,
) -> dict:
    prev = previous if previous and previous.get("schema", 1) <= SCHEMA else None
    today = now.date().isoformat()
    status = {
        "schema": SCHEMA,
        "generated_at": now.isoformat().replace("+00:00", "Z"),
        "run": run,
    }
    for name in CHECKS:
        piece = pieces.get(name)
        if piece is None:
            status[name] = {"check": name, "missing": True}
            # A failed night shows no findings but forgets none: when the
            # check comes back, what it finds is dated from before.
            known = known_findings((prev or {}).get(name))
            if name != "clang-tidy" and known:
                status[name]["known"] = known
        elif name == "clang-tidy":
            status[name] = piece
        else:
            status[name] = date_findings(piece, (prev or {}).get(name), today)
    status["coverage"] = coverage if coverage is not None else {"missing": True}
    if scanning is not None and not isinstance(scanning, str):
        status["code_scanning"] = scanning
    else:
        # Without an answer the alerts are not shown, but their identities
        # are kept, so the next answer says which alerts are new and fixed.
        cs: dict = (
            {"error": scanning} if isinstance(scanning, str) else {"missing": True}
        )
        ids = alert_ids((prev or {}).get("code_scanning"))
        if ids is not None:
            cs["alert_ids"] = ids
        status["code_scanning"] = cs
    return status


def pct(hit_total) -> float | None:
    hit, total = hit_total
    return round(100 * hit / total, 2) if total else None


def history_point(status: dict) -> dict:
    cov, cs, tidy = status["coverage"], status["code_scanning"], status["clang-tidy"]
    point = {
        "date": status["generated_at"],
        "run": status["run"].get("id"),
        "sha": status["run"].get("sha"),
    }
    if not cov.get("missing"):
        point["lines"] = pct((cov["lines"]["hit"], cov["lines"]["total"]))
        point["functions"] = pct((cov["functions"]["hit"], cov["functions"]["total"]))
    for name in ("asan-ubsan", "tsan"):
        s = status[name]
        if not s.get("missing"):
            by = s.get("by_sanitizer")
            if by is None:
                by = {}
                for f in s["findings"]:
                    by[f["sanitizer"]] = by.get(f["sanitizer"], 0) + 1
            point[name] = {
                "findings": s["finding_count"],
                "by": by,
                "failed": s["failed_without_report_count"],
            }
    if not cs.get("missing") and not cs.get("error"):
        point["code_scanning"] = {
            "open": cs["open"],
            "by_severity": cs["by_severity"],
            "by_language": cs["by_language"],
        }
    if not tidy.get("missing"):
        point["clang-tidy"] = {
            "enabled": tidy["enabled_finding_count"],
            "outside": tidy["outside_count"],
            "enabled_files": tidy["enabled_files"],
            "files": tidy["files"],
        }
    return point


def update_history(old: dict | None, point: dict) -> dict:
    points = [
        p
        for p in ((old or {}).get("points") or [])
        if p.get("run") != point.get("run") or point.get("run") is None
    ]
    points.append(point)
    return {"schema": SCHEMA, "points": points[-HISTORY_POINTS:]}


def read_json(path: Path | None):
    if not path or not path.is_file() or not path.stat().st_size:
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return None


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")


def main(argv=None, now: datetime.datetime | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    parser.add_argument(
        "--source-root", default=os.environ.get("GITHUB_WORKSPACE", str(ROOT))
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser(
        "sanitizer", help="summarize lit's JSON reports of a sanitizer build"
    )
    s.add_argument("--name", required=True, choices=["asan-ubsan", "tsan"])
    s.add_argument(
        "--logs",
        type=Path,
        help="the directory of the sanitizers' log_path files (AIE_SANITIZER_LOG_DIR)",
    )
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("lit", nargs="*", type=Path)
    t = sub.add_parser("tidy", help="summarize clang-tidy's output")
    t.add_argument("--files", type=Path, required=True)
    t.add_argument("--config", type=Path, default=ROOT / ".pre-commit-config.yaml")
    t.add_argument("--out", type=Path, required=True)
    t.add_argument(
        "--exit-code",
        type=int,
        default=0,
        help="utils/run_clang_tidy.sh's exit status; an unfinished run writes nothing",
    )
    t.add_argument("log", type=Path)
    p = sub.add_parser(
        "publish", help="fold the night into status.json and history.json"
    )
    p.add_argument("--coverage", type=Path)
    p.add_argument("--piece", type=Path, action="append", default=[])
    p.add_argument("--code-scanning", action="store_true")
    p.add_argument("--previous", type=Path)
    p.add_argument("--history", type=Path)
    p.add_argument("--out", type=Path, required=True)
    for command in (s, t, p):
        command.add_argument("--source-root", default=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.source_root

    if args.cmd == "sanitizer":
        lits = [j for j in (read_json(p) for p in args.lit) if j]
        if not lits:
            print(
                "no lit reports loaded; no sanitizer summary written", file=sys.stderr
            )
            return 1
        logs: dict[str, list[str]] = {}
        if args.logs and args.logs.is_dir():
            for path in sorted(p for p in args.logs.iterdir() if p.is_file()):
                logs.setdefault(path.name.split(".")[0], []).append(
                    path.read_text(encoding="utf-8", errors="replace")
                )
        piece = summarize_sanitizer(args.name, lits, root, logs)
        write_json(args.out, piece)
        print(
            f"{args.name}: {piece['finding_count']} findings in {piece['ran']} tests run; "
            f"{piece['failed_without_report_count']} failed with no report",
            file=sys.stderr,
        )
        return 0

    if args.cmd == "tidy":
        files = [f for f in args.files.read_text(encoding="utf-8").split() if f]
        log = (
            args.log.read_text(encoding="utf-8", errors="replace")
            if args.log.is_file()
            else ""
        )
        why = tidy_incomplete(log, args.exit_code)
        if why:
            print(
                f"{why}; analysis incomplete, no clang-tidy summary written",
                file=sys.stderr,
            )
            return 1
        piece = summarize_tidy(log, files, enabled_pattern(args.config), root)
        write_json(args.out, piece)
        print(
            f"clang-tidy: {piece['enabled_finding_count']} in enabled files, "
            f"{piece['outside_count']} outside; {len(piece['ready'])} files ready to enable",
            file=sys.stderr,
        )
        return 0

    now = now or datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    previous = read_json(args.previous)
    pieces = {}
    for path in args.piece:
        piece = read_json(path)
        if piece and piece.get("check") in CHECKS:
            pieces[piece["check"]] = piece
        else:
            print(
                f"{path}: no summary (its job failed or was skipped)", file=sys.stderr
            )
    coverage = None
    if args.coverage and args.coverage.is_file():
        coverage = coverage_status(
            read_lcov(args.coverage.read_text(encoding="utf-8"), root),
            (previous or {}).get("coverage"),
        )
    scanning = None
    if args.code_scanning:
        try:
            scanning = scanning_status(
                github(os.environ.get("GITHUB_TOKEN", "")),
                os.environ.get("GITHUB_REPOSITORY", "Xilinx/mlir-aie"),
                (previous or {}).get("code_scanning"),
            )
        except ApiError as e:
            scanning = str(e)
    status = build_status(pieces, coverage, scanning, previous, now, run_info())
    history = update_history(read_json(args.history), history_point(status))
    write_json(args.out / "status.json", status)
    write_json(args.out / "history.json", history)
    print(
        f"code health: {', '.join(sorted(pieces)) or 'no pieces'}"
        f"{', coverage' if coverage else ''}{', code scanning' if args.code_scanning else ''}; "
        f"{len(history['points'])} history points",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
