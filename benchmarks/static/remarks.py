# benchmarks/static/remarks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Parse Peano optimization-record YAML into per-kernel static metrics.

Record shapes below were read off real ``-fsave-optimization-record`` output
from llvm-aie 22.0.0.2026090201 (see STATIC_CHECKS.md for the probe); the
pass that schedules loops reports as ``pipeliner``, not ``postpipeliner``.

  pipeliner             Passed   schedule        -> II, NS, Loop, PrologueBundles,
                                                    EpilogueBundles, Pipeliner
  pipeliner             Missed   canPipelineLoop -> "Failed to pipeline loop";
                                                    loop identified by DebugLoc
  pipeliner             Analysis schedule        -> MII / SwpMaxMii ("Minimal
                                                    Initiation Interval too
                                                    large"), "Unable to find
                                                    schedule"
  aie-hardware-loops    Analysis analysis        -> BasicBlock, Zero-Overhead-Loop
  aie-asm-printer       Analysis analysis        -> BasicBlock, BundleCount, ByteCount
  aie-multi-slot-pseudo Missed   missing-memory-bank -> a load with no DM bank
plus the stderr warnings channel (-Wpass-failed, "No memory bank assigned").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

PASSES = "pipeliner|aie-hardware-loops|aie-asm-printer|aie-multi-slot-pseudo"
REMARK_FLAGS = [
    "-fsave-optimization-record",
    f"-foptimization-record-passes={PASSES}",
    # human-readable copies on stderr as well; harmless in CI logs
    "-Rpass=pipeliner",
    "-Rpass-missed=pipeliner|aie-multi-slot-pseudo",
    "-Rpass-analysis=pipeliner|aie-hardware-loops|aie-asm-printer",
]

_WARN_PASS_FAILED = re.compile(
    r"^.*warning: .*\[-Wpass-failed[^\]]*\].*$", re.MULTILINE
)
_WARN_NO_BANK = re.compile(r"warning: No memory bank assigned to load")


# LLVM remark YAML uses !Passed / !Missed / !Analysis tags; treat them as plain
# mappings and remember the tag.
class _Loader(yaml.SafeLoader):
    pass


def _tagged(loader, tag_suffix, node):
    m = loader.construct_mapping(node, deep=True)
    m["_kind"] = tag_suffix
    return m


_Loader.add_multi_constructor("!", _tagged)


def _args(rec: dict) -> dict:
    """Flatten the remark 'Args' list of single-key dicts into one dict.

    Repeated ``String`` fragments (the human-readable message is split around
    the typed values) are concatenated so the whole sentence survives.
    """
    out: dict = {}
    for a in rec.get("Args", []) or []:
        if not isinstance(a, dict):
            continue
        for k, v in a.items():
            if k == "DebugLoc":
                continue
            if k == "String":
                out[k] = out.get(k, "") + str(v)
            else:
                out[k] = v
    return out


def _message(rec: dict) -> str:
    """The remark sentence as clang prints it: every Arg value, in order.

    The typed values (MII, SwpMaxMii, II, ...) sit between ``String``
    fragments; joining only the strings would print
    "Minimal Initiation Interval too large:  > ." with the numbers missing.
    """
    parts = []
    for a in rec.get("Args", []) or []:
        if isinstance(a, dict):
            parts.extend(str(v) for k, v in a.items() if k != "DebugLoc")
    return "".join(parts)


def _loc(rec: dict) -> str:
    """``L<line>`` from the record's top-level DebugLoc; the only handle the
    Missed records give for *which* loop the pipeliner gave up on."""
    loc = rec.get("DebugLoc") or {}
    return f"L{loc.get('Line', '?')}"


def _source(rec: dict) -> tuple[str | None, int | None]:
    """``(file, line)`` of the record's DebugLoc, or ``(None, None)``.

    The file is the one the loop was *inlined from* -- for a vector kernel
    that is usually an ``aie_api`` header, not the kernel source -- which
    is still the line a reader has to open to see what the pipeliner
    scheduled.
    """
    loc = rec.get("DebugLoc") or {}
    return (str(loc["File"]) if "File" in loc else None), _int(loc.get("Line"))


_MBB_PREFIX = re.compile(r"^bb\.\d+\.")


def _block(name) -> str | None:
    """Normalise a basic-block name so the three passes key the same loop.

    The pipeliner reports machine-basic-block names (``bb.1.for.body.i``);
    aie-hardware-loops and aie-asm-printer report the IR block
    (``for.body.i``). Strip the ``bb.<n>.`` prefix so II, ZOL and bundle
    counts land on one LoopInfo instead of two half-filled ones.
    """
    if name is None:
        return None
    return _MBB_PREFIX.sub("", str(name)) or None


@dataclass
class LoopInfo:
    function: str
    block: str
    ii: int | None = None
    ns: int | None = None
    prologue_bundles: int | None = None
    epilogue_bundles: int | None = None
    pipelined: bool | None = None  # True Passed, False Missed, None unseen
    pipeliner: str | None = None  # which engine produced the schedule
    missed_reason: str | None = None
    zol: bool | None = None
    bundle_count: int | None = None
    byte_count: int | None = None
    # Where the scheduled (or declined) loop lives, from the record's DebugLoc.
    file: str | None = None
    line: int | None = None


@dataclass
class StaticReport:
    loops: dict[tuple[str, str], LoopInfo] = field(default_factory=dict)
    pm_bytes_by_function: dict[str, int] = field(default_factory=dict)
    missing_bank_loads: int = 0
    pass_failed_warnings: int = 0
    # The -Wpass-failed lines themselves ("loop not unrolled: ..."), each
    # naming file:line:col; a dropped pragma is a kernel-source bug, and the
    # line is what the static workflow annotates on a pull request.
    pass_failed: list[str] = field(default_factory=list)
    # pipeliner Analysis/schedule notes ("MII too large", "Unable to find
    # schedule"): kept for the meta file, not a graph series.
    schedule_notes: list[str] = field(default_factory=list)

    def loop(self, fn: str, bb: str) -> LoopInfo:
        return self.loops.setdefault((fn, bb), LoopInfo(fn, bb))

    # ---- aggregates used as benchmark rows (all smaller-is-better) ----
    @property
    def unpipelined_loops(self) -> int:
        return sum(1 for loop in self.loops.values() if loop.pipelined is False)

    @property
    def non_zol_loops(self) -> int:
        return sum(1 for loop in self.loops.values() if loop.zol is False)

    @property
    def pm_bytes(self) -> int:
        return sum(self.pm_bytes_by_function.values())


def parse_yaml(path: str | Path, report: StaticReport | None = None) -> StaticReport:
    r = report or StaticReport()
    with open(path) as f:
        docs = list(yaml.load_all(f, Loader=_Loader))
    for d in docs:
        if not isinstance(d, dict):
            continue
        p, name, kind = d.get("Pass"), d.get("Name"), d.get("_kind")
        fn = d.get("Function", "?")
        a = _args(d)
        if p == "pipeliner":
            if kind == "Passed" and name == "schedule":
                # A schedule with no Loop arg has been seen (zero_scalar_*);
                # fall back to the source line rather than a shared "?" key.
                loop = r.loop(fn, _block(a.get("Loop")) or _loc(d))
                loop.pipelined = True
                loop.ii = _int(a.get("II"))
                loop.ns = _int(a.get("NS"))
                loop.prologue_bundles = _int(a.get("PrologueBundles"))
                loop.epilogue_bundles = _int(a.get("EpilogueBundles"))
                loop.pipeliner = str(a.get("Pipeliner", "")) or None
                loop.file, loop.line = _source(d)
            elif kind == "Missed" and name == "canPipelineLoop":
                loop = r.loop(fn, _loc(d))
                loop.pipelined = False
                loop.missed_reason = str(a.get("String", "")) or None
                loop.file, loop.line = _source(d)
            elif kind == "Analysis" and name == "schedule":
                r.schedule_notes.append(f"{fn}@{_loc(d)}: {_message(d)}")
        elif p == "aie-hardware-loops":
            bb = str(a.get("BasicBlock", "?"))
            r.loop(fn, bb).zol = str(a.get("Zero-Overhead-Loop", "")).lower() == "true"
        elif p == "aie-asm-printer":
            bb = str(a.get("BasicBlock", "?"))
            bc, byc = _int(a.get("BundleCount")), _int(a.get("ByteCount"))
            r.pm_bytes_by_function[fn] = r.pm_bytes_by_function.get(fn, 0) + (byc or 0)
            if (fn, bb) in r.loops:  # only annotate known loop blocks
                r.loops[(fn, bb)].bundle_count = bc
                r.loops[(fn, bb)].byte_count = byc
        elif p == "aie-multi-slot-pseudo" and name == "missing-memory-bank":
            r.missing_bank_loads += 1
    return r


def parse_stderr(text: str, report: StaticReport) -> StaticReport:
    dropped = [line.strip() for line in _WARN_PASS_FAILED.findall(text)]
    report.pass_failed_warnings += len(dropped)
    report.pass_failed.extend(dropped)
    # The stderr warning duplicates the YAML remark; only count it when no
    # YAML record was produced for the build (keeps the two channels consistent).
    if report.missing_bank_loads == 0:
        report.missing_bank_loads = len(_WARN_NO_BANK.findall(text))
    return report


def _int(v):
    try:
        return int(str(v).strip("'\""))
    except (TypeError, ValueError):
        return None


# ---- GitHub Actions annotations ------------------------------------------
#
# The workflow never comments on a pull request for these; a dropped pragma
# or a kernel that fails to compile is attached to the line it names via a
# workflow command (``::warning file=...,line=...::message``), which GitHub
# renders in the checks summary and on the Files tab. Outside Actions the
# lines are just printed.

_DIAG_RE = re.compile(r"^(?P<file>.+?):(?P<line>\d+):(?P<col>\d+): (?:warning|error): ")


def _escape(text: str, *, prop: bool = False) -> str:
    text = str(text).replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    if prop:
        text = text.replace(":", "%3A").replace(",", "%2C")
    return text


def annotation(
    level: str,
    message: str,
    *,
    title: str | None = None,
    file: str | None = None,
    line: int | None = None,
    root: str | None = None,
) -> str:
    """One workflow command: ``::<level> file=..,line=..,title=..::<message>``.

    ``file`` is made relative to ``root`` (the checkout) when it lies inside
    it; a path outside the checkout (a wheel header) is dropped, since the
    annotation could not be placed on a file of the pull request anyway.
    """
    props = []
    if file and root:
        try:
            file = str(Path(file).resolve().relative_to(Path(root).resolve()))
        except ValueError:
            file = None
    if file:
        props.append(f"file={_escape(file, prop=True)}")
        if line is not None:
            props.append(f"line={line}")
    if title:
        props.append(f"title={_escape(title, prop=True)}")
    head = f"::{level}" + (" " + ",".join(props) if props else "")
    return f"{head}::{_escape(message)}"


def annotations(
    name: str, report: StaticReport | None, detail: str, root: str | None = None
) -> list[str]:
    """Annotations for one kernel build: its dropped pragmas, or its compile failure."""
    if report is None:
        first = next((ln for ln in detail.splitlines() if "error:" in ln), None)
        return [
            annotation(
                "error",
                first or detail.strip()[-500:],
                title=f"{name}: kernel failed to compile",
            )
        ]
    out = []
    for warning in report.pass_failed:
        m = _DIAG_RE.match(warning)
        out.append(
            annotation(
                "warning",
                warning[m.end() :] if m else warning,
                title=f"{name}: pragma dropped by the compiler",
                file=m.group("file") if m else None,
                line=int(m.group("line")) if m else None,
                root=root,
            )
        )
    return out


def rows(report: StaticReport, prefix: str, extra: str) -> list[dict]:
    """benchmark-action rows for one kernel build. Smaller is better throughout."""
    from ..kernels._util import row

    out = [
        row(f"{prefix}/unpipelined_loops", "loops", report.unpipelined_loops, extra),
        row(f"{prefix}/non_zol_loops", "loops", report.non_zol_loops, extra),
        row(f"{prefix}/missing_bank_loads", "loads", report.missing_bank_loads, extra),
        row(
            f"{prefix}/pass_failed_warnings",
            "warnings",
            report.pass_failed_warnings,
            extra,
        ),
        row(f"{prefix}/pm_bytes", "bytes", report.pm_bytes, extra),
    ]
    for (fn, bb), loop in sorted(report.loops.items()):
        if loop.ii is not None:
            # The hover text names the source line, so a reader of an II
            # alert can open the loop without decoding a basic-block name.
            where = (
                f" at {Path(loop.file).name}:{loop.line}"
                if loop.file and loop.line is not None
                else ""
            )
            out.append(
                row(
                    f"{prefix}/loop/{fn}/{bb}/II",
                    "cycles",
                    loop.ii,
                    extra,
                    f"NS={loop.ns} pro={loop.prologue_bundles} "
                    f"epi={loop.epilogue_bundles} zol={loop.zol} "
                    f"via={loop.pipeliner}{where}",
                )
            )
    return out
