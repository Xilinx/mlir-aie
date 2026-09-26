# remarks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Static kernel checks: compile every library kernel with Peano and read its remarks.

    python -m aie.utils.compile.remarks --target aie2p --out static.json \
        --out-pm static-pm.json --meta static-meta.json
    python -m aie.utils.compile.remarks --target aie2p --only '^gelu' \
        --out static.json --sources . --baseline-sources ../mlir-aie-base
    python -m aie.utils.compile.remarks --target aie2p --only '^tanh/' \
        --cases test/python/npu/kernel_cases.py --out static.json
    python -m aie.utils.compile.remarks --target aie2p --out static.json \
        --build cascade_mm:dim_m=16,dim_k=24,dim_n=32 --build cascade_mm

CPU-only. Every factory in ``aie.iron.kernels`` (at its defaults and for each
entry of its ``.dtypes`` table) is compiled exactly as the JIT compiles it
(``aie.utils.compile.utils.cxx_core_compile_command``), plus the
optimization-record flags below, and the records become per-kernel series
for benchmark-action: a Peano bump that changes a loop's schedule shows up
here before anyone looks at device numbers. With ``--sources DIR`` (or
``MLIR_AIE_KERNEL_SOURCES=DIR``) naming a checkout, that checkout's
``aie_kernels/`` and ``aie_runtime_lib/`` are compiled instead of the
installed copies. With ``--cases``, the builds
are the ones a cases file's tests run (shape and options baked in), each
named by its case; with ``--build``, the factory builds named on the
command line, for a shape that picks a code path no default reaches.

Record shapes, as llvm-aie 22.0.0.2026090201 emits them (they are Peano's,
not LLVM's documented ones):

| Pass | Kind / Name | Args | Tracked as |
| --- | --- | --- | --- |
| ``pipeliner`` | ``Passed`` / ``schedule`` | ``II``, ``NS``, ``Loop``, ``Pipeliner``, prologue/epilogue bundles | ``loop/<fn>/<bb>/II`` (rest as hover text) |
| ``pipeliner`` | ``Missed`` / ``canPipelineLoop`` | "Failed to pipeline loop"; located by ``DebugLoc`` only | ``unpipelined_loops`` (keyed ``L<line>``) |
| ``pipeliner`` | ``Analysis`` / ``schedule`` | ``MII``, ``SwpMaxMii``, "Unable to find schedule" | ``schedule_notes`` in the meta file |
| ``aie-hardware-loops`` | ``Analysis`` / ``analysis`` | ``BasicBlock``, ``Zero-Overhead-Loop`` | ``non_zol_loops``, and ``loop/<fn>/<bb>/not_zol`` per loop |
| ``aie-asm-printer`` | ``Analysis`` / ``analysis`` | ``BasicBlock``, ``BundleCount``, ``ByteCount`` | ``pm_bytes`` (summed over the shipped functions) |
| ``aie-multi-slot-pseudo`` | ``Missed`` / ``missing-memory-bank`` | ``Instruction`` | ``missing_bank_loads`` |
| stderr | ``-Wpass-failed`` | a ``#pragma clang loop`` / ``AIE_*`` macro the compiler dropped | ``pass_failed_warnings``, text kept |
| the object | ``llvm-readobj`` sections, symbols, relocations | what the entry symbol reaches | the shipped functions; ``libcalls`` (e.g. ``__divsf3``) |
| the object | ``llvm-readobj --stack-sizes`` (``-fstack-size-section``) | frame bytes per function | ``kernel_stack_bytes``, the deepest path from the entry |
| the IR (``-emit-llvm``) | ``opt`` ``print<scalar-evolution>`` | a constant backedge-taken count | ``loop/<fn>/<bb>/II_x_trips`` |

The loop counts and ``pm_bytes`` cover only the functions the entry symbol
reaches in the object, which are the ones the core link keeps.
``kernel_stack_bytes`` counts the kernel's frames only: the core that calls
it also holds ``main``'s, which aiecc's measured stack size includes
(dwconv1d_channels_last on AIE2P: 64 here, 256 for the core). Over the
contract's ``stack_bytes`` (else the device default) it prints a warning:
the design reserves that much for the whole core, and an overflow corrupts
the neighbouring memory without a fault.
``--baseline-sources DIR`` compiles everything a second time from ``DIR``
and prints each row that differs, for a before/after of a kernel change;
a loop LLVM only renamed, its rows unchanged, is counted but not listed,
and a build either tree fails to compile is named and left out.
It names the ``aie_kernels/`` this tree compiled and warns when that is the
installed copy (``MLIR_AIE_KERNEL_SOURCES`` unset) or the baseline itself.
``--keep DIR`` keeps the objects, which ``--meta`` names per build.

The loop-scheduling pass reports as ``pipeliner`` (a ``postpipeliner``
filter records nothing); it names loops by machine basic block
(``bb.1.for.body.i``) while the other passes use the IR block
(``for.body.i``), so the prefix is stripped to join them.
``unpipelined_loops`` counts every loop the pipeliner declined, outer loops
included, so its change is the signal, not its value.

The integer series alert on any increase; ``pm_bytes`` goes to its own file
(``--out-pm``) so it can carry a percentage threshold. Nothing gates: under
GitHub Actions a dropped pragma is a ``::warning`` at its file and line, a
kernel that fails to compile an ``::error``, and the run then exits 3 with
"RESULTS INVALID", after writing the rows of the builds that compiled.
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import contextlib
import enum
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

import yaml

from .utils import cxx_core_compile_command

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


_Loader.add_multi_constructor(
    "!",
    lambda loader, tag, node: {
        **loader.construct_mapping(node, deep=True),
        "_kind": tag,
    },
)


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


def _loc(rec: dict) -> str:
    """Return ``L<line>`` from the record's top-level DebugLoc.

    It is the only handle the Missed records give for *which* loop the
    pipeliner gave up on.
    """
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
    """Normalize a basic-block name so the three passes key the same loop.

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
    # Iterations per entry into the loop, when the IR's is a constant.
    trips: int | None = None
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
    # From the object (``linked``): the functions the entry symbol reaches,
    # None until it has been read, and the runtime-library calls among them.
    shipped: set[str] | None = None
    libcalls: list[str] = field(default_factory=list)
    # The kernel's own call path, not the core's: main's frame is not in it.
    kernel_stack_bytes: int | None = None

    def loop(self, fn: str, bb: str) -> LoopInfo:
        return self.loops.setdefault((fn, bb), LoopInfo(fn, bb))

    def ships(self, fn: str) -> bool:
        return self.shipped is None or fn in self.shipped

    # ---- aggregates used as benchmark rows (all smaller-is-better) ----
    # Over the shipped functions only: a standalone copy of an inlined helper
    # would count its loops and bytes a second time.
    @property
    def unpipelined_loops(self) -> int:
        return sum(
            1
            for loop in self.loops.values()
            if loop.pipelined is False and self.ships(loop.function)
        )

    @property
    def non_zol_loops(self) -> int:
        return sum(
            1
            for loop in self.loops.values()
            if loop.zol is False and self.ships(loop.function)
        )

    @property
    def pm_bytes(self) -> int:
        return sum(n for fn, n in self.pm_bytes_by_function.items() if self.ships(fn))


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
                # The sentence as clang prints it: every Arg value in order,
                # since the typed values (MII, II, ...) sit between the
                # String fragments and joining only the strings would print
                # "Minimal Initiation Interval too large:  > ." without them.
                message = "".join(
                    str(v)
                    for arg in d.get("Args", []) or []
                    if isinstance(arg, dict)
                    for k, v in arg.items()
                    if k != "DebugLoc"
                )
                r.schedule_notes.append(f"{fn}@{_loc(d)}: {message}")
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
    annotation could not be placed on a file of the pull request anyway. The
    relative path is emitted with forward slashes on every runner OS, which is
    what GitHub matches against the files of the pull request.
    """
    props = []
    if file and root:
        try:
            file = Path(file).resolve().relative_to(Path(root).resolve()).as_posix()
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


def workflow_annotations(
    name: str,
    report: StaticReport | None,
    detail: str,
    root: str | None = None,
    seen: set[tuple] | None = None,
) -> list[str]:
    """Annotations for one kernel build: its dropped pragmas, or its compile failure.

    One source is compiled once per factory build and per target, so the same
    dropped pragma comes back several times a run; ``seen`` (shared across
    builds) keeps each file, line and message to its first annotation.
    """
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
        key = (
            (m.group("file"), m.group("line"), warning[m.end() :]) if m else (warning,)
        )
        if seen is not None:
            if key in seen:
                continue
            seen.add(key)
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


# ---- what the linked kernel keeps ----------------------------------------
#
# Peano gives every function its own section (``.text.<fn>``) and the core
# link drops unreferenced sections, so what ships is what the entry symbol
# reaches through relocations. A helper that is inlined at its call sites and
# also emitted standalone has asm-printer records for both; counting only the
# reached functions keeps it from counting twice. A relocation from a reached
# section to an undefined symbol is a call into the runtime library -- on
# AIE2P that is scalar float divide, int-to-float and the like, each a
# software routine of hundreds of cycles.


@dataclass
class Linked:
    functions: set[str]
    undefined: list[str]
    # Bytes of stack on the deepest call path from the entry, from the
    # ``.stack_sizes`` section; None without one, or on recursion. Runtime
    # routines are outside the object and not counted.
    stack: int | None = None


def linked(obj: Path, entry: str) -> Linked:
    """Functions and undefined symbols ``entry`` reaches in the object ``obj``."""
    from aie.utils import config

    out = subprocess.run(
        [
            config.readobj_path(),
            "--elf-output-style=JSON",
            "--sections",
            "--symbols",
            "--relocations",
            "--stack-sizes",
            str(obj),
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return parse_readobj(json.loads(out)[0], entry)


def parse_readobj(doc: dict, entry: str) -> Linked:
    """``linked`` on the parsed ``llvm-readobj --elf-output-style=JSON`` document."""
    alloc = {}
    for s in doc["Sections"]:
        s = s["Section"]
        alloc[s["Index"]] = any(f["Name"] == "SHF_ALLOC" for f in s["Flags"]["Flags"])
    symbols = [s["Symbol"] for s in doc["Symbols"]]
    # A symbol names its section by index; an undefined one names index 0.
    home = [s["Section"]["Value"] or None for s in symbols]
    functions = {}
    for s, sec in zip(symbols, home):
        if s["Type"]["Name"] == "Function" and sec is not None:
            functions.setdefault(sec, set()).add(s["Name"]["Name"])
    edges: dict[int, set[int]] = {}
    calls: dict[int, set[str]] = {}
    for rel in doc.get("Relocations", []):
        # A relocation section applies to the section its sh_info names.
        src = next(
            s["Section"]["Info"]
            for s in doc["Sections"]
            if s["Section"]["Index"] == rel["SectionIndex"]
        )
        for r in rel["Relocs"]:
            i = r["Relocation"]["Symbol"]["Value"]
            if home[i] is not None:
                edges.setdefault(src, set()).add(home[i])
            elif symbols[i]["Section"]["Name"] == "Undefined":
                calls.setdefault(src, set()).add(symbols[i]["Name"]["Name"])
    frames = {
        name: e["Entry"]["Size"]
        for e in doc.get("StackSizes", [])
        for name in e["Entry"]["Functions"]
    }

    def deepest(sec: int, path: frozenset) -> int | None:
        if sec in path:
            return None
        below = 0
        # Only code sections: a jump table in .rodata points back at its
        # function and would read as recursion.
        for callee in edges.get(sec, set()) - {sec}:
            if callee in functions:
                d = deepest(callee, path | {sec})
                if d is None:
                    return None
                below = max(below, d)
        return (
            max((frames.get(n, 0) for n in functions.get(sec, ())), default=0) + below
        )

    roots = [sec for sec, names in functions.items() if entry in names]
    todo = list(roots)
    seen: set[int] = set()
    while todo:
        sec = todo.pop()
        if sec in seen or not alloc.get(sec):
            continue
        seen.add(sec)
        todo.extend(edges.get(sec, ()))
    return Linked(
        functions={n for sec in seen for n in functions.get(sec, ())},
        undefined=sorted({u for sec in seen for u in calls.get(sec, ())}),
        stack=deepest(roots[0], frozenset()) if frames and roots else None,
    )


def _row(name: str, unit: str, value, extra: str, rng: str | None = None) -> dict:
    r = {"name": name, "unit": unit, "value": value, "extra": extra}
    if rng:
        r["range"] = rng
    return r


# ---- trace markers in the optimized IR -----------------------------------
#
# A cycle count is one event0 -> event1 interval per kernel call, so the
# markers must bracket the entry symbol's whole call: exactly one event0 then
# one event1 on every path to ``ret``, and none inside a loop. The -O2 IR keeps
# the control flow that answers this; the object's disassembly does not. A
# marker in a sibling function the entry never calls (``zero.cc`` included
# beside ``mm.cc``) is not reachable, and one around an inlined helper called
# in a loop sits inside that loop, so neither counts.

_IR_DEFINE = re.compile(r"^define [^@]*@([-\w.$]+)\(")
# LLVM quotes a label holding a `$`, as an inlined lambda's exit block does.
_IR_LABEL = re.compile(r'^"?([-\w.$]+)"?:')
_IR_SUCC = re.compile(r'\blabel %"?([-\w.$]+)')
_IR_EVENT = re.compile(r"@llvm\.aie\w*\.event\(i32 ([01])\)")
_IR_CALL = re.compile(r"\bcall\b[^@]*@([-\w.$]+)\(")
# Longer event sequences are wrong whatever they say, so stop growing them.
_MAX_EVENTS = 4


def _ir_functions(ir: str) -> dict[str, list[tuple[str, list[str]]]]:
    """Return ``{function: [(block, lines)]}``; an unnamed entry block is ``"0"``."""
    functions: dict[str, list[tuple[str, list[str]]]] = {}
    blocks = None
    for line in ir.splitlines():
        if blocks is None:
            if m := _IR_DEFINE.match(line):
                blocks = functions.setdefault(m.group(1), [("0", [])])
        elif line.startswith("}"):
            blocks = None
        elif m := _IR_LABEL.match(line):
            blocks.append((m.group(1), []))
        elif line.strip():
            blocks[-1][1].append(line)
    return {f: [b for b in bs if b[1] or b[0] != "0"] for f, bs in functions.items()}


def _in_loop(succs: dict[str, list[str]], block: str) -> bool:
    """Whether ``block`` can reach itself, a self-loop included."""
    seen: set[str] = set()
    work = list(succs[block])
    while work:
        b = work.pop()
        if b == block:
            return True
        if b not in seen:
            seen.add(b)
            work.extend(succs[b])
    return False


def _marker_paths(functions, name: str, memo: dict) -> set[str] | str:
    """Return the event0/event1 sequences a call of ``name`` can emit, or why they are not a set.

    ``{"01"}`` is one whole-call pair on every path, ``{""}`` no markers. A
    string return is the reason the markers are not per call (one inside a
    loop, or recursion).
    """
    if name in memo:
        return memo[name] if memo[name] is not None else f"{name} recurses"
    memo[name] = None
    blocks = functions[name]
    succs = {b: _IR_SUCC.findall("\n".join(lines)) for b, lines in blocks}
    local: dict[str, set[str]] = {}
    for b, lines in blocks:
        seqs = {""}
        for line in lines:
            if m := _IR_EVENT.search(line):
                step: set[str] | str = {m.group(1)}
            elif (m := _IR_CALL.search(line)) and m.group(1) in functions:
                step = _marker_paths(functions, m.group(1), memo)
                if isinstance(step, str):
                    memo[name] = step
                    return step
            else:
                continue
            seqs = {(s + t)[:_MAX_EVENTS] for s in seqs for t in step}
        local[b] = seqs
    for b, _ in blocks:
        if local[b] != {""} and _in_loop(succs, b):
            memo[name] = f"a marker in {name} sits inside a loop (block {b})"
            return memo[name]
    # Paths from the entry block: acyclic once the event-free loops are
    # collapsed, so a worklist of (block, sequence) pairs terminates.
    body = dict(blocks)
    entry = blocks[0][0]
    seen: set[tuple[str, str]] = set()
    work = [(entry, s) for s in local[entry]]
    ends: set[str] = set()
    while work:
        b, s = work.pop()
        if (b, s) in seen:
            continue
        seen.add((b, s))
        if body[b] and body[b][-1].lstrip().startswith("ret"):
            ends.add(s)
        for n in succs[b]:
            work.extend((n, (s + t)[:_MAX_EVENTS]) for t in local[n])
    memo[name] = ends or {""}
    return memo[name]


def trace_markers(ir: str, entry: str) -> str:
    """Classify how ``event0()``/``event1()`` bracket a call of ``entry``.

    ``"whole_call"`` when one pair brackets every call and nothing else
    emits a marker, ``"none"`` when a call emits no marker at all, and
    otherwise a sentence saying what the markers do instead.
    """
    functions = _ir_functions(ir)
    if entry not in functions:
        raise ValueError(f"{entry} is not defined in the compiled IR")
    paths = _marker_paths(functions, entry, {})
    if isinstance(paths, str):
        return paths
    if paths == {"01"}:
        return "whole_call"
    if paths == {""}:
        return "none"
    shown = ", ".join(repr(p) for p in sorted(paths))
    return (
        f"a call of {entry} emits marker sequences {shown}, not one event0 then event1"
    )


def report_rows(report: StaticReport, prefix: str, extra: str) -> list[dict]:
    """benchmark-action rows for one kernel build. Smaller is better throughout."""
    out = [
        _row(f"{prefix}/unpipelined_loops", "loops", report.unpipelined_loops, extra),
        _row(f"{prefix}/non_zol_loops", "loops", report.non_zol_loops, extra),
        _row(f"{prefix}/missing_bank_loads", "loads", report.missing_bank_loads, extra),
        _row(
            f"{prefix}/pass_failed_warnings",
            "warnings",
            report.pass_failed_warnings,
            extra,
        ),
        _row(f"{prefix}/pm_bytes", "bytes", report.pm_bytes, extra),
        _row(
            f"{prefix}/libcalls",
            "symbols",
            len(report.libcalls),
            extra,
            " ".join(report.libcalls) or None,
        ),
    ]
    if report.kernel_stack_bytes is not None:
        out.append(
            _row(
                f"{prefix}/kernel_stack_bytes",
                "bytes",
                report.kernel_stack_bytes,
                extra,
                "the kernel's frames; the core adds main's"
                + (" and the runtime routines'" if report.libcalls else ""),
            )
        )
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
                _row(
                    f"{prefix}/loop/{fn}/{bb}/II",
                    "cycles",
                    loop.ii,
                    extra,
                    f"NS={loop.ns} pro={loop.prologue_bundles} "
                    f"epi={loop.epilogue_bundles} zol={loop.zol} "
                    f"trips={loop.trips} via={loop.pipeliner}{where}",
                )
            )
            if loop.trips is not None:
                # II alone reads a loop that does 4 blocks per iteration as
                # slower than one that does 1; II x trips compares them.
                out.append(
                    _row(
                        f"{prefix}/loop/{fn}/{bb}/II_x_trips",
                        "cycles",
                        loop.ii * loop.trips,
                        extra,
                        f"{loop.trips} trips per entry{where}",
                    )
                )
        if loop.zol is not None:
            # 1 when the loop is not a zero-overhead loop: an inner loop that
            # falls off the hardware loop unit costs its bundle count in
            # branches, whatever its II says.
            out.append(
                _row(
                    f"{prefix}/loop/{fn}/{bb}/not_zol",
                    "loops",
                    int(not loop.zol),
                    extra,
                )
            )
    return out


# --------------------------------------------------------------------------
# Compiling the library
# --------------------------------------------------------------------------

# Cheap upstream clang checks on top of the library's own flags, never
# instead of them.
_EXTRA_WARNINGS = [
    "-Wcast-align",
    "-Walign-mismatch",
    "-Wunaligned-access",
    "-Wframe-larger-than=1024",
]


def _kernel_file(ext_fn, out_dir: Path) -> tuple[Path, list[str]]:
    if ext_fn.use_chess:
        raise ValueError(
            f"{ext_fn.name}: built with xchesscc; Peano remarks do not apply"
        )
    include_dirs = list(ext_fn.include_dirs)
    if ext_fn.source_file is not None:
        src = Path(ext_fn.source_file)
        # As the JIT: the source's own directory, for "../aie_kernel_utils.h".
        if str(src.parent) not in include_dirs:
            include_dirs.append(str(src.parent))
    else:
        src = out_dir / f"{ext_fn.name}.cc"
        src.write_text(ext_fn.source_string)
    return src, include_dirs


# clang's "error:" and "fatal error:", "LLVM ERROR:", and a failed assertion.
_ERROR_LINE = re.compile(r"(?i)\berror:|\bAssertion .* failed")


def compile_failure(stderr: str, *, first: int = 5, tail: int = 2000) -> str:
    """Keep a failed compile's first error lines, then its last ``tail`` characters.

    The first error names the cause, and a template error's notes or a
    crash's stack dump push it far above the end of the output, which ends
    in "3 errors generated." or a backtrace. The error lines already inside
    the tail are not repeated.
    """
    text = stderr.strip()
    if len(text) <= tail:
        return text
    end = text[-tail:]
    errors = [ln for ln in text.splitlines() if _ERROR_LINE.search(ln)]
    head = [ln for ln in errors[:first] if ln not in end]
    if len(errors) > first:
        head.append(f"({len(errors) - first} more error lines)")
    return "\n".join([*head, "...", end]) if head else end


def entry_symbol(ext_fn) -> str:
    """Return the symbol the kernel source defines, before the JIT's per-build prefix."""
    return getattr(ext_fn, "_original_name", ext_fn.name)


def trace_shape(ext_fn, target: str, out_dir: Path) -> str:
    """Compile ``ext_fn`` to -O2 IR and classify its entry's markers (``trace_markers``)."""
    src, include_dirs = _kernel_file(ext_fn, out_dir)
    cmd = cxx_core_compile_command(
        str(src),
        target,
        str(out_dir / f"{ext_fn.name}.ll"),
        include_dirs=include_dirs,
        compile_args=list(ext_fn.compile_flags),
        inline=True,
    )
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"{ext_fn.name}: {compile_failure(p.stderr)}")
    ir = (out_dir / f"{ext_fn.name}.ll").read_text()
    return trace_markers(ir, entry_symbol(ext_fn))


def compile_command(ext_fn, target: str, out_dir: Path) -> tuple[list[str], Path]:
    """Return the exact Peano command the JIT would run for ``ext_fn``, plus remark flags.

    Inline-source kernels (the aie2 LUT activations) are written out under the
    kernel's symbol name first, as the JIT does. Kernels built with
    ``use_chess`` are rejected: the remarks are Peano's.
    """
    src, include_dirs = _kernel_file(ext_fn, out_dir)
    yaml_out = out_dir / f"{ext_fn.name}.opt.yaml"
    cmd = cxx_core_compile_command(
        str(src),
        target,
        str(out_dir / f"{ext_fn.name}.o"),
        include_dirs=include_dirs,
        compile_args=[
            *ext_fn.compile_flags,
            *REMARK_FLAGS,
            f"-foptimization-record-file={yaml_out}",
            *_EXTRA_WARNINGS,
            # A non-allocated section of frame sizes; the code is unchanged.
            "-fstack-size-section",
        ],
    )
    return cmd, yaml_out


_SCEV_FUNCTION = re.compile(r"^Determining loop execution counts for: @\"?([^\"]+)\"?$")
_SCEV_COUNT = re.compile(
    r"^Loop %\"?([^\":]+)\"?: backedge-taken count is (?:i\d+ )?(\d+)$"
)


def parse_trip_counts(scev: str) -> dict[tuple[str, str], int]:
    """Map ``(function, header block)`` to trips, from ``print<scalar-evolution>``.

    Only loops whose backedge-taken count is a constant; the trip count is
    one more.
    """
    trips, fn = {}, None
    for line in scev.splitlines():
        if m := _SCEV_FUNCTION.match(line):
            fn = m.group(1)
        elif (m := _SCEV_COUNT.match(line)) and fn is not None:
            trips[fn, m.group(1)] = int(m.group(2)) + 1
    return trips


def trip_counts(ext_fn, target: str, out_dir: Path) -> dict[tuple[str, str], int]:
    """Constant trip counts of the loops in the optimized IR the backend compiles.

    The build's command, stopped before code generation, with value names
    kept so its blocks carry the names the remarks use. Empty if it fails:
    trips annotate the report, they do not gate it.
    """
    src, include_dirs = _kernel_file(ext_fn, out_dir)
    ll = out_dir / f"{ext_fn.name}.ll"
    cmd = cxx_core_compile_command(
        str(src),
        target,
        str(ll),
        include_dirs=include_dirs,
        compile_args=[*ext_fn.compile_flags, "-fno-discard-value-names"],
        inline=True,
    )
    if subprocess.run(cmd, capture_output=True).returncode != 0:
        return {}
    opt = Path(cmd[0]).with_name("opt")
    p = subprocess.run(
        [str(opt), "-passes=print<scalar-evolution>", "-disable-output", str(ll)],
        capture_output=True,
        text=True,
    )
    return parse_trip_counts(p.stderr) if p.returncode == 0 else {}


def analyze(ext_fn, target: str, workdir: Path) -> tuple[StaticReport | None, str]:
    """Compile one kernel and parse its records; ``(None, reason)`` when it fails to compile."""
    cmd, yaml_out = compile_command(ext_fn, target, workdir)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"compile failed: {compile_failure(p.stderr)}"
    rep = parse_yaml(yaml_out) if yaml_out.exists() else StaticReport()
    for key, trips in trip_counts(ext_fn, target, workdir).items():
        if key in rep.loops:
            rep.loops[key].trips = trips
    parse_stderr(p.stderr, rep)
    reached = linked(workdir / f"{ext_fn.name}.o", entry_symbol(ext_fn))
    rep.shipped, rep.libcalls = reached.functions, reached.undefined
    rep.kernel_stack_bytes = reached.stack
    return rep, "ok"


def kernel_builds():
    """Yield ``(name, ExternalFunction)`` for every factory build the library offers.

    Each exported factory at its defaults, plus one build per entry of its
    ``.dtypes`` table; a factory that refuses the current device
    (``NotImplementedError``) is skipped. Remarks depend on source and flags,
    not on the shape a test runs, so this is the whole surface.
    """
    from aie.iron import kernels

    for name in kernels.factories():
        f = getattr(kernels, name)
        combos = [{}] + [dict(c) for c in getattr(f, "dtypes", ()) if c]
        seen: set[str] = set()
        for combo in combos:
            try:
                ef = f(**combo)
            except NotImplementedError:
                continue  # exists only for the other architecture
            if ef.object_file_name in seen:
                continue  # the default build is one of the dtypes entries
            seen.add(ef.object_file_name)
            yield _build_name(name, combo), ef


def _build_name(factory: str, kwargs: dict) -> str:
    from aie.utils.bfp import dtype_name

    def text(v):
        if isinstance(v, enum.Enum):
            return v.value
        return dtype_name(v) if isinstance(v, type) else v

    return factory + "".join(f"/{k}={text(v)}" for k, v in sorted(kwargs.items()))


def _parameter_value(param: inspect.Parameter, text: str):
    import numpy as np
    from aie.utils import bfp

    kind = type(param.default)
    if param.default is None or param.default is param.empty:
        kind = next(
            (t for t in get_args(param.annotation) if t is not type(None)),
            param.annotation,
        )
    if isinstance(param.default, type) or kind is type:
        return bfp.v8bfp16ebs8 if text == "bfp16ebs8" else np.dtype(text).type
    if kind is bool:
        if text not in ("True", "False"):
            raise ValueError(f"{param.name} takes True or False, not {text!r}")
        return text == "True"
    if isinstance(kind, type) and issubclass(kind, enum.Enum):
        return kind(text)
    if kind in (int, float, str):
        return kind(text)
    raise ValueError(f"{param.name} is not settable from the command line")


def parse_build(spec: str) -> tuple[str, dict]:
    """``FACTORY:KEY=VALUE,...`` -> ``(factory, kwargs)``, each value parsed as its parameter's type.

    A dtype parameter takes a numpy name (``int16``, ``bfloat16``) or
    ``bfp16ebs8``, a bool ``True`` or ``False``, an enum a member's value.
    """
    from aie.iron import kernels

    factory, _, rest = spec.partition(":")
    if factory not in kernels.factories():
        raise ValueError(f"{spec}: {factory!r} is not a kernel factory")
    params = inspect.signature(getattr(kernels, factory)).parameters
    kwargs = {}
    for item in filter(None, rest.split(",")):
        key, eq, text = item.partition("=")
        if not eq or key not in params:
            raise ValueError(
                f"{spec}: {item!r} is not KEY=VALUE for a parameter of "
                f"{factory} ({', '.join(params)})"
            )
        try:
            kwargs[key] = _parameter_value(params[key], text)
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{spec}: {item!r}: {e}") from None
    return factory, kwargs


def spec_builds(specs):
    """Yield ``(name, ExternalFunction)`` for each ``parse_build`` result, named as :func:`kernel_builds` names a dtypes entry."""
    from aie.iron import kernels

    for factory, kwargs in specs:
        name = _build_name(factory, kwargs)
        try:
            ef = getattr(kernels, factory)(**kwargs)
        except (NotImplementedError, ValueError) as e:
            raise ValueError(f"--build {name}: {e}") from None
        yield name, ef


def case_builds(
    path: str,
    device: str,
    only: str | None = None,
    coverage: collections.Counter | None = None,
):
    """Yield ``(name, ExternalFunction)`` for every build the cases in ``path`` run.

    ``path`` is a Python file defining ``CASES`` (``kernel_cases.py``); each
    case needs a ``name``, a ``fn()`` that returns its kernel, and optionally
    the ``devices`` it runs on. The file is read at run time, so the test tree
    stays out of this package's imports. A case's shape and options become
    ``-D`` flags, so most cases build an object no factory default does; the
    rows carry the case's name, the key its device series use too. Of the
    cases ``only`` matches, those that share an object compile once, under
    the first name. ``coverage`` counts where every case went, for
    :func:`case_coverage`.
    """
    import dataclasses
    import importlib.util

    coverage = collections.Counter() if coverage is None else coverage
    directory = str(Path(path).resolve().parent)
    sys.path.insert(0, directory)  # a cases file imports its sibling modules
    try:
        spec = importlib.util.spec_from_file_location("_remarks_cases", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(directory)
    seen: set[str] = set()
    for case in module.CASES:
        if only and not re.search(only, case.name):
            coverage["only"] += 1
            continue
        devices = getattr(case, "devices", ())
        if devices and device not in devices:
            coverage["devices"] += 1
            continue
        if len(devices) > 1:
            # fn() binds the case's first device, which need not be this one.
            case = dataclasses.replace(case, devices=(device,))
        try:
            ef = case.fn()
        except NotImplementedError:
            coverage["arch"] += 1  # exists only for the other architecture
            continue
        if ef.object_file_name in seen:
            coverage["shared"] += 1
        else:
            coverage["builds"] += 1
            seen.add(ef.object_file_name)
            yield case.name, ef


def case_coverage(c: collections.Counter) -> str:
    """One line accounting for every case :func:`case_builds` was given."""
    line = (
        f"cases: {c['builds']} builds for {c['builds'] + c['shared']} cases "
        f"({c['shared']} share an earlier case's build); "
        f"skipped {c['devices']} by devices, {c['arch']} other-architecture only"
    )
    return line + (f", {c['only']} not matching --only" if c["only"] else "")


def _selected_builds(only: str | None, builds=kernel_builds) -> list:
    return [
        (name, ef) for name, ef in builds() if not (only and not re.search(only, name))
    ]


def _analyze_builds(builds, target: str, workdir: Path, jobs: int) -> list:
    def compile_one(indexed):
        index, (_, ef) = indexed
        # A directory per build: the outputs are named after the kernel symbol,
        # and nothing guarantees two builds of one factory do not share it.
        cell = workdir / f"build{index}"
        cell.mkdir(parents=True, exist_ok=True)
        return analyze(ef, target, cell)

    # Each build is an independent Peano subprocess, so these fan out; the
    # results are consumed in list order and the records do not depend on
    # how many ran at once.
    with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as pool:
        return list(pool.map(compile_one, enumerate(builds)))


def current_kernel_sources(baseline: str) -> tuple[str, str | None]:
    """Return the ``aie_kernels/`` "this tree" compiles, and a warning if suspect.

    Without ``MLIR_AIE_KERNEL_SOURCES`` the factories compile the installed
    copy, which is the checkout as of its last build or install: a
    before/after of an uncommitted edit then compares the baseline against
    the old kernels and reads as no change.
    """
    from aie.utils import config

    current = config.aie_kernels_dir()
    if not os.environ.get("MLIR_AIE_KERNEL_SOURCES"):
        return current, (
            f"MLIR_AIE_KERNEL_SOURCES is unset, so this tree's kernels are the "
            f"installed copy {current}, as of the last build or install; set it "
            "to the checkout under test"
        )
    if os.path.realpath(current) == os.path.realpath(
        os.path.join(baseline, "aie_kernels")
    ):
        return current, (
            f"this tree and the baseline are both {baseline}; set "
            "MLIR_AIE_KERNEL_SOURCES to the checkout under test"
        )
    return current, None


@contextlib.contextmanager
def kernel_sources(tree: str):
    """Take the factories' kernels from the checkout ``tree`` while inside."""
    from aie.iron import ExternalFunction

    saved = os.environ.get("MLIR_AIE_KERNEL_SOURCES")
    # Factories resolve their source when called, so the builds are taken
    # again under ``tree`` -- after forgetting the ones made before, which
    # share their object names with a different source.
    os.environ["MLIR_AIE_KERNEL_SOURCES"] = tree
    ExternalFunction._instances.clear()
    try:
        yield
    finally:
        if saved is None:
            os.environ.pop("MLIR_AIE_KERNEL_SOURCES")
        else:
            os.environ["MLIR_AIE_KERNEL_SOURCES"] = saved


def _baseline(
    tree: str,
    only,
    target: str,
    workdir: Path,
    jobs: int,
    rows,
    locations=None,
    builds=kernel_builds,
    skip=frozenset(),
) -> dict:
    """Every row whose value differs when the kernels come from ``tree``.

    The builds named in ``skip`` (this tree's failures) are not compiled, and
    the builds the baseline fails are listed under ``"failed"`` with their
    rows left out of the comparison: either side missing is no change to show.
    ``locations`` is this tree's per-loop ``(file, line)``, from ``_run``.
    """
    with kernel_sources(tree):
        selected = [b for b in _selected_builds(only, builds) if b[0] not in skip]
        analyzed = _analyze_builds(selected, target, workdir / "baseline", jobs)
    failed = {
        name: detail
        for (name, _), (rep, detail) in zip(selected, analyzed)
        if rep is None
    }
    base = {
        r["name"]: r["value"]
        for (name, _), (rep, _) in zip(selected, analyzed)
        if rep is not None
        for r in report_rows(rep, name, "")
    }
    base_locations = {}
    for (name, _), (rep, _) in zip(selected, analyzed):
        if rep is not None:
            base_locations.update(_loop_locations(name, rep))
    prefixes = tuple(f"{name}/" for name in failed)
    current = {
        r["name"]: r["value"] for r in rows if not r["name"].startswith(prefixes)
    }
    return {
        **diff_rows(base, current, base_locations, locations or {}),
        "failed": failed,
    }


_LOOP_ROW = re.compile(r"^(.*/loop/[^/]+)/([^/]+)/([^/]+)$")


def _loops(rows: dict) -> dict:
    """``{(prefix/loop/fn, bb): {metric: value}}`` for the loop rows."""
    loops = collections.defaultdict(dict)
    for name, value in rows.items():
        if m := _LOOP_ROW.match(name):
            loops[m[1], m[2]][m[3]] = value
    return loops


def _loop_locations(
    prefix: str, rep: StaticReport
) -> dict[tuple[str, str], tuple[str | None, int | None]]:
    """``{(prefix/loop/fn, bb): (file, line)}``, keyed like ``_loops()``.

    Fed to ``diff_rows`` so its rename heuristic can tell two loops with the
    same metrics apart by where they live, not just by the metrics.
    """
    return {
        (f"{prefix}/loop/{fn}", bb): (loop.file, loop.line)
        for (fn, bb), loop in rep.loops.items()
    }


def diff_rows(
    base: dict,
    current: dict,
    base_locations: dict[tuple[str, str], tuple[str | None, int | None]] | None = None,
    current_locations: (
        dict[tuple[str, str], tuple[str | None, int | None]] | None
    ) = None,
) -> dict:
    """Map each row that differs to ``[base, current]``, less renamed loops.

    LLVM numbers its blocks per function, so an edit anywhere in a function
    can rename every loop after it (``for.body20.i`` -> ``for.body23.i``)
    without changing one. Within a function, a changed loop whose rows
    match another changed loop's in the other tree is taken to be that loop
    renamed -- provided their source locations agree too, when both sides'
    are known: several loops sharing one metric tuple by coincidence (the
    same trip count and II turning up twice in a function) is common enough
    that the tuple alone is not a safe identity, and pairing the wrong two
    would misreport a loop that actually regressed as merely renamed.
    ``"renamed"`` pairs them, and their rows are left out; a metric match
    whose locations disagree falls through to a disappeared loop and an
    appeared one, same as no match at all.
    """
    base_locations = base_locations or {}
    current_locations = current_locations or {}
    changed = {
        name: [base.get(name), current.get(name)]
        for name in sorted(base.keys() | current.keys())
        if base.get(name) != current.get(name)
    }
    before, after = _loops(base), _loops(current)
    moved = {m.group(1, 2) for n in changed if (m := _LOOP_ROW.match(n))}
    unmatched = collections.defaultdict(list)
    for key in sorted(moved & before.keys()):
        bucket = (
            key[0],
            tuple(sorted(before[key].items())),
            base_locations.get(key, (None, None)),
        )
        unmatched[bucket].append(key[1])
    renamed, matched_before, matched_after = [], set(), set()
    for key in sorted(moved & after.keys()):
        bucket = (
            key[0],
            tuple(sorted(after[key].items())),
            current_locations.get(key, (None, None)),
        )
        olds = unmatched[bucket]
        if olds:
            old = olds.pop(0)
            renamed.append([f"{key[0]}/{old}", f"{key[0]}/{key[1]}"])
            matched_before.add((key[0], old))
            matched_after.add(key)

    def settled(key):
        return (key not in before or key in matched_before) and (
            key not in after or key in matched_after
        )

    changed = {
        n: v
        for n, v in changed.items()
        if not ((m := _LOOP_ROW.match(n)) and settled(m.group(1, 2)))
    }
    return {"rows": changed, "renamed": renamed}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m aie.utils.compile.remarks",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--out", required=True, help="benchmark-action rows (integer series)"
    )
    ap.add_argument(
        "--out-pm",
        metavar="JSON",
        help="write the pm_bytes rows here instead of --out, so program memory can "
        "carry a percentage threshold while II and the loop counts keep any-increase",
    )
    ap.add_argument("--meta", help="per-kernel loops, notes and warnings, as JSON")
    ap.add_argument("--target", default="aie2p", choices=["aie2", "aie2p"])
    ap.add_argument("--only", help="regex on kernel names")
    ap.add_argument(
        "--cases",
        metavar="FILE",
        help="compile the builds the CASES in FILE run (test/python/npu/"
        "kernel_cases.py), named by case, instead of each factory's defaults",
    )
    ap.add_argument(
        "--build",
        action="append",
        metavar="FACTORY:KEY=VALUE,...",
        help="compile this factory build instead of each factory's defaults "
        "(repeatable), e.g. cascade_mm:dim_m=16,dim_k=24,dim_n=32 or "
        "scale:dtype=bfloat16; a shape can pick another code path",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=os.cpu_count() or 1,
        help="concurrent Peano compiles; the records do not depend on it",
    )
    ap.add_argument(
        "--annotate",
        action="store_true",
        default=os.environ.get("GITHUB_ACTIONS") == "true",
        help="print GitHub workflow commands (::warning / ::error) for dropped "
        "pragmas and compile failures; on by default under Actions",
    )
    ap.add_argument(
        "--keep",
        metavar="DIR",
        help="compile into DIR (one build<i> per kernel build, named in --meta) "
        "instead of a temporary directory",
    )
    ap.add_argument(
        "--baseline-sources",
        metavar="DIR",
        help="also compile every build with its kernels from DIR (a checkout "
        "root, as MLIR_AIE_KERNEL_SOURCES) and print each row that differs; "
        "the rows written stay this tree's",
    )
    ap.add_argument(
        "--sources",
        metavar="DIR",
        help="take this tree's kernels from DIR (a checkout root), as "
        "MLIR_AIE_KERNEL_SOURCES=DIR does; without either they are the "
        "installed copy",
    )
    a = ap.parse_args(argv)
    if a.build and a.cases:
        ap.error("--build and --cases each pick the builds; give one")
    try:
        a.build = [parse_build(spec) for spec in a.build or ()]
    except ValueError as e:
        ap.error(str(e))
    if not a.sources:
        return _run(a)
    with kernel_sources(a.sources):
        return _run(a)


def _run(a: argparse.Namespace) -> int:
    from aie.iron.device import from_name
    from aie.utils.benchmark import provenance
    from aie.utils.hostruntime import set_current_device

    # Factories pick their source and mac_dims through the current device.
    generation = "npu1" if a.target == "aie2" else "npu2"
    device = from_name(generation, n_cols=1)
    set_current_device(device)
    extra = provenance(target=a.target)
    workdir = Path(a.keep or tempfile.mkdtemp(prefix="aie-static-"))
    print(f"compiling into {workdir}")
    source_root = os.environ.get("MLIR_AIE_KERNEL_SOURCES")
    if a.baseline_sources:
        current_sources, suspect = current_kernel_sources(a.baseline_sources)
        if suspect:
            print(f"warning: {suspect}", file=sys.stderr)
    rows: list[dict] = []
    locations: dict = {}
    failed: dict[str, str] = {}
    meta: dict = {"kernels": {}}
    annotated: set[tuple] = set()

    sweep = kernel_builds
    coverage: collections.Counter = collections.Counter()
    if a.build:

        def sweep():
            return spec_builds(a.build)

    if a.cases:

        def sweep():
            coverage.clear()
            return case_builds(a.cases, generation, a.only, coverage)

    try:
        builds = _selected_builds(a.only, sweep)
    except ValueError as e:  # a --build its factory refuses
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.cases:
        print(case_coverage(coverage))
    analyzed = _analyze_builds(builds, a.target, workdir, a.jobs)

    for index, ((name, ef), (rep, detail)) in enumerate(zip(builds, analyzed)):
        source = ef.source_file or f"<inline {ef.name}.cc>"
        budget = (
            ef.contract and ef.contract.stack_bytes
        ) or device.default_core_stack_bytes
        if rep is None:
            failed[name] = detail
        else:
            rows += report_rows(rep, name, extra)
            locations.update(_loop_locations(name, rep))
            meta["kernels"][name] = {
                "source": source,
                "symbol": entry_symbol(ef),
                "object": str(workdir / f"build{index}" / f"{ef.name}.o"),
                "loops": {
                    f"{fn}/{bb}": vars(loop) for (fn, bb), loop in rep.loops.items()
                },
                "missing_bank_loads": rep.missing_bank_loads,
                "pass_failed_warnings": rep.pass_failed_warnings,
                "pass_failed": rep.pass_failed,
                "pm_bytes": rep.pm_bytes,
                "pm_bytes_by_function": {
                    fn: n for fn, n in rep.pm_bytes_by_function.items() if rep.ships(fn)
                },
                "libcalls": rep.libcalls,
                "kernel_stack_bytes": rep.kernel_stack_bytes,
                "stack_budget": budget,
                "schedule_notes": rep.schedule_notes,
            }
        print(
            f"[{'OK' if rep else 'FAIL'}] {name}: {entry_symbol(ef)} from {source}"
            + (f" {detail}" if rep is None else "")
        )
        if rep and rep.pass_failed:
            print("\n".join(f"  dropped pragma: {w}" for w in rep.pass_failed))
        if rep and rep.libcalls:
            print(f"  calls the runtime library: {' '.join(rep.libcalls)}")
        if (
            rep
            and rep.kernel_stack_bytes is not None
            and rep.kernel_stack_bytes > budget
        ):
            print(
                f"  stack: the kernel alone takes {rep.kernel_stack_bytes} bytes, "
                f"over the {budget} the design reserves for the whole core; "
                "the core overwrites its neighbours silently"
            )
        if a.annotate:
            print(
                "\n".join(
                    workflow_annotations(
                        f"{name} ({a.target})", rep, detail, source_root, annotated
                    )
                )
            )

    meta["failed"] = failed
    if a.baseline_sources:
        changed = _baseline(
            a.baseline_sources,
            a.only,
            a.target,
            workdir,
            a.jobs,
            rows,
            locations,
            sweep,
            failed,
        )
        meta["baseline"] = {
            "sources": a.baseline_sources,
            "current_sources": current_sources,
            "warning": suspect,
            "changed": changed["rows"],
            "renamed": changed["renamed"],
            "failed": changed["failed"],
        }
        if suspect:
            print(f"warning: {suspect}")
        for name, detail in changed["failed"].items():
            print(f"  baseline fails to compile {name}, not compared: {detail}")
        print(
            f"baseline {a.baseline_sources} -> this tree ({current_sources}): "
            f"{len(changed['rows'])} rows differ"
            + (
                f"; {len(changed['renamed'])} loops renamed with the same rows, "
                "not listed"
                if changed["renamed"]
                else ""
            )
        )
        for name, (before, after) in changed["rows"].items():
            print(f"  {name}: {before} -> {after}")
    if a.meta:
        Path(a.meta).write_text(json.dumps(meta, indent=1, default=str))
    # The builds that compiled keep their rows even when another fails.
    if a.out_pm:
        pm_rows = [r for r in rows if r["name"].endswith("/pm_bytes")]
        rows = [r for r in rows if not r["name"].endswith("/pm_bytes")]
        if pm_rows or not failed:
            _write_rows(a.out_pm, pm_rows)
            print(f"wrote {len(pm_rows)} pm_bytes rows to {a.out_pm}")
    if rows or not failed:
        _write_rows(a.out, rows)
        print(f"wrote {len(rows)} rows to {a.out}")
    if failed:  # every kernel must compile
        print(
            "RESULTS INVALID: "
            + "; ".join(f"{name}: {detail}" for name, detail in failed.items()),
            file=sys.stderr,
        )
        return 3
    return 0


def _write_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty benchmark file")
    Path(path).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
