# remarks.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Static kernel checks: compile every library kernel with Peano and read its remarks.

    python -m aie.utils.compile.remarks --target aie2p --out static.json \
        --out-pm static-pm.json --meta static-meta.json

CPU-only. Every factory in ``aie.iron.kernels`` (at its defaults and for each
entry of its ``.dtypes`` table) is compiled exactly as the JIT compiles it
(:func:`aie.utils.compile.utils.cxx_core_compile_command`), plus the
optimization-record flags below, and the records become per-kernel series
for benchmark-action: a Peano bump that changes a loop's schedule shows up
here before anyone looks at device numbers. With ``MLIR_AIE_KERNEL_SOURCES``
set to a checkout, that checkout's ``aie_kernels/`` and ``aie_runtime_lib/``
are compiled instead of the installed copies.

Record shapes, as llvm-aie 22.0.0.2026090201 emits them (they are Peano's,
not LLVM's documented ones):

| Pass | Kind / Name | Args | Tracked as |
| --- | --- | --- | --- |
| ``pipeliner`` | ``Passed`` / ``schedule`` | ``II``, ``NS``, ``Loop``, ``Pipeliner``, prologue/epilogue bundles | ``loop/<fn>/<bb>/II`` (rest as hover text) |
| ``pipeliner`` | ``Missed`` / ``canPipelineLoop`` | "Failed to pipeline loop"; located by ``DebugLoc`` only | ``unpipelined_loops`` (keyed ``L<line>``) |
| ``pipeliner`` | ``Analysis`` / ``schedule`` | ``MII``, ``SwpMaxMii``, "Unable to find schedule" | ``schedule_notes`` in the meta file |
| ``aie-hardware-loops`` | ``Analysis`` / ``analysis`` | ``BasicBlock``, ``Zero-Overhead-Loop`` | ``non_zol_loops``, and ``loop/<fn>/<bb>/not_zol`` per loop |
| ``aie-asm-printer`` | ``Analysis`` / ``analysis`` | ``BasicBlock``, ``BundleCount``, ``ByteCount`` | ``pm_bytes`` (summed per function) |
| ``aie-multi-slot-pseudo`` | ``Missed`` / ``missing-memory-bank`` | ``Instruction`` | ``missing_bank_loads`` |
| stderr | ``-Wpass-failed`` | a ``#pragma clang loop`` / ``AIE_*`` macro the compiler dropped | ``pass_failed_warnings``, text kept |

The loop-scheduling pass reports as ``pipeliner`` (a ``postpipeliner``
filter records nothing); it names loops by machine basic block
(``bb.1.for.body.i``) while the other passes use the IR block
(``for.body.i``), so the prefix is stripped to join them.
``unpipelined_loops`` counts every loop the pipeliner declined, outer loops
included, so its change is the signal, not its value.

Regression rules: the integer series (``II``, ``not_zol``, ``unpipelined_loops``,
``non_zol_loops``, ``missing_bank_loads``, ``pass_failed_warnings``) alert on
any increase, so a loop that falls off the zero-overhead loop unit is an
alert on its own row, not a change hidden in an II hover text; ``pm_bytes`` is written to its own file (``--out-pm``) so it
can carry a percentage threshold, since a toolchain bump routinely moves
program memory by a few bytes. Nothing gates: under GitHub Actions a dropped
pragma becomes a ``::warning`` on the file and line it names and a kernel
that fails to compile a ``::error``; the run exits 3 on a compile failure
and writes nothing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

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
    """Reassemble the remark sentence as clang prints it: every Arg value, in order.

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


def _row(name: str, unit: str, value, extra: str, rng: str | None = None) -> dict:
    r = {"name": name, "unit": unit, "value": value, "extra": extra}
    if rng:
        r["range"] = rng
    return r


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
                _row(
                    f"{prefix}/loop/{fn}/{bb}/II",
                    "cycles",
                    loop.ii,
                    extra,
                    f"NS={loop.ns} pro={loop.prologue_bundles} "
                    f"epi={loop.epilogue_bundles} zol={loop.zol} "
                    f"via={loop.pipeliner}{where}",
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

_NOT_FACTORIES = {"mm_stream_dims", "mm_acc_dtype"}


def compile_command(ext_fn, target: str, out_dir: Path) -> tuple[list[str], Path]:
    """Return the exact Peano command the JIT would run for ``ext_fn``, plus remark flags.

    Inline-source kernels (the aie2 LUT activations) are written out under the
    kernel's symbol name first, as the JIT does. Kernels built with
    ``use_chess`` are rejected: the remarks are Peano's.
    """
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
        ],
    )
    return cmd, yaml_out


def analyze(ext_fn, target: str, workdir: Path) -> tuple[StaticReport | None, str]:
    """Compile one kernel and parse its records; ``(None, reason)`` when it fails to compile."""
    cmd, yaml_out = compile_command(ext_fn, target, workdir)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"compile failed: {p.stderr.strip()[-2000:]}"
    rep = parse_yaml(yaml_out) if yaml_out.exists() else StaticReport()
    parse_stderr(p.stderr, rep)
    return rep, "ok"


def kernel_builds():
    """Yield ``(name, ExternalFunction)`` for every factory build the library offers.

    Each exported factory at its defaults, plus one build per entry of its
    ``.dtypes`` table; a factory that refuses the current device
    (``NotImplementedError``) is skipped. Remarks depend on source and flags,
    not on the shape a test runs, so this is the whole surface.
    """
    import inspect

    from aie.iron import kernels
    from aie.utils.kernel_harness import dtype_name

    for name in kernels.__all__:
        f = getattr(kernels, name)
        if not inspect.isfunction(f) or name.endswith("_ref") or name in _NOT_FACTORIES:
            continue
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
            suffix = "".join(
                f"/{k}={dtype_name(v) if isinstance(v, type) else v}"
                for k, v in sorted(combo.items())
            )
            yield f"{name}{suffix}", ef


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
        "--annotate",
        action="store_true",
        default=os.environ.get("GITHUB_ACTIONS") == "true",
        help="print GitHub workflow commands (::warning / ::error) for dropped "
        "pragmas and compile failures; on by default under Actions",
    )
    a = ap.parse_args(argv)

    from aie.iron.device import from_name
    from aie.utils.benchmark import provenance
    from aie.utils.hostruntime import set_current_device

    # Factories pick their source and mac_dims through the current device.
    set_current_device(from_name("npu1" if a.target == "aie2" else "npu2", n_cols=1))
    extra = provenance(target=a.target)
    workdir = Path(tempfile.mkdtemp(prefix="aie-static-"))
    source_root = os.environ.get("MLIR_AIE_KERNEL_SOURCES")
    rows: list[dict] = []
    failed: list[str] = []
    meta: dict = {"kernels": {}}
    annotated: set[tuple] = set()

    for name, ef in kernel_builds():
        if a.only and not re.search(a.only, name):
            continue
        rep, detail = analyze(ef, a.target, workdir)
        if rep is None:
            failed.append(f"{name}: {detail}")
        else:
            rows += report_rows(rep, name, extra)
            meta["kernels"][name] = {
                "loops": {
                    f"{fn}/{bb}": vars(loop) for (fn, bb), loop in rep.loops.items()
                },
                "missing_bank_loads": rep.missing_bank_loads,
                "pass_failed_warnings": rep.pass_failed_warnings,
                "pass_failed": rep.pass_failed,
                "pm_bytes": rep.pm_bytes,
                "schedule_notes": rep.schedule_notes,
            }
        print(f"[{'OK' if rep else 'FAIL'}] {name} {detail if rep is None else ''}")
        if rep and rep.pass_failed:
            print("\n".join(f"  dropped pragma: {w}" for w in rep.pass_failed))
        if a.annotate:
            print(
                "\n".join(
                    workflow_annotations(
                        f"{name} ({a.target})", rep, detail, source_root, annotated
                    )
                )
            )

    meta["failed"] = failed
    if a.meta:
        Path(a.meta).write_text(json.dumps(meta, indent=1, default=str))
    if failed:  # every kernel must compile
        print("RESULTS INVALID: " + "; ".join(failed), file=sys.stderr)
        return 3
    if a.out_pm:
        pm_rows = [r for r in rows if r["name"].endswith("/pm_bytes")]
        rows = [r for r in rows if not r["name"].endswith("/pm_bytes")]
        _write_rows(a.out_pm, pm_rows)
        print(f"wrote {len(pm_rows)} pm_bytes rows to {a.out_pm}")
    _write_rows(a.out, rows)
    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


def _write_rows(path: str, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty benchmark file")
    Path(path).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    sys.exit(main())
