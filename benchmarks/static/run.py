#!/usr/bin/env python3
# benchmarks/static/run.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""CPU-only static analysis of the kernel library via Peano remarks.

    python -m benchmarks.static.run --out static.json --meta meta.json

For every light kernel entry in the registry, compile its ExternalFunction
source with the library's own flags plus the remark flags, parse the YAML
optimization record and the stderr warnings channel, and emit rows:
  <kernel>/<case>/unpipelined_loops, non_zol_loops, missing_bank_loads,
  pass_failed_warnings, pm_bytes, loop/<fn>/<bb>/II

Needs only the llvm-aie wheel and the mlir_aie wheel (for aie_kernels
sources and headers). Runs on any x86 builder in minutes. A kernel that
fails to *compile* is a hard failure (exit 3, no JSON written).

The wheel carries its own copy of ``aie_kernels/`` and ``aie_runtime_lib/``
from the commit it was built at. ``--source-root <checkout>`` compiles the
checkout's copies instead, which is what makes a run on a pull request
speak to that pull request's kernel sources.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from aie.iron.device import NPU1Col1, NPU2Col1
from aie.utils.compile.utils import cxx_core_compile_command
from aie.utils.hostruntime import set_current_device

from ..kernels import registry
from ..kernels._util import provenance, write_rows
from . import remarks

# The kernel factories pick their source per architecture through the current
# iron device (aie2 vs aie2p LUT sources, mm.cc mac_dims, ...), so a static
# run for a target has to set the matching device before building anything.
_DEVICE_FOR_TARGET = {"aie2": NPU1Col1, "aie2p": NPU2Col1}

# Cheap upstream clang checks the perf guide's alignment section justifies;
# on top of the library's own flags, never instead of them.
_EXTRA_WARNINGS = [
    "-Wcast-align",
    "-Walign-mismatch",
    "-Wunaligned-access",
    "-Wframe-larger-than=1024",
]


# Directories the factories resolve from the installed wheel that also live
# in the repository, as (wheel-relative, checkout-relative) pairs.
_RELOCATABLE = (
    (("include", "aie_kernels"), ("aie_kernels",)),
    (("aie_runtime_lib",), ("aie_runtime_lib",)),
)


def relocate(text: str, wheel_root: str, source_root: str) -> str:
    """Rewrite wheel paths in ``text`` to the checkout under ``source_root``.

    Covers a ``source_file`` path, an include directory and the aie2 LUT
    factories' ``source_string`` (an ``#include`` of the wheel's copy of
    the kernel and of ``aie_runtime_lib/AIE2/lut_based_ops.cpp``). Only
    whole directory prefixes are rewritten, so ``.../aie_kernels_x`` is
    left alone.
    """
    for wheel_parts, src_parts in _RELOCATABLE:
        old = str(Path(wheel_root, *wheel_parts)) + os.sep
        new = str(Path(source_root, *src_parts)) + os.sep
        text = text.replace(old, new)
    return text


def compile_command(
    ext_fn, target: str, out_dir: Path, source_root: str | None = None
) -> tuple[list[str], Path]:
    """Build the exact Peano command the JIT would run for ``ext_fn``, plus remarks.

    Uses ``aie.utils.compile.utils.cxx_core_compile_command`` so the target
    triple, warning set, defines and section flags cannot drift from what the
    library compiles with; only the optimization-record and diagnostic flags
    are added. Inline-source kernels (the aie2 LUT activations) are written
    out under the kernel's symbol name first, as the JIT does. With
    ``source_root`` the kernel sources are taken from that checkout instead
    of the wheel (see :func:`relocate`).
    """
    if ext_fn.use_chess:
        raise ValueError(
            f"{ext_fn.name}: built with xchesscc; Peano remarks do not apply"
        )

    def fix(text: str) -> str:
        if source_root is None:
            return text
        from aie.utils import config

        return relocate(text, config.root_path(), os.path.abspath(source_root))

    include_dirs = [fix(d) for d in ext_fn.include_dirs]
    if ext_fn.source_file is not None:
        src = Path(fix(ext_fn.source_file))
        # Same as the JIT: the source's own directory, so relative includes
        # such as "../aie_kernel_utils.h" resolve.
        if str(src.parent) not in include_dirs:
            include_dirs.append(str(src.parent))
    else:
        src = out_dir / f"{ext_fn.name}.cc"
        src.write_text(fix(ext_fn.source_string))
    yaml_out = out_dir / f"{ext_fn.name}.opt.yaml"
    cmd = cxx_core_compile_command(
        str(src),
        target,
        str(out_dir / f"{ext_fn.name}.o"),
        include_dirs=include_dirs,
        compile_args=[
            *ext_fn.compile_flags,
            *remarks.REMARK_FLAGS,
            f"-foptimization-record-file={yaml_out}",
            *_EXTRA_WARNINGS,
        ],
    )
    return cmd, yaml_out


def analyze(
    case, target: str, workdir: Path, source_root: str | None = None
) -> tuple[remarks.StaticReport | None, str]:
    ext_fn = case.fn()
    cmd, yaml_out = compile_command(ext_fn, target, workdir, source_root)
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        return None, f"compile failed: {p.stderr.strip()[-2000:]}"
    rep = remarks.parse_yaml(yaml_out) if yaml_out.exists() else remarks.StaticReport()
    remarks.parse_stderr(p.stderr, rep)
    return rep, "ok"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--out-pm",
        metavar="JSON",
        help="write the pm_bytes rows here instead of --out, so program memory "
        "can be tracked under a percentage threshold while II and the loop "
        "counts keep the any-increase rule",
    )
    ap.add_argument("--meta")
    ap.add_argument("--target", default="aie2p", choices=["aie2", "aie2p"])
    ap.add_argument("--only")
    ap.add_argument(
        "--source-root",
        metavar="CHECKOUT",
        help="compile aie_kernels/ and aie_runtime_lib/ from this checkout "
        "instead of the installed wheel's copies",
    )
    ap.add_argument(
        "--annotate",
        action="store_true",
        default=os.environ.get("GITHUB_ACTIONS") == "true",
        help="print GitHub workflow commands (::warning / ::error) for dropped "
        "pragmas and compile failures; on by default under Actions",
    )
    a = ap.parse_args(argv)
    if a.source_root and not Path(a.source_root, "aie_kernels").is_dir():
        ap.error(f"--source-root {a.source_root}: no aie_kernels/ directory")

    extra = provenance() + f" | target {a.target}"
    workdir = Path(tempfile.mkdtemp(prefix="aie-static-"))
    rows: list[dict] = []
    failed: list[str] = []
    meta = {"kernels": {}}

    set_current_device(_DEVICE_FOR_TARGET[a.target]())
    seen: set[str] = set()
    for case in registry.perf_cases(a.only):
        # One representative case per factory is enough: remarks depend on the
        # compiled source and flags, not on the runtime shape.
        key = f"{case.factory}{sorted(case.kwargs.items())!r}"
        if key in seen or (case.arch and case.arch != a.target):
            continue
        seen.add(key)
        name = case.name
        rep, detail = analyze(case, a.target, workdir, a.source_root)
        ok = rep is not None
        if not ok:
            failed.append(f"{name}: {detail}")
        if rep:
            rows += remarks.rows(rep, name, extra)
            meta["kernels"][name] = {
                "loops": {
                    f"{fn}/{bb}": vars(loop) for (fn, bb), loop in rep.loops.items()
                },
                "missing_bank_loads": rep.missing_bank_loads,
                "pass_failed_warnings": rep.pass_failed_warnings,
                "pm_bytes": rep.pm_bytes,
                # why the pipeliner declined a loop ("MII too large", "Unable
                # to find schedule"); not a graph series, but the first thing
                # to read when an II or unpipelined_loops row moves.
                "schedule_notes": rep.schedule_notes,
            }
        print(f"[{'OK' if ok else 'FAIL'}] {name} {detail if not ok else ''}")
        if rep and rep.pass_failed:
            print("\n".join(f"  dropped pragma: {w}" for w in rep.pass_failed))
        if a.annotate:
            print("\n".join(remarks.annotations(name, rep, detail, a.source_root)))

    meta["failed"] = failed
    if a.meta:
        Path(a.meta).write_text(json.dumps(meta, indent=1, default=str))
    if failed:  # every kernel must compile
        print("RESULTS INVALID: " + "; ".join(failed), file=sys.stderr)
        return 3
    if a.out_pm:
        pm_rows = [r for r in rows if r["name"].endswith("/pm_bytes")]
        rows = [r for r in rows if not r["name"].endswith("/pm_bytes")]
        write_rows(a.out_pm, pm_rows)
        print(f"wrote {len(pm_rows)} pm_bytes rows to {a.out_pm}")
    write_rows(a.out, rows)
    print(f"wrote {len(rows)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
