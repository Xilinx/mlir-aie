# readobj.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""What a linked kernel keeps, read from its object with ``llvm-readobj``.

Peano gives every function its own section (``.text.<fn>``) and the core
link drops unreferenced sections, so what ships is what the entry symbol
reaches through relocations. A helper that is inlined at its call sites and
also emitted standalone has asm-printer records for both; counting only the
reached functions keeps it from counting twice. A relocation from a reached
section to an undefined symbol is a call into the runtime library -- on
AIE2P that is scalar float divide, int-to-float and the like, each a
software routine of hundreds of cycles.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from aie.utils import config


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
