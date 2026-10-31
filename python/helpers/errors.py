# errors.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Report design failures against the user's source rather than IRON's.

Set `IRON_FULL_TRACEBACK=1` to keep IRON's own frames, and the dump of an op
an error names, when debugging IRON.
"""

import contextlib
import linecache
import os
import re
import types

from ..ir import (  # pyright: ignore[reportMissingImports]
    CallSiteLoc,
    FileLineColLoc,
    FusedLoc,
    Location,
    MLIRError,
    NameLoc,
)
from .sourceloc import IRON_PREFIX, is_internal_file

_TOOL_ERROR = re.compile(r"^(.+?):(\d+):\d+: error: ", re.M)


def _attach_source_frame(
    exc: BaseException, filename: str, lineno: int, name: str = "<design>"
) -> bool:
    """Append a frame at `filename:lineno` as the innermost entry of `exc`.

    Args:
        exc: The exception whose traceback grows.
        filename: Source file of the failing statement.
        lineno: Its line, which Python quotes when printing the traceback.
        name: Function name shown for the frame.

    Returns:
        Whether the frame was attached; it is not when the line can't be read.
    """
    real = linecache.getline(filename, lineno).strip()
    if not real:
        return False
    # Python omits carets when the statement spans the whole line, and MLIR
    # gives no end column to underline up to.
    pad = max(0, (len(real) - len("raise __hop__")) // 2)
    source = "\n" * (lineno - 1) + "raise " + "(" * pad + "__hop__" + ")" * pad
    code = compile(source, filename, "exec").replace(co_name=name)
    hop = Exception()
    with contextlib.suppress(Exception):
        exec(code, {"__hop__": hop})
    assert hop.__traceback__ is not None and hop.__traceback__.tb_next is not None
    frame = hop.__traceback__.tb_next

    entries, tb = [], exc.__traceback__
    while tb is not None:
        entries.append(tb)
        tb = tb.tb_next
    rebuilt = types.TracebackType(None, frame.tb_frame, frame.tb_lasti, lineno)
    for entry in reversed(entries):
        rebuilt = types.TracebackType(
            rebuilt, entry.tb_frame, entry.tb_lasti, entry.tb_lineno
        )
    exc.with_traceback(rebuilt)
    return True


def design_error(exc: BaseException) -> BaseException:
    """Rewrite `exc`, raised while building a design, to point at the design.

    The `aie` package's frames are dropped, and an `MLIRError` gains a frame at
    the location of its first error diagnostic. Its message prints each
    diagnostic as aiecc does: `design.py:12:4: error: act_in: ...`, then the
    calls that led there, without the dump of an op an IRON object names.

    Args:
        exc: The exception to rewrite in place.

    Returns:
        `exc`.
    """
    full = os.environ.get("IRON_FULL_TRACEBACK", "") not in ("", "0")
    if not full:
        kept, tb = [], exc.__traceback__
        while tb is not None:
            if not is_internal_file(tb.tb_frame.f_code.co_filename):
                kept.append(tb)
            tb = tb.tb_next
        rebuilt = None
        for entry in reversed(kept):
            rebuilt = types.TracebackType(
                rebuilt, entry.tb_frame, entry.tb_lasti, entry.tb_lineno
            )
        exc.with_traceback(rebuilt)
    if not isinstance(exc, MLIRError):
        return exc
    lines, attached = [f"{exc.message}:"], False  # pyright: ignore[reportAttributeAccessIssue]  # fmt: skip
    for diag in exc.error_diagnostics:  # pyright: ignore[reportAttributeAccessIssue]  # fmt: skip
        site = _design_site(diag.location)
        objects, callers = [], []
        if site is None:
            # Unwrapped as MLIRError prints it: `unknown`, `"name"`.
            where = str(diag.location).removeprefix("loc(").removesuffix(")")
        else:
            filename, line, col, callers, names = site
            objects = [
                n.removeprefix(IRON_PREFIX) for n in names if n.startswith(IRON_PREFIX)
            ]
            if not attached:
                # "iron:act_in" or "py:build.<locals>.core_fn": the frame reads
                # as the object or function, without its kind.
                short = [(n.partition(":")[2] or n).rpartition(".")[2] for n in names]
                frame_name = ", ".join(dict.fromkeys(short)) or "<design>"
                attached = _attach_source_frame(exc, filename, line, frame_name)
            where = f"{filename}:{line}:{col}"
        named = f"{', '.join(objects)}: " if objects else ""
        # Continuation lines, such as an op's dump, indent under their line.
        message = diag.message.replace("\n", "\n  ")
        lines.append(f"{where}: error: {named}{message}")
        lines += [f"{f}:{n}:{c}: note: called from" for f, n, c in callers]
        for note in diag.notes:
            # The location already points at the statement that declared the
            # object; IRON_FULL_TRACEBACK keeps the dump for debugging IRON.
            if (
                objects
                and not full
                and note.message.startswith("see current operation")
            ):
                continue
            site = _design_site(note.location)
            where = (
                f"{site[0]}:{site[1]}:{site[2]}"
                if site
                else str(note.location).removeprefix("loc(").removesuffix(")")
            )
            message = note.message.replace("\n", "\n  ")
            lines.append(f"{where}: note: {message}")
    exc.args = ("\n".join(lines),)
    return exc


def _design_site(
    loc: Location,
) -> tuple[str, int, int, list[tuple[str, int, int]], list[str]] | None:
    """Return where a diagnostic's location points in the design.

    Args:
        loc: The location, as IRON builds them: names around a file location,
            its callers, or several of those fused.

    Returns:
        (filename, line, col, callers, names): the first file location, a
        call site's callee; the (filename, line, col) of each call that led
        to it, innermost first; and every name on the whole location, as
        "iron:act_in" or "py:core_fn". None when there is no file location.
    """
    names, site = [], None
    # Each location still to visit, with the callers of the call sites
    # around it, innermost first. A diagnostic's location comes as a plain
    # Location; from_attr gives its kind, and its parts come typed.
    pending = [(Location.from_attr(loc.attr, loc.context), [])]
    while pending:
        loc, callers = pending.pop(0)
        if isinstance(loc, NameLoc):
            if loc.name_str not in names:
                names.append(loc.name_str)
            pending.insert(0, (loc.child_loc, callers))
        elif isinstance(loc, CallSiteLoc):
            pending.insert(0, (loc.callee, [loc.caller] + callers))
        elif isinstance(loc, FusedLoc):
            pending[0:0] = [(child, callers) for child in loc.locations]
        elif isinstance(loc, FileLineColLoc) and site is None:
            site = (loc.filename, loc.start_line, loc.start_col, callers)
    if site is None:
        return None
    filename, line, col, callers = site
    # A caller is a file location, a chain of call sites, or either of those
    # named or fused.
    caller_sites = []
    for caller in callers:
        stack = [caller]
        while stack:
            loc = stack.pop()
            if isinstance(loc, NameLoc):
                stack.append(loc.child_loc)
            elif isinstance(loc, CallSiteLoc):
                stack += [loc.caller, loc.callee]
            elif isinstance(loc, FusedLoc):
                stack += reversed(list(loc.locations))
            elif isinstance(loc, FileLineColLoc):
                caller_sites.append((loc.filename, loc.start_line, loc.start_col))
    return filename, line, col, caller_sites, names


def attach_tool_location(exc: BaseException, output: str) -> bool:
    """Point `exc` at the first located error a tool such as aiecc printed.

    Args:
        exc: The exception to raise for the tool's failure.
        output: The tool's output, with errors as `file:line:col: error: ...`.

    Returns:
        Whether a frame was attached.
    """
    match = _TOOL_ERROR.search(output)
    if match is None:
        return False
    return _attach_source_frame(exc, match.group(1), int(match.group(2)))
