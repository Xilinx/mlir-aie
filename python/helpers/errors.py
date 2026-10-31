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
from collections.abc import Iterator

from ..ir import (  # pyright: ignore[reportMissingImports]
    CallSiteLoc,
    FileLineColLoc,
    FusedLoc,
    Location,
    MLIRError,
    NameLoc,
    UnknownLoc,
)
from .sourceloc import IRON_PREFIX, is_internal_file

_TOOL_ERROR = re.compile(r"^(.+?):(\d+):\d+: error: ", re.M)
# A design's MLIRError prints as MLIR's SourceMgrDiagnosticHandler prints
# aiecc's diagnostics, so the two read line for line alike.
_CALL_STACK_LIMIT = 10
_OP_DUMP = "see current operation"


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
        # A diagnostic's location comes as a plain Location; from_attr gives
        # its kind, and its parts come typed.
        loc = Location.from_attr(diag.location.attr, diag.location.context)
        names = list(
            dict.fromkeys(n.name_str for n in _walk(loc) if isinstance(n, NameLoc))
        )
        objects = [
            n.removeprefix(IRON_PREFIX) for n in names if n.startswith(IRON_PREFIX)
        ]
        site = next((f for f in _walk(loc) if isinstance(f, FileLineColLoc)), None)
        if site is not None and not attached:
            # "iron:act_in" or "py:build.<locals>.core_fn": the frame reads as
            # the object or function, without its kind.
            short = [(n.partition(":")[2] or n).rpartition(".")[2] for n in names]
            frame_name = ", ".join(dict.fromkeys(short)) or "<design>"
            attached = _attach_source_frame(
                exc, site.filename, site.start_line, frame_name
            )
        named = f"{', '.join(objects)}: " if objects else ""
        lines.append(_diagnostic_line(loc, "error", named + diag.message))
        call = _call_site(loc)
        for _ in range(_CALL_STACK_LIMIT):
            if call is None:
                break
            lines.append(_diagnostic_line(call.caller, "note", "called from"))
            call = _call_site(call.caller)
        for note in diag.notes:
            # The location already points at the statement that declared the
            # object; IRON_FULL_TRACEBACK keeps the dump for debugging IRON.
            if objects and not full and note.message.startswith(_OP_DUMP):
                continue
            note_loc = Location.from_attr(note.location.attr, note.location.context)
            lines.append(_diagnostic_line(note_loc, "note", note.message))
    exc.args = ("\n".join(lines),)
    return exc


def _walk(loc: Location) -> Iterator[Location]:
    """Yield `loc` and every location within it, in MLIR's (pre-)order."""
    stack = [loc]
    while stack:
        loc = stack.pop()
        yield loc
        if isinstance(loc, NameLoc):
            stack.append(loc.child_loc)
        elif isinstance(loc, CallSiteLoc):
            stack += [loc.caller, loc.callee]
        elif isinstance(loc, FusedLoc):
            stack += reversed(list(loc.locations))


def _call_site(loc: Location) -> CallSiteLoc | None:
    """Return the call site `loc` is, under names, or the first fused one."""
    if isinstance(loc, NameLoc):
        return _call_site(loc.child_loc)
    if isinstance(loc, CallSiteLoc):
        return loc
    if isinstance(loc, FusedLoc):
        return next((c for c in map(_call_site, loc.locations) if c is not None), None)
    return None


def _diagnostic_line(loc: Location, severity: str, message: str) -> str:
    """Return a diagnostic's line: at `loc`'s first file location, if any.

    Args:
        loc: Where the diagnostic is, typed.
        severity: "error" or "note".
        message: Its text.

    Returns:
        "design.py:12:4: error: ..."; with no file location, as LLVM's
        SourceMgr prints a message it cannot place, "<unknown>:0: error: ..."
        and, unless the location is unknown, `loc(...)` before the message.
    """
    site = next((f for f in _walk(loc) if isinstance(f, FileLineColLoc)), None)
    if site is not None:
        where = f"{site.filename}:{site.start_line}:{site.start_col}"
        return f"{where}: {severity}: {message}"
    if isinstance(loc, UnknownLoc):
        return f"<unknown>:0: {severity}: {message}"
    return f"<unknown>:0: {severity}: {loc}: {message}"


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
