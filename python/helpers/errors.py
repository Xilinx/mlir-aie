# errors.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Report compile failures the way Python reports errors.

Two kinds of failure reach a user building a design, and neither reads well on
its own.

IRON's own guards raise ordinary exceptions, so they already carry a traceback
-- but most of it is IRON. A rejected transfer produces some forty frames, two
of which are the user's; the line that actually caused it sits in the middle,
surrounded by internals nobody outside the compiler can act on.

MLIR's verifiers report the other kind, as text with a `file:line:col` prefix.
That names the right place but never shows it, so the reader has to go open the
file themselves -- and the prefix is the only part of the message that refers
to code they wrote.

Both are handled here by getting the failure into the shape Python already
knows how to print: frames the reader recognizes, and the source line quoted
underneath. Frames are filtered rather than discarded, in the manner of JAX's
`jax/_src/traceback_util.py`, and the full trace stays one environment variable
away for anyone debugging IRON itself.

This sits beside `sourceloc` rather than under `iron` because the two are
halves of one story -- capture where the user wrote something, then report
against it -- and because the failures arrive from both directions: `iron`
builds the IR, but `utils.compile` is what runs aiecc over it, and `iron`
imports `utils`, so only a layer below both can serve both.
"""

import functools
import linecache
import os
import re
import types

from ..ir import MLIRError  # pyright: ignore[reportMissingImports]
from .sourceloc import is_internal_file

# A diagnostic location, optionally named and optionally a callsite chain:
#   "core_fn"(callsite("design.py":45:4 at "design.py":49:13))
_FILE_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')
_NAMED_LOC = re.compile(r'"([^"]+)"\(')

# The same failure as a command-line tool prints it. The bindings quote the
# location after `error:`; MLIR's SourceMgr handler, which is what aiecc
# installs, leads with a bare one instead:
#   /path/to/design.py:42:7: error: 'func.return' op has 0 operands
_TOOL_ERROR = re.compile(r"^(.+?):(\d+):(\d+): error: (.*)$")

_FULL_TRACEBACK_ENV = "IRON_FULL_TRACEBACK"


class IronCompileError(MLIRError):
    """A design that could not be compiled, reported against the user's source.

    Subclasses `MLIRError` rather than `Exception` so that code already
    catching MLIR verification failures keeps working: the change here is how
    the failure reads, not what it is.
    """


class _Hop(Exception):
    """Carrier used to capture a synthesized frame. Never escapes this module."""


def _show_full_traceback() -> bool:
    return os.environ.get(_FULL_TRACEBACK_ENV, "") not in ("", "0")


def _synthesize_frame(filename: str, lineno: int, name: str):
    """Return a traceback whose one frame points at `filename:lineno` in `name`.

    Python reads the source line from `filename` itself when printing, which is
    what puts the offending code in the message rather than just its address.
    """
    try:
        # Python draws carets from the *synthesized* statement's columns, which
        # would underline an arbitrary 13-character prefix of the real line. It
        # omits them entirely when the statement spans the whole line, so pad
        # the raise out to that width -- MLIR gives a start column but no end,
        # so there is no honest sub-range to point at anyway.
        real = linecache.getline(filename, lineno).rstrip("\n").strip()
        pad = max(0, (len(real) - len("raise __hop__")) // 2)
        statement = "raise " + "(" * pad + "__hop__" + ")" * pad
        source = "\n" * (lineno - 1) + statement
        code = compile(source, filename, "exec").replace(co_name=name)
    except (SyntaxError, ValueError):  # pragma: no cover - defensive
        return None
    try:
        exec(code, {"__hop__": _Hop()})
    except _Hop as hop:
        tb = hop.__traceback__
        return tb.tb_next if tb is not None else None
    return None  # pragma: no cover - the exec above always raises


def _rebuild(frames, exc):
    """Attach `frames` (innermost first) to `exc` as a traceback."""
    tb = exc.__traceback__
    for filename, lineno, name in frames:
        hop = _synthesize_frame(filename, lineno, name)
        if hop is not None:
            tb = types.TracebackType(tb, hop.tb_frame, hop.tb_lasti, hop.tb_lineno)
    return exc.with_traceback(tb)


def filter_internal_frames(exc: BaseException) -> BaseException:
    """Drop IRON's own frames from `exc`, keeping the user's.

    The frames are removed rather than summarized: what remains is the path
    through the user's design, which is the part they can change. Set
    ``IRON_FULL_TRACEBACK=1`` to keep everything, which is what you want when
    the bug is in IRON rather than in the design.
    """
    if _show_full_traceback():
        return exc

    kept, tb = [], exc.__traceback__
    while tb is not None:
        if not is_internal_file(tb.tb_frame.f_code.co_filename):
            kept.append(tb)
        tb = tb.tb_next
    if not kept:
        # Caught inside a constructor, the traceback runs from the guard that
        # raised down to the entry wrapper and is internal end to end -- the
        # caller's frame is only appended when this is re-raised. Dropping the
        # lot leaves exactly that frame, which is the declaration at fault.
        return exc.with_traceback(None)

    rebuilt = None
    for entry in reversed(kept):
        rebuilt = types.TracebackType(
            rebuilt, entry.tb_frame, entry.tb_lasti, entry.tb_lineno
        )
    return exc.with_traceback(rebuilt)


def _tool_diagnostic(line: str):
    """(path, line, message) for a tool-printed error, or None.

    The path is confirmed to exist before it is believed. A synthesized frame
    is only worth making if Python can read the line back out to quote it --
    one pointing at a file that isn't there prints as a bare address, which is
    what this module exists to stop. Returning None leaves the caller to report
    the raw output, which is the honest answer when we can't place the failure.

    Trimming to the longest suffix that exists also recovers a path that
    arrived glued to other output, as happens when a tool writes progress with
    no trailing newline.
    """
    match = _TOOL_ERROR.match(line)
    if not match:
        return None
    path, lineno, message = match.group(1), int(match.group(2)), match.group(4)
    # Longest first, and both sides of each separator: the real path may be
    # absolute (keep the leading slash) or relative to the cwd (drop it).
    starts = [0]
    for i, char in enumerate(path):
        if char in "/\\":
            starts += [i, i + 1]
    for start in starts:
        candidate = path[start:]
        if os.path.isfile(candidate) and linecache.getline(candidate, lineno):
            return candidate, lineno, message
    return None


def _parse_diagnostic(text: str):
    """(message, frames innermost-first) for the first error in an MLIR diagnostic.

    Both renderings are accepted, because the same failure reaches here two
    ways: as a string built by the Python bindings when a verifier runs
    in-process, and as a tool's stderr when it runs under aiecc.
    """
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("error:"):
            tool = _tool_diagnostic(stripped)
            # A tool prints one location, so there is no callsite chain to
            # walk -- a single frame at the offending line is the whole trace.
            if tool:
                path, lineno, message = tool
                return message, [(path, lineno, "<design>")]
            continue
        body = stripped[len("error:") :].strip()
        locations = _FILE_LOC.findall(body)
        if not locations:
            continue
        named = _NAMED_LOC.match(body)
        # Everything up to the location prefix is the location itself; the rest
        # is what actually went wrong.
        message = body
        for _ in range(len(locations)):
            message = re.sub(r'^[^:]*"[^"]+":\d+:\d+\)*:?\s*', "", message, count=1)
        message = message.lstrip(") :")
        # A callsite lists the innermost frame first, which is also the order a
        # traceback is built in. Only that frame belongs to the named body; the
        # rest are the design code that declared it.
        frames = [
            (f, int(line_no), named.group(1) if (named and i == 0) else "<design>")
            for i, (f, line_no, _) in enumerate(locations)
        ]
        return message or body, frames
    return None, []


def compile_error_from_output(output: str) -> BaseException | None:
    """Build an `IronCompileError` for the first located error in `output`.

    Returns None when nothing in `output` carries a location, leaving the
    caller to report the failure however it already did -- a diagnostic we
    cannot place is better raw than dressed up as a frame that points nowhere.
    """
    message, frames = _parse_diagnostic(output)
    if not frames:
        return None
    return _rebuild(frames, IronCompileError(message))


def mlir_error_to_python(exc: BaseException) -> BaseException:
    """Recast an MLIR diagnostic as an `IronCompileError` against user source.

    Returns `exc` unchanged when it carries no usable location, so a failure we
    cannot place is never made harder to read than it already was.
    """
    rebuilt = compile_error_from_output(str(exc))
    if rebuilt is None:
        return exc
    rebuilt.__cause__ = exc if _show_full_traceback() else None
    return rebuilt


def _is_mlir_diagnostic(exc: BaseException) -> bool:
    """Whether `exc` is an MLIR verifier failure carrying a textual diagnostic."""
    return type(exc).__name__ == "MLIRError" or str(exc).lstrip().startswith(
        "Verification failed"
    )


def design_boundary(fn):
    """Report failures inside `fn` against the user's design rather than IRON.

    Applied where a design is turned into MLIR: everything below is IRON and
    everything above is the user, so this is the point at which a failure stops
    being a stack trace through a compiler and becomes a message about a design.
    """

    # Python appends the raising frame to a traceback, so this function's own
    # frame always survives filtering. Named for what it is, so the one IRON
    # frame left in the report reads as the boundary rather than as noise.
    @functools.wraps(fn)
    def iron_design_boundary(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except IronCompileError:
            raise
        except Exception as exc:
            if _is_mlir_diagnostic(exc):
                raise mlir_error_to_python(exc) from None
            raise filter_internal_frames(exc) from None

    return iron_design_boundary
