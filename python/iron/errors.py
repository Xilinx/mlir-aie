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
"""

import functools
import os
import re
import types

from ..ir import MLIRError  # pyright: ignore[reportMissingImports]
from ..helpers.sourceloc import is_internal_file

# A diagnostic location, optionally named and optionally a callsite chain:
#   "core_fn"(callsite("design.py":45:4 at "design.py":49:13))
_FILE_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')
_NAMED_LOC = re.compile(r'"([^"]+)"\(')

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
    """A traceback whose one frame points at `filename:lineno` inside `name`.

    Python reads the source line from `filename` itself when printing, which is
    what puts the offending code in the message rather than just its address.
    """
    try:
        source = "\n" * (lineno - 1) + "raise __hop__"
        code = compile(source, filename, "exec").replace(co_name=name)
    except (SyntaxError, ValueError):  # pragma: no cover - defensive
        return None
    try:
        exec(code, {"__hop__": _Hop()})
    except _Hop as hop:
        return hop.__traceback__.tb_next
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
        # Nothing but internals: an IRON bug, not a design error. Keep the
        # whole trace rather than reporting an exception with no frames.
        return exc

    rebuilt = None
    for entry in reversed(kept):
        rebuilt = types.TracebackType(
            rebuilt, entry.tb_frame, entry.tb_lasti, entry.tb_lineno
        )
    return exc.with_traceback(rebuilt)


def _parse_diagnostic(text: str):
    """(message, frames innermost-first) for the first error in an MLIR diagnostic."""
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith("error:"):
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


def mlir_error_to_python(exc: BaseException) -> BaseException:
    """Recast an MLIR diagnostic as an `IronCompileError` against user source.

    Returns `exc` unchanged when it carries no usable location, so a failure we
    cannot place is never made harder to read than it already was.
    """
    message, frames = _parse_diagnostic(str(exc))
    if not frames:
        return exc
    rebuilt = _rebuild(frames, IronCompileError(message))
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
