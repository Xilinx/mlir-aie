# errors.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Report design failures against the user's source rather than IRON's.

Set `IRON_FULL_TRACEBACK=1` to keep IRON's own frames when debugging IRON.
"""

import contextlib
import linecache
import os
import re
import types

from ..ir import MLIRError  # pyright: ignore[reportMissingImports]
from .sourceloc import is_internal_file

_FILE_LOC = re.compile(r'"([^"]+)":(\d+):(\d+)')
_NAMED_LOC = re.compile(r'^loc\("([^"]+)"\(')
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
    the location of its first error diagnostic.

    Args:
        exc: The exception to rewrite in place.

    Returns:
        `exc`.
    """
    if os.environ.get("IRON_FULL_TRACEBACK", "") in ("", "0"):
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
    if isinstance(exc, MLIRError):
        for diag in exc.error_diagnostics:  # pyright: ignore[reportAttributeAccessIssue]  # fmt: skip
            loc = str(diag.location)
            match = _FILE_LOC.search(loc)
            if match is not None:
                named = _NAMED_LOC.match(loc)
                name = named.group(1).rpartition(".")[2] if named else "<design>"
                _attach_source_frame(exc, match.group(1), int(match.group(2)), name)
                break
    return exc


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
