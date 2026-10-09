# sourceloc.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Source locations for IRON objects, recorded where the user declares them."""

import contextlib
import inspect
import itertools
import os
from pathlib import Path
from types import FunctionType

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

# Unresolved, like co_filename: a dev build symlinks build/python/aie to source.
AIE_ROOT = Path(__file__).parent.parent
_AIE_PREFIX = f"{AIE_ROOT}{os.sep}"


def is_internal_file(filename: str) -> bool:
    """Whether `filename` is part of the `aie` package rather than user code."""
    return filename.startswith(_AIE_PREFIX)


class SourceSite:
    """The user statement that declared an IRON object.

    Recorded while the declaring frame is live, then turned into an
    `ir.Location` at resolve time, once a Context exists.
    """

    def __init__(self, filename: str | None, line: int = 0, col: int = 0):
        self.filename = filename
        self.line = line
        self.col = col

    @classmethod
    def capture(cls) -> "SourceSite":
        """Record the innermost frame outside the `aie` package.

        Returns:
            The site, or one with no `filename` when the whole stack is
            internal.
        """
        frame = inspect.currentframe()
        while frame is not None and is_internal_file(frame.f_code.co_filename):
            frame = frame.f_back
        if frame is None:
            return cls(None)
        # f_lasti counts bytes; co_positions has an entry per 2-byte unit.
        positions = itertools.islice(
            frame.f_code.co_positions(), frame.f_lasti // 2, None
        )
        col = next(positions)[2] or 0
        return cls(frame.f_code.co_filename, frame.f_lineno, col)

    def location(self, name: str | None = None) -> "ir.Location":
        """Return this site as an `ir.Location` in the current Context.

        Args:
            name: Wraps the location in a `NameLoc`, e.g. the object's symbol.

        Returns:
            The location, or the ambient one when there is no user site.
        """
        if self.filename is None:
            return ir.Location.current
        loc = ir.Location.file(self.filename, self.line, self.col)
        return ir.Location.name(name, childLoc=loc) if name else loc


def user_code_location() -> "ir.Location | None":
    """Return the innermost statement outside the `aie` package as a location.

    Returns:
        Its file location in the current Context, or None, so that an op
        takes the ambient location, outside a Context or when the whole stack
        is internal.
    """
    if ir.Context.current is None:
        return None
    site = SourceSite.capture()
    return site.location() if site.filename else None


def traced_body(fn) -> contextlib.AbstractContextManager:
    """Return the scope that gives each op `fn` emits its own source statement.

    Args:
        fn: A Worker or Runtime body about to run.

    Returns:
        A traceback-location scope, or a null one when IRON wrote `fn`, whose
        ops then take the declaring object's location.
    """
    if isinstance(fn, FunctionType) and is_internal_file(fn.__code__.co_filename):
        return contextlib.nullcontext()
    return ir.loc_tracebacks(
        max_depth=1, on_explicit_actn=ir.OnExplicitAction.USE_TRACEBACK
    )
