# sourceloc.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Attribution of IRON objects back to the user code that declared them.

IRON objects are *declared* in one place and *resolved* into MLIR somewhere
else entirely: a user writes ``ObjectFifo(...)`` at the top of a design
function, but the matching ``aie.objectfifo`` op is not created until
[`Program`][iron.Program]'s `resolve_program` runs, long after that frame has
been popped. Asking for the user's location at resolve time therefore finds
only IRON's own internals.

The split below fixes that. `capture_source_site` runs during `__init__`,
while the declaring frame is still live, and records nothing but a filename,
line and column -- no MLIR Context is required, which matters because designs
are usually built before `resolve_program` opens one. `SourceSite.to_location`
then turns those coordinates into an `ir.Location` at resolve time, when a
Context does exist.
"""

import contextlib
import inspect
import sys
import threading
from pathlib import Path

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

# The `aie` package root, deliberately *not* resolved through symlinks: a dev
# build symlinks build/python/aie/... back at the source tree, and frames
# report whichever path was imported. Leaving both unresolved keeps this
# comparison and `f_code.co_filename` in the same namespace.
_AIE_ROOT = Path(__file__).parent.parent


def _is_internal(filename: str) -> bool:
    path = Path(filename)
    # sys.prefix covers the pip-installed case, where `aie` and its
    # dependencies all live under site-packages.
    return path.is_relative_to(_AIE_ROOT) or path.is_relative_to(sys.prefix)


class SourceSite:
    """Where in the user's source an IRON object was declared."""

    __slots__ = ("filename", "line", "col")

    def __init__(self, filename: str, line: int, col: int):
        self.filename = filename
        self.line = line
        self.col = col

    def to_location(self, name: str | None = None) -> ir.Location:
        """Materialize this site as an `ir.Location`. Requires a live Context.

        Args:
            name (str | None, optional): If given, wrap the file location in a
                named location, so diagnostics read ``"of_in"("design.py":42:4)``
                rather than bare coordinates. Defaults to None.
        """
        loc = ir.Location.file(self.filename, self.line, self.col)
        return ir.Location.name(name, childLoc=loc) if name else loc

    def __repr__(self) -> str:
        return f"{self.filename}:{self.line}:{self.col}"


def capture_source_site() -> SourceSite | None:
    """Record the user frame that is declaring an IRON object.

    Call this from an IRON object's `__init__`. Returns None when the whole
    stack is internal, which happens when IRON builds objects on the user's
    behalf and there is no user line to point at.
    """
    frame = inspect.currentframe()
    if frame is None:  # pragma: no cover - no frame support on this interpreter
        return None
    frame = frame.f_back
    while frame is not None and _is_internal(frame.f_code.co_filename):
        frame = frame.f_back
    if frame is None:
        return None
    # context=0 skips reading the source file while still giving exact columns.
    info = inspect.getframeinfo(frame, 0)
    col = info.positions.col_offset if info.positions else 0
    return SourceSite(info.filename, info.lineno, col or 0)


def site_of_function(fn) -> SourceSite | None:
    """Locate the `def` line of a user-supplied callable.

    Bodies such as a [`Worker`][iron.Worker]'s `core_fn` are turned into MLIR by
    *running* them, so their ops have no single declaration site. Pointing at
    the function definition keeps a diagnostic inside the right body even when
    the individual statement cannot be pinpointed.
    """
    code = getattr(fn, "__code__", None)
    if code is None or _is_internal(code.co_filename):
        return None
    return SourceSite(code.co_filename, code.co_firstlineno, 0)


def site_location(site: SourceSite | None, name: str | None = None):
    """Materialize `site`, or return None so the caller keeps MLIR's default."""
    return site.to_location(name) if site is not None else None


# ---------------------------------------------------------------------------
# Traced bodies
#
# A Worker's core_fn and a Runtime's sequence body become MLIR by being *run*.
# While they run, the user's own frame is live and carries the line currently
# executing, so ops built in that window can be attributed statement by
# statement rather than to the enclosing `def`.
#
# Everywhere else -- resolving a Buffer declared long ago -- the nearest user
# frame is whoever called `resolve_program`, which is the wrong answer. A
# marker around the body is what tells the two apart.
# ---------------------------------------------------------------------------

_state = threading.local()


def _body_stack() -> list:
    stack = getattr(_state, "bodies", None)
    if stack is None:
        stack = _state.bodies = []
    return stack


@contextlib.contextmanager
def traced_body(name: str | None = None, declared_at: "SourceSite | None" = None):
    """Mark the dynamic extent in which a user-supplied body is executing.

    Args:
        name (str | None, optional): The body's function name.
        declared_at (SourceSite | None, optional): Where the object owning this
            body -- the Worker, the Runtime -- was declared. Carried into the
            op's location as a caller frame so an error inside a core body can
            be reported with the declaration that put it there.
    """
    _body_stack().append((name, declared_at))
    try:
        yield
    finally:
        _body_stack().pop()


def current_body() -> tuple:
    """(name, declaration site) of the body being traced, or (None, None)."""
    stack = _body_stack()
    return stack[-1] if stack else (None, None)
