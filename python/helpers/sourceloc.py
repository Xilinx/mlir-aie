# sourceloc.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Source locations for IRON objects, recorded where the user declares them."""

import ast
import contextlib
import functools
import inspect
import itertools
import os
import sys
from pathlib import Path
from types import CodeType, FunctionType

from .. import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

# A NameLoc's name says what it names: an IRON object ("iron:of0") or the
# Python function a traced statement ran in ("py:core_fn").
IRON_PREFIX = "iron:"
PYTHON_PREFIX = "py:"


def iron_name(name: str) -> str:
    """Return the NameLoc name for the IRON object called `name`."""
    return IRON_PREFIX + name


# Unresolved, like co_filename: a dev build symlinks build/python/aie to source.
AIE_ROOT = Path(__file__).parent.parent
# normcase: Windows paths compare without regard to case.
_AIE_PREFIX = os.path.normcase(f"{AIE_ROOT}{os.sep}")
# A `with` block over an `aie` @contextmanager enters through contextlib, whose
# frame sits between the package and the user's `with` statement.
CONTEXTLIB_FILE = contextlib.__file__
_PASSTHROUGH = (_AIE_PREFIX, os.path.normcase(CONTEXTLIB_FILE))


def is_internal_file(filename: str) -> bool:
    """Whether `filename` is part of the `aie` package rather than user code."""
    return os.path.normcase(filename).startswith(_AIE_PREFIX)


# Finding a statement's column walks its code, and captures repeat a few sites.
@functools.lru_cache(maxsize=4096)
def _column(code: CodeType, lasti: int) -> int:
    if sys.version_info < (3, 11):
        return 0
    # f_lasti counts bytes; co_positions has an entry per 2-byte unit.
    positions = itertools.islice(code.co_positions(), lasti // 2, None)
    return next(positions)[2] or 0


# User frames an IRON object records: its declaring statement, then the calls
# that led there, so objects a helper builds tell apart which call built them.
# 1 records the statement alone.
DEFAULT_DEPTH = 4
DEPTH_ENV = "AIE_SOURCE_LOC_DEPTH"


def _depth() -> int:
    try:
        return max(1, int(os.environ.get(DEPTH_ENV, DEFAULT_DEPTH)))
    except ValueError:
        return DEFAULT_DEPTH


def _is_user_frame(filename: str) -> bool:
    # "<frozen runpy>" is import machinery; "<string>", "<stdin>" and IPython's
    # "<ipython-input-...>" are the user's own code.
    return not filename.startswith("<frozen ") and not os.path.normcase(
        filename
    ).startswith(_PASSTHROUGH)


class SourceSite:
    """The user statement that declared an IRON object, and its callers.

    Recorded while the declaring frame is live, then turned into an
    `ir.Location` at resolve time, once a Context exists.
    """

    # How many helper functions an object may be returned through and still
    # be named after the statement that assigns it.
    RETURN_DEPTH = 4

    def __init__(
        self,
        filename: str | None,
        line: int = 0,
        col: int = 0,
        callers: tuple[tuple[str, int, int], ...] = (),
        returned_to: tuple = (),
    ):
        self.filename = filename
        self.line = line
        self.col = col
        # (filename, line, col) of each calling user statement, innermost first.
        self.callers = callers
        # (filename, line, col) of each user frame that called the one
        # before it, innermost first.
        self.returned_to = returned_to

    @classmethod
    def capture(cls) -> "SourceSite":
        """Record the innermost user frames outside the `aie` package.

        Returns:
            The site, or one with no `filename` when the whole stack is
            internal.
        """
        frames = []
        depth = _depth()
        frame = inspect.currentframe()
        while frame is not None and len(frames) < depth:
            filename = frame.f_code.co_filename
            if _is_user_frame(filename):
                frames.append(
                    (
                        # co_filename is relative for a module imported through
                        # a relative sys.path entry, which a reader of the IR
                        # cannot resolve. "<string>" names no path.
                        (
                            filename
                            if filename.startswith("<")
                            else os.path.abspath(filename)
                        ),
                        frame.f_lineno,
                        _column(frame.f_code, frame.f_lasti),
                    )
                )
            frame = frame.f_back
        if not frames:
            return cls(None)
        # The frames a helper may have returned the object through: the
        # user frames that directly called the declaring one.
        returned_to = []
        frame = inspect.currentframe()
        while frame is not None and not _is_user_frame(frame.f_code.co_filename):
            frame = frame.f_back
        caller = frame.f_back if frame is not None else None
        while (
            caller is not None
            and len(returned_to) < cls.RETURN_DEPTH
            and _is_user_frame(caller.f_code.co_filename)
        ):
            returned_to.append(
                (
                    caller.f_code.co_filename,
                    caller.f_lineno,
                    _column(caller.f_code, caller.f_lasti),
                )
            )
            caller = caller.f_back
        return cls(
            *frames[0], callers=tuple(frames[1:]), returned_to=tuple(returned_to)
        )

    def assigned_name(self) -> str | None:
        """Return the name the declaring statement assigns the object to.

        `act_in = ObjectFifo(...)` gives "act_in", as do `self.act_in = ...`,
        `act_in[i] = ...`, the first target of `act_in = other = ...` and the
        matching element of `act_in, other = ObjectFifo(...), ...`. Only the
        constructing statement counts: a later `alias = act_in` does not.
        Each object an element of a list or tuple display builds, as in
        `consts = [Buffer(w), Buffer(b), *([Buffer(t)] if act else [])]` or
        `bufs = [Buffer(t) for t in ts]`, is named after the list (then
        numbered: consts, consts_1, ...), as is either branch of
        `x = A() if c else B()`.
        An object a helper returns, as `return Buffer(...)` or as one element
        of `return Buffer(...), offsets`, is named where the helper's caller
        assigns it: `wts = load(...)`, or `wts, offsets = pack(...)`.
        None when the object is not assigned where it is built or returned to
        (an argument to a call), or the source cannot be read.
        """
        if self.filename is None:
            return None
        # The element of a returned tuple the object is, if it is one.
        index = None
        for site in ((self.filename, self.line, self.col),) + self.returned_to:
            binding = _binding(*site)
            if binding is None:
                return None
            kind, element, names = binding
            if kind == "assign":
                if index is None:
                    return names[0] if len(names) == 1 else None
                # `a, b = helper()` unpacks the tuple the helper returned.
                if len(names) > 1 and len(names) > index:
                    return names[index]
                return None
            # A return passes the object to the caller's statement.
            if index is not None and element is not None:
                return None
            index = element if index is None else index
        return None

    def location(self, name: str | None = None) -> "ir.Location":
        """Return this site as an `ir.Location` in the current Context.

        Args:
            name: Wraps the location in a `NameLoc`, e.g. the object's symbol.

        Returns:
            The statement's file location, as the callee of a `CallSiteLoc`
            when callers were recorded, or the ambient location when there is
            no user site.
        """
        if self.filename is None:
            return ir.Location.current
        loc = ir.Location.file(self.filename, self.line, self.col)
        if self.callers:
            loc = ir.Location.callsite(
                loc, [ir.Location.file(*caller) for caller in self.callers]
            )
        return ir.Location.name(iron_name(name), childLoc=loc) if name else loc


def _target_name(target: ast.AST) -> str | None:
    """`x` for `x = ...`, `self.x = ...` and `x[i] = ...`."""
    while isinstance(target, ast.Subscript):
        target = target.value
    if isinstance(target, ast.Name):
        return target.id
    if isinstance(target, ast.Attribute):
        return target.attr
    return None


# Parsed rather than read from bytecode: how a list, comprehension or
# conditional element reaches its store differs between Python versions.
@functools.lru_cache(maxsize=64)
def _statements(filename: str) -> tuple:
    """Each assignment and return of a value in `filename`.

    Entries are (kind, target names, value, unpacks): kind "assign" or
    "return", and whether the statement takes a tuple apart element by
    element, as `a, b = X(), Y()` and `return X(), offsets` do.
    """
    try:
        with open(filename) as f:
            tree = ast.parse(f.read(), filename)
    except (OSError, SyntaxError, ValueError):
        return ()
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Return) and node.value is not None:
            found.append(("return", (), node.value, isinstance(node.value, ast.Tuple)))
            continue
        if isinstance(node, ast.Assign):
            targets, value = node.targets, node.value
        elif isinstance(node, ast.AnnAssign) and node.value is not None:
            targets, value = [node.target], node.value
        else:
            continue
        # `a = b = ObjectFifo(...)` is named after the first target, and
        # `a, b = ...` after the element in the object's place.
        target = targets[0]
        unpacks = isinstance(target, (ast.Tuple, ast.List))
        elts = target.elts if unpacks else [target]
        found.append(("assign", tuple(_target_name(t) for t in elts), value, unpacks))
    return tuple(found)


def _contains(node: ast.expr, line: int, col: int) -> bool:
    if col == 0:  # Before Python 3.11 there are no columns; match the line.
        return node.lineno <= line <= (node.end_lineno or node.lineno)
    start = (node.lineno, node.col_offset)
    end = (node.end_lineno, node.end_col_offset)
    return start <= (line, col) <= end


def _spine(value: ast.expr) -> list:
    """Return the calls `value` chains: both calls of `a.cons().forward()`."""
    calls = []
    while isinstance(value, (ast.Call, ast.Attribute)):
        if isinstance(value, ast.Call):
            calls.append(value)
            value = value.func
        else:
            value = value.value
    return calls


def _element(value: ast.expr, line: int, col: int) -> ast.expr | None:
    """Return the part of an assigned value that holds the site.

    Goes into the element of a list, tuple or set display (`[A(), *rest]`),
    the element expression of a comprehension (`[A(t) for t in ts]`) and the
    chosen branch of a conditional (`[A()] if c else []`), so the object an
    element builds counts as the value. Returns None when the site is in a
    conditional's test or a comprehension's iterable.
    """
    while True:
        if isinstance(value, (ast.List, ast.Tuple, ast.Set)):
            inner = next((e for e in value.elts if _contains(e, line, col)), None)
        elif isinstance(value, ast.Starred):
            inner = value.value
        elif isinstance(value, ast.IfExp):
            inner = next(
                (b for b in (value.body, value.orelse) if _contains(b, line, col)),
                None,
            )
        elif isinstance(value, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            inner = value.elt if _contains(value.elt, line, col) else None
        else:
            return value
        if inner is None:
            return None
        value = inner


@functools.lru_cache(maxsize=4096)
def _binding(filename: str, line: int, col: int) -> tuple | None:
    """How the statement at a call site binds the call's result.

    Returns:
        (kind, element, names): `kind` is "assign" or "return"; `element` is
        the call's place in a tuple value, `a, b = X(...), Y(...)` or
        `return X(...), offsets`, else None; `names` are an assignment's
        targets. None when the call's result is not the statement's value
        or an element of it, e.g. an argument to another call.
    """
    for kind, names, value, unpacks in _statements(filename):
        if not _contains(value, line, col):
            continue
        element = None
        if unpacks and isinstance(value, (ast.Tuple, ast.List)):
            element = next(
                (i for i, e in enumerate(value.elts) if _contains(e, line, col)),
                None,
            )
            if element is None:
                return None
            # `a, b = X(), Y()` pairs elements with targets; any other
            # assignment of a tuple names no one element.
            if kind == "assign" and len(names) != len(value.elts):
                return None
            value = value.elts[element]
        elif kind == "assign" and len(names) != 1:
            # `a, b = helper()`: the caller unpacks a returned tuple.
            if not isinstance(value, ast.Call) or not _contains(value, line, col):
                return None
        elif kind == "assign" and not unpacks:
            # `consts = [Buffer(...), Buffer(...)]`: each element is named
            # after the list.
            value = _element(value, line, col)
            if value is None:
                return None
        # The object is the statement's value, not built inside an argument
        # of it: the innermost call around the site must be on the value's
        # chain of calls. ast.walk goes outside in, so the last call around
        # the site is the innermost one.
        innermost = None
        for node in ast.walk(value):
            if isinstance(node, ast.Call) and _contains(node, line, col):
                innermost = node
        if innermost is None or not any(innermost is c for c in _spine(value)):
            return None
        if kind == "assign" and element is not None:
            return kind, None, (names[element],)
        return kind, element, names
    return None


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


def _python_names(loc: "ir.Location") -> "ir.Location":
    """Return `loc` with each unprefixed NameLoc name marked as a function."""
    if isinstance(loc, ir.NameLoc):
        name = loc.name_str
        if not name.startswith((IRON_PREFIX, PYTHON_PREFIX)):
            name = PYTHON_PREFIX + name
        return ir.Location.name(name, childLoc=_python_names(loc.child_loc))
    if isinstance(loc, ir.CallSiteLoc):
        return ir.Location.callsite(
            _python_names(loc.callee), [_python_names(loc.caller)]
        )
    if isinstance(loc, ir.FusedLoc):
        return ir.Location.fused(
            [_python_names(child) for child in loc.locations], loc.metadata
        )
    return loc


def _mark_python_names(op: "ir.Operation") -> None:
    op.location = _python_names(op.location)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                _mark_python_names(child.operation)


@contextlib.contextmanager
def traced_body(fn):
    """Give each op `fn` emits its own source statement.

    `ir.loc_tracebacks` names each statement's location after the function it
    ran in; those names are marked as Python functions ("py:") on exit, so a
    reader can tell them from IRON objects' names ("iron:").

    Args:
        fn: A Worker or Runtime body about to run.

    Yields:
        Nothing. When IRON wrote `fn`, its ops take the declaring object's
        location instead.
    """
    if isinstance(fn, FunctionType) and is_internal_file(fn.__code__.co_filename):
        yield
        return
    block = ir.InsertionPoint.current.block
    with ir.loc_tracebacks(
        max_depth=1, on_explicit_actn=ir.OnExplicitAction.USE_TRACEBACK
    ):
        yield
    for op in block.operations:
        _mark_python_names(op.operation)
