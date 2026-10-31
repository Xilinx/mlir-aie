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
import linecache
import os
from collections.abc import Callable
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
def _position(code: CodeType, lasti: int) -> tuple[int, int, int]:
    """Return the column, end line and end column of the instruction at `lasti`.

    Each is 0 when Python recorded none (-X no_debug_ranges).
    """
    # f_lasti counts bytes; co_positions has an entry per 2-byte unit.
    positions = itertools.islice(code.co_positions(), lasti // 2, None)
    _, end_line, col, end_col = next(positions)
    return col or 0, end_line or 0, end_col or 0


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
        callees: tuple[str | None, ...] = (),
        end: tuple[int, int] = (0, 0),
    ):
        self.filename = filename
        self.line = line
        self.col = col
        # (filename, line, col) of each calling user statement, innermost first.
        self.callers = callers
        # (filename, line, col) of each user frame that called the one
        # before it, innermost first.
        self.returned_to = returned_to
        # The function each of the declaring and returned-to statements called,
        # as its source spells it ("ObjectFifo", "forward", "load"), for a
        # site with no column; None for one with a column.
        self.callees = callees
        # Where the declaring call ends, which tells the calls of a chain
        # (`a.cons().forward()`) apart; (0, 0) when unknown.
        self.end = end

    @classmethod
    def capture(cls) -> "SourceSite":
        """Record the innermost user frames outside the `aie` package.

        Returns:
            The site, or one with no `filename` when the whole stack is
            internal.
        """
        frames, returned_to, callees = [], [], []
        end = (0, 0)
        depth = _depth()
        # Whether every frame since the declaring one is a user frame: those
        # are the frames a helper may have returned the object through.
        returning = True
        frame = inspect.currentframe()
        assert frame is not None, "CPython provides frames"
        below = frame
        while frame is not None and (
            len(frames) < depth or (returning and len(returned_to) < cls.RETURN_DEPTH)
        ):
            filename = frame.f_code.co_filename
            # "<frozen runpy>" is import machinery; "<string>", "<stdin>" and
            # IPython's "<ipython-input-...>" are the user's own code.
            if filename.startswith("<frozen ") or os.path.normcase(filename).startswith(
                _PASSTHROUGH
            ):
                if frames:
                    returning = False
            else:
                col, end_line, end_col = _position(frame.f_code, frame.f_lasti)
                site = (
                    # co_filename is relative for code compiled under a
                    # relative name, as a runner's exec(compile(src, path)),
                    # which neither a reader of the IR nor a later read
                    # after a chdir can resolve. "<string>" names no path.
                    filename if filename.startswith("<") else os.path.abspath(filename),
                    frame.f_lineno,
                    col,
                )
                # Without a column, the function a site called picks its call
                # on the line; reading a class's name costs the frame's locals.
                callee = below.f_code.co_name if site[2] == 0 else None
                if not frames:
                    if callee == "__init__" and "self" in below.f_locals:
                        callee = type(below.f_locals["self"]).__name__
                    callees.append(callee)
                    end = (end_line, end_col)
                elif returning and len(returned_to) < cls.RETURN_DEPTH:
                    returned_to.append(site)
                    callees.append(callee)
                if len(frames) < depth:
                    frames.append(site)
            below = frame
            frame = frame.f_back
        if not frames:
            return cls(None)
        return cls(
            *frames[0],
            callers=tuple(frames[1:]),
            returned_to=tuple(returned_to),
            callees=tuple(callees),
            end=end,
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
        numbered: consts_0, consts_1, ...), as is either branch of
        `x = A() if c else B()`.
        An object a helper returns, as `return Buffer(...)` or as one element
        of `return Buffer(...), offsets`, is named where the helper's caller
        assigns it: `wts = load(...)`, or `wts, offsets = pack(...)`.

        Returns:
            The variable's name; None when the object is not assigned where
            it is built or returned to (an argument to a call), or the source
            cannot be read.
        """
        if self.filename is None:
            return None
        # The element of a returned tuple the object is, if it is one.
        index = None
        sites = ((self.filename, self.line, self.col),) + self.returned_to
        callees = self.callees or (None,) * len(sites)
        # Only the declaring call's end is known.
        ends = [self.end] + [(0, 0)] * len(self.returned_to)
        for (filename, line, col), callee, end in zip(
            sites, callees, ends, strict=True
        ):
            # A design edited and rerun in one process is read afresh.
            try:
                stat = os.stat(filename)
                stamp = (stat.st_size, stat.st_mtime_ns)
            except OSError:
                stamp = None
            binding = _binding(filename, line, col, stamp, callee, end)
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

    def describe(self, kind: str) -> str:
        """Return how an error names an object before it has a name.

        Args:
            kind: The object's class, e.g. "ObjectFifo".

        Returns:
            The variable its statement assigns it to, as "act_in"; else where
            it was declared, as "ObjectFifo at design.py:12"; else `kind`.
        """
        name = self.assigned_name()
        if name is not None:
            return name
        if self.filename is None:
            return kind
        return f"{kind} at {os.path.basename(self.filename)}:{self.line}"

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


class Declared:
    """An object a design declares; `_site` is the statement that did.

    An error IRON raises from one of its methods is pointed at that statement.
    """

    _site: SourceSite


def _target_name(target: ast.AST) -> str | None:
    """`x` for `x = ...`, `self.x = ...` and `x[i] = ...`; `f` for `obj.f(...)`."""
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
def _statements(filename: str, stamp: tuple[int, int] | None) -> tuple:
    """Each assignment and return of a value in `filename`.

    Args:
        filename: A source file, or a cell IPython registered in `linecache`.
        stamp: The file's size and modification time, so an edited file
            misses the cache; None when it has none.

    Returns:
        Entries (kind, target names, value, unpacks): kind "assign" or
        "return", and whether the statement takes a tuple apart element by
        element, as `a, b = X(), Y()` and `return X(), offsets` do.
    """
    linecache.checkcache(filename)
    try:
        tree = ast.parse("".join(linecache.getlines(filename)), filename)
    except (SyntaxError, ValueError):
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
def _binding(
    filename: str,
    line: int,
    col: int,
    stamp: tuple[int, int] | None,
    callee: str | None = None,
    end: tuple[int, int] = (0, 0),
) -> tuple | None:
    """How the statement at a call site binds the call's result.

    Args:
        filename: The call site's file.
        line: Its line.
        col: Its column, 0 when unknown.
        stamp: The file's size and modification time, as `_statements`
            takes it.
        callee: The name of the function the site called, which picks its
            call among the line's when `col` is 0; None for any call.
        end: Where the site's call ends, (0, 0) when unknown. When known, the
            object is the statement's value only if that call is the value's
            outermost, not an earlier link of a chain.

    Returns:
        (kind, element, names): `kind` is "assign" or "return"; `element` is
        the call's place in a tuple value, `a, b = X(...), Y(...)` or
        `return X(...), offsets`, else None; `names` are an assignment's
        targets. None when the call's result is not the statement's value
        or an element of it, e.g. an argument to another call.
    """
    if col == 0:
        # Python run with -X no_debug_ranges records no columns, so any call
        # to `callee` spanning the line may be the site: answer only when
        # every one gives the same.
        answers = {
            _binding(filename, call.lineno, call.col_offset + 1, stamp)
            for statement in _statements(filename, stamp)
            for call in ast.walk(statement[2])
            if isinstance(call, ast.Call)
            and call.lineno <= line <= (call.end_lineno or call.lineno)
            and callee in (None, _target_name(call.func))
        }
        return answers.pop() if len(answers) == 1 else None
    for kind, names, value, unpacks in _statements(filename, stamp):
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
        exact = next(
            (
                node
                for node in ast.walk(value)
                if isinstance(node, ast.Call)
                and (node.end_lineno, node.end_col_offset) == end
            ),
            None,
        )
        if exact is not None:
            if exact is not value:
                return None
            if kind == "assign" and element is not None:
                return kind, None, (names[element],)
            return kind, element, names
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


def _traced_location(loc: "ir.Location") -> "ir.Location":
    """Return a traced statement's `loc` as an IRON object's would read.

    Each unprefixed NameLoc name is marked as a Python function, and a
    relative file path is made absolute, as `SourceSite.capture` makes its
    own: code compiled under a relative name records one.
    """
    if isinstance(loc, ir.NameLoc):
        name = loc.name_str
        if not name.startswith((IRON_PREFIX, PYTHON_PREFIX)):
            name = PYTHON_PREFIX + name
        return ir.Location.name(name, childLoc=_traced_location(loc.child_loc))
    if isinstance(loc, ir.CallSiteLoc):
        return ir.Location.callsite(
            _traced_location(loc.callee), [_traced_location(loc.caller)]
        )
    if isinstance(loc, ir.FusedLoc):
        return ir.Location.fused(
            [_traced_location(child) for child in loc.locations], loc.metadata
        )
    if isinstance(loc, ir.FileLineColLoc) and not (
        loc.filename.startswith("<") or os.path.isabs(loc.filename)
    ):
        return ir.Location.file(
            os.path.abspath(loc.filename),
            loc.start_line,
            loc.start_col,
            loc.end_line,
            loc.end_col,
        )
    return loc


def relocate(
    op: "ir.Operation", rewrite: "Callable[[ir.Location], ir.Location]"
) -> None:
    """Rewrite the location of `op` and of every op nested in it.

    Args:
        op: The outermost op.
        rewrite: Takes an op's location and returns its new one.
    """
    op.location = rewrite(op.location)
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                relocate(child.operation, rewrite)


@contextlib.contextmanager
def traced_body(fn):
    """Give each op `fn` emits its own source statement.

    `ir.loc_tracebacks` names each statement's location after the function it
    ran in; on exit, even when `fn` raises, those names are marked as Python
    functions ("py:"), so a reader can tell them from IRON objects' names
    ("iron:"), and relative file paths are made absolute.

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
    try:
        with ir.loc_tracebacks(
            max_depth=1, on_explicit_actn=ir.OnExplicitAction.USE_TRACEBACK
        ):
            yield
    finally:
        for op in block.operations:
            relocate(op.operation, _traced_location)
