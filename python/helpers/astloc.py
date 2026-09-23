# astloc.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Statement-precise MLIR locations for bodies that are traced by execution.

A Worker's `core_fn` and a Runtime's sequence body become MLIR by being run, so
the ops they emit take whatever location is ambient at the time. Setting that
ambient location once per body is the best a wrapper can do, and it lands every
op on the enclosing `def`.

The ops in question are not ours to fix one by one: `elem_out[0] = elem_in[0] +
elem_in[1]` goes through eudsl's `MemRefValue.__setitem__` and
`ArithValue.__add__`, and `objectfifo.acquire` goes through a generated builder.
None of them accept a location from us, and all of them read the ambient one.

So the ambient location is made to track the statement instead. The body's AST
is rewritten to wrap each statement in a location scope, which every builder
underneath then picks up for free. This is the approach Triton takes, and the
rewrite machinery -- including the line-number bookkeeping that keeps a
recompiled function pointing at its original source -- already ships in
`aie.extras.ast.canonicalize`.
"""

import ast
import contextlib
import inspect
import logging
import types

from ..extras.ast.canonicalize import (  # pyright: ignore[reportMissingImports]
    StrictTransformer,
    transform_ast,
)
from ..extras.ast.util import (  # pyright: ignore[reportMissingImports]
    ast_call,
    copy_func,
    set_lineno,
)
from ..ir import Context, Location  # pyright: ignore[reportMissingImports]
from .sourceloc import current_body

logger = logging.getLogger(__name__)

# Injected into the rewritten body's globals under a name a user will not
# collide with.
_HOOK = "__iron_stmt_loc__"

# Wrapping these changes meaning or is pointless: a docstring stops being one,
# and scope declarations must stay at the top of their block.
_UNWRAPPABLE = (ast.Global, ast.Nonlocal, ast.Pass, ast.Import, ast.ImportFrom)


def _statement_location(line: int, col: int):
    """Runtime hook: make the caller's `file:line:col` the ambient location.

    The filename is taken from the calling frame rather than baked into the
    rewrite, because `transform_ast` recompiles against the original
    `co_filename` -- so the executing body already reports the right file.
    """
    # A rewritten body may also be called outside any MLIR Context -- IRON
    # inspects some of them without tracing. Setting a location is meaningless
    # there, and must not be fatal.
    if Context.current is None:
        return contextlib.nullcontext()
    frame = inspect.currentframe()
    caller = frame.f_back if frame else None
    filename = caller.f_code.co_filename if caller else "<unknown>"
    loc = Location.file(filename, line, col)
    name, declared_at = current_body()
    # The declaration becomes a caller frame. MLIR renders this as a callsite,
    # which is noisy to read directly -- its value is that it survives into the
    # IR so an error reporter can rebuild the nesting as a Python traceback.
    if declared_at is not None:
        loc = Location.callsite(loc, [declared_at.to_location()])
    return Location.name(name, childLoc=loc) if name else loc


class _StatementLocations(StrictTransformer):
    """Wrap every statement in `with __iron_stmt_loc__(line, col):`."""

    # transform_func drives a transformer by calling generic_visit on the
    # FunctionDef itself, so that -- not visit_FunctionDef -- is the entry point.
    def generic_visit(self, node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            node.body = self._wrap_all(node.body)
        return node

    def _wrap_all(self, body: list) -> list:
        return [w for stmt in body for w in self._wrap(stmt)]

    def _wrap(self, stmt: ast.stmt) -> list:
        # Recurse first so nested blocks are wrapped statement-by-statement too.
        for field in ("body", "orelse", "finalbody"):
            inner = getattr(stmt, field, None)
            if isinstance(inner, list) and inner and isinstance(inner[0], ast.stmt):
                setattr(stmt, field, self._wrap_all(inner))
        for handler in getattr(stmt, "handlers", []):
            handler.body = self._wrap_all(handler.body)

        if isinstance(stmt, _UNWRAPPABLE):
            return [stmt]

        # first_lineno maps the re-parsed (1-based) CST back onto the file.
        line = stmt.lineno + self.first_lineno
        col = getattr(stmt, "col_offset", 0) or 0
        item = ast.withitem(
            context_expr=ast_call(_HOOK, args=[ast.Constant(line), ast.Constant(col)]),
            optional_vars=None,
        )
        # Position only the synthesized nodes. Running set_lineno over the whole
        # wrapper would flatten the wrapped statement's own sub-expressions onto
        # one line, costing precision in any Python traceback through it.
        set_lineno(item.context_expr, stmt.lineno)
        wrapper = ast.With(items=[item], body=[stmt])
        wrapper.lineno = stmt.lineno
        wrapper.end_lineno = getattr(stmt, "end_lineno", stmt.lineno)
        wrapper.col_offset = getattr(stmt, "col_offset", 0)
        wrapper.end_col_offset = getattr(stmt, "end_col_offset", 0)
        return [wrapper]


def with_statement_locations(fn):
    """Return a copy of `fn` whose statements each scope an MLIR location.

    Returns `fn` unchanged when it cannot be rewritten -- a builtin, a
    `functools.partial`, or a body whose source is unavailable (a REPL). Losing
    statement precision is not worth failing a compile over, but the reason is
    logged rather than swallowed.
    """
    if not hasattr(fn, "__code__"):
        return fn
    try:
        copy = copy_func(fn)
        copy.__globals__[_HOOK] = _statement_location
        rewritten = transform_ast(copy, [_StatementLocations])
        if fn.__closure__ is None:
            return rewritten
        # transform_ast recompiles the body standalone, so eudsl re-materializes
        # the closed-over variables as fresh cells holding a snapshot of their
        # values. A body that mutates closed-over state -- an accumulator built
        # up across a metaprogramming loop, say -- would then silently update a
        # copy. Rebind the original cells; the rewrite only wraps statements, so
        # the free variables are unchanged.
        if rewritten.__code__.co_freevars != fn.__code__.co_freevars:
            raise ValueError(
                f"free variables changed: {fn.__code__.co_freevars} -> "
                f"{rewritten.__code__.co_freevars}"
            )
        rebound = types.FunctionType(
            rewritten.__code__,
            fn.__globals__,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )
        rebound.__kwdefaults__ = fn.__kwdefaults__
        rebound.__dict__.update(fn.__dict__)
        return rebound
    except Exception as exc:
        logger.debug(
            "no statement locations for %s: %s", getattr(fn, "__name__", fn), exc
        )
        return fn
