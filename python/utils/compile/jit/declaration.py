# declaration.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""What an operator class declares, read once as its body finishes.

The record only reads the class: each ``param()``, ``tunable()``,
``per_call()`` and ``In``/``Out``/``InOut`` stays the attribute it was written
as.
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Mapping
from dataclasses import MISSING
from enum import Enum
from types import FunctionType
from typing import Any, ClassVar, get_args, get_origin

from .markers import _Operand
from .params import Member, Param, PerCall, Select, _text


def _names(dims) -> set[str]:
    """Return the parameters ``dims`` name, through ``Select``."""
    out: set[str] = set()
    for d in dims:
        if isinstance(d, Param):
            out.add(d.name)
        elif isinstance(d, Select):
            out |= _names((d.flag, *d.when_true, *d.when_false))
    return out


def _present(m: Member, flags: Mapping[str, Any]) -> bool:
    """Return whether ``m`` exists under ``flags``; a flag left out is its default."""
    if m.when is None:
        return True
    value = flags.get(m.when.name, m.when.default)
    return value is not MISSING and bool(value)


class Declaration:
    """What an operator class declares.

    Args:
        cls: The subclass, its body just run.
        library: The base whose attributes the library reads; a parameter or
            member may not hide one of them.

    Attributes:
        name: The class's name.
        params: Every ``param()`` and ``tunable()``, by name, a base's first.
        members: The operands and per-call values, the most derived body's
            order first.
        nullable: The parameters annotated ``T | None``: a shape dimension
            of one is present only where it is not None.
        tunables: The tunables a caller or the tuner sets (``init=True``).
        key: The parameters the cores are built from: every tunable, what a
            stream's ``obj_shape=``, ``per=``, ``dtype=`` or ``when=`` names,
            and every ``param(array=True)``.

    Raises:
        TypeError: The class breaks a declaration rule; the message names it.
    """

    def __init__(self, cls: type, library: type) -> None:
        self.name = cls.__name__
        hints = inspect.get_annotations(cls, eval_str=True)
        for name, value in vars(cls).items():
            hint = hints.get(name)
            if name in hints and isinstance(value, Member):
                raise TypeError(
                    f"{cls.__name__}.{name}: members are declared without an "
                    f"annotation; annotating one makes it a constructor argument"
                )
            if (isinstance(value, (Param, Member))) and any(
                name in vars(b) for b in library.__mro__
            ):
                raise TypeError(
                    f"{cls.__name__}.{name} hides {library.__name__}.{name}, which the library reads; name it otherwise"
                )
            if isinstance(value, Param) and name not in hints:
                raise TypeError(
                    f"{cls.__name__}.{name}: annotate the parameter with its type, `{name}: int = {value!r}`"
                )
            if (
                name in hints
                and not isinstance(value, Param)
                and hint is not ClassVar
                and get_origin(hint) is not ClassVar
            ):
                raise TypeError(
                    f"{cls.__name__}.{name}: an annotated attribute is a param(), a tunable() or a ClassVar"
                )

        params: dict[str, Param] = {}
        nullable: set[str] = set()
        for klass in reversed(cls.__mro__):
            for name, value in vars(klass).items():
                if isinstance(value, Param):
                    params[name] = value
            if klass is not cls:
                nullable |= getattr(vars(klass).get("declaration"), "nullable", set())
        for name, value in vars(cls).items():
            inherited = params.get(name)
            if (
                inherited is not None
                and inherited.owner is not cls
                and not isinstance(value, (Param, property, classmethod, staticmethod))
                and not callable(value)
            ):
                raise TypeError(
                    f"{cls.__name__}.{name} = {value!r} does not override the inherited "
                    f"parameter: redeclare it, `{name}: int = tunable(default={value!r}, init=False)`"
                )
        nullable |= {
            n for n, h in hints.items() if n in params and type(None) in get_args(h)
        }

        # The most derived body's order wins for the members it declares;
        # inherited members it does not redeclare follow, in their own order.
        # A subclass hides an inherited member by assigning it None.
        members: dict[str, Member] = {}
        seen: set[str] = set()
        for klass in cls.__mro__:
            for name, value in vars(klass).items():
                if name not in seen:
                    seen.add(name)
                    if isinstance(value, Member):
                        members[name] = value

        for m in members.values():
            if m.owner is cls:
                self._check(cls, m, nullable)

        self.params = params
        self.members = tuple(members.values())
        self.nullable = frozenset(nullable)
        self.tunables = tuple(n for n, p in params.items() if p.tunable and p.init)
        named = {n for n, p in params.items() if p.array}
        for m in self.members:
            if isinstance(m, _Operand) and m.streamed:
                named |= _names(m.obj_dims) | _names(m.per or ())
                if isinstance(m.declared_dtype, Param):
                    named.add(m.declared_dtype.name)
                if m.when is not None:
                    named.add(m.when.name)
        self.key = tuple(n for n in params if n in named)

    @staticmethod
    def _check(owner: type, m: Member, nullable: set[str]) -> None:
        where = f"{owner.__name__}.{m.name}"
        if isinstance(m, PerCall):
            if m.bounds is not None and (
                not isinstance(m.bounds, Param) or m.bounds.tunable
            ):
                raise TypeError(
                    f"{where}: per_call(bounds={m.bounds!r}) must name a param(), the one a call may bound"
                )
            return
        assert isinstance(m, _Operand)
        for d in m.dims:
            for leaf in d.when_true + d.when_false if isinstance(d, Select) else (d,):
                if isinstance(leaf, bool) or not isinstance(leaf, (int, Param)):
                    raise TypeError(
                        f"{where}: dimension {leaf!r} is not a param() or an integer; declare an expression as a param()"
                    )
                if isinstance(leaf, Param) and leaf.tunable:
                    raise TypeError(
                        f"{where}: dimension {leaf.name} is a tunable(); a host shape may not depend on tuning"
                    )
        optional = [
            d.name for d in m.dims if isinstance(d, Param) and d.name in nullable
        ]
        if len(optional) > 1:
            raise TypeError(
                f"{where}: at most one T | None dimension {optional}, since the rank tells whether it is present"
            )
        if m.when is not None and (not isinstance(m.when, Param) or m.when.tunable):
            raise TypeError(
                f"{where}: when={m.when!r} must be a param(), the flag the operand exists under"
            )
        for d in m.obj_dims or ():
            if isinstance(d, bool) or not isinstance(d, (int, Param)):
                raise TypeError(
                    f"{where}: obj_shape dimension {d!r} is not a param(), a tunable() or an integer"
                )
        for ref in m.per or ():
            if not isinstance(ref, Param):
                raise TypeError(f"{where}: per={ref!r} must be a param() or tunable()")

    def __repr__(self) -> str:
        """Return what was declared and inferred, a line a kind."""
        params = self.params
        given = [n for n, p in params.items() if not p.tunable and p.derive is None]
        derived = [n for n, p in params.items() if p.derive is not None]
        lines = [
            self.name,
            f"  params:   {', '.join(given) or '-'}",
            f"  derived:  {', '.join(derived) or '-'}",
            "  tunables: "
            + (", ".join(_tunable(params[n]) for n in self.tunables) or "-"),
            "  resolved: "
            + (
                ", ".join(n for n, p in params.items() if p.tunable and not p.init)
                or "-"
            ),
            *(f"  {_member(m)}" for m in self.members),
            f"  key:      ({', '.join(self.key)})",
        ]
        return "\n".join(lines)

    def infer(self, inputs: Mapping[str, Any], outputs=(), **kwargs) -> dict[str, Any]:
        """Return the parameters the operands' shapes give.

        Each declared dimension is a ``param()`` or a literal, so this is a
        lookup. An optional input's ``when=`` flag is true where ``inputs``
        names it.

        Args:
            inputs: The input shapes, by operand name.
            outputs: The shapes of caller-supplied outputs, in declaration
                order.
            **kwargs: Construction arguments; the dimensions and flags among
                them pin values and are checked for agreement.

        Returns:
            ``{param: value}``: the pinned and the inferred parameters, and
            the operand flags.

        Raises:
            TypeError: The operands are not the declared ones.
            ValueError: A shape disagrees with the declaration or another
                operand.
        """
        operands = [m for m in self.members if isinstance(m, _Operand)]
        names = set(self.params)
        for m in operands:
            names |= _names(m.dims)
        given = {k: v for k, v in kwargs.items() if k in names}
        flags: dict[str, bool] = {}
        for m in operands:
            if not m.fills or m.when is None:
                continue
            flag, has = m.when.name, m.name in inputs
            if flags.setdefault(flag, has) != has:
                raise TypeError(
                    f"{self.name}: the operands {flag}= brings are given together"
                )
            if flag in given and bool(given[flag]) != has:
                raise TypeError(
                    f"{self.name}({flag}=True) takes {m.name}="
                    if not has
                    else f"{self.name}: {m.name}= is an operand only where {flag} is true"
                )
        given |= flags
        ins = [m for m in operands if m.fills and _present(m, given)]
        if [m.name for m in ins] != list(inputs):
            raise TypeError(
                f"{self.name} takes {len(ins)} operand(s) ({', '.join(m.name for m in ins)}), got {len(inputs)}"
            )
        outs = [m for m in operands if not m.fills and _present(m, given)]
        if outputs and len(outputs) != len(outs):
            raise TypeError(
                f"{self.name} produces {len(outs)} output(s) ({', '.join(m.name for m in outs)}), got {len(outputs)}"
            )
        bound: dict[str, Any] = dict(given)
        # Each dimension an operand's shape gives: (param, value, where).
        found: list[tuple[Param, int | None, str]] = []
        for m, shape in [(m, inputs[m.name]) for m in ins] + list(zip(outs, outputs)):
            shape = tuple(int(s) for s in shape)
            dims = list(m.dims)
            at = next(
                (
                    i
                    for i, d in enumerate(dims)
                    if isinstance(d, Param) and d.name in self.nullable
                ),
                None,
            )
            if at is not None:
                optional = dims.pop(at)
                if len(shape) == len(dims) + 1:
                    found.append((optional, shape[at], f"{m.name}.shape[{at}]"))
                    shape = shape[:at] + shape[at + 1 :]
                elif len(shape) == len(dims):
                    found.append((optional, None, f"{m.name} (rank {len(shape)})"))
                else:
                    raise ValueError(
                        f"{self.name}: operand {m.name} has rank {len(shape)}, declared {m!r}"
                    )
            expanded: list = []
            while dims:
                d = dims.pop(0)
                if not isinstance(d, Select):
                    expanded.append(d)
                    continue
                value = bound.get(d.flag.name, d.flag.default)
                if value is MISSING or isinstance(value, FunctionType):
                    raise ValueError(
                        f"{self.name}: {d.flag.name} selects {m.name}'s shape and has no default; pass it explicitly"
                    )
                dims[:0] = d.when_true if value else d.when_false
            dims = expanded
            if len(dims) == 1 and len(shape) != 1:
                # A flat operand takes a tensor of any rank: its one
                # dimension is the element count.
                shape = (math.prod(shape),)
            if len(shape) != len(dims):
                raise ValueError(
                    f"{self.name}: operand {m.name} has rank {len(shape)} {shape}, declared rank {len(dims)} {m!r}"
                )
            for i, (d, n) in enumerate(zip(dims, shape)):
                if isinstance(d, Param):
                    found.append((d, n, f"{m.name}.shape[{i}]"))
                elif int(d) != n:
                    raise ValueError(
                        f"{self.name}: operand {m.name}.shape[{i}] is {n}, declared {d}"
                    )
        origin = {k: "given" for k in given}
        for ref, value, where in found:
            if ref.name in bound and bound[ref.name] != value:
                raise ValueError(
                    f"{self.name}: {ref.name} is {value} from {where} but {bound[ref.name]} from {origin[ref.name]}"
                )
            bound[ref.name] = value
            origin.setdefault(ref.name, where)
        return bound


def _tunable(p: Param) -> str:
    """Return a tunable and its domain in a few words."""
    domain = p.domain
    if domain is None:
        return p.name
    if isinstance(domain, type) and issubclass(domain, Enum):
        text = "{" + ", ".join(m.name for m in domain) + "}"
    elif isinstance(domain, tuple):
        text = "{" + ", ".join(map(repr, domain)) + "}"
    else:
        text = "f(op, dev)" if domain.__name__ == "<lambda>" else domain.__name__
    return f"{p.name} in {text}"


def _member(m: Member) -> str:
    """Return a member's kind, name and, for an operand, its symbolic shape."""
    if isinstance(m, _Operand):
        per = "" if m.per is None else f" per [{', '.join(map(_text, m.per))}]"
        return f"{type(m).__name__:<9} {m.name} [{', '.join(map(_text, m.dims))}]{per}"
    return f"{'per_call':<9} {m.name}"
