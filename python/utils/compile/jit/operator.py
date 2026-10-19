# operator.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""The declaration half of an operator: construction, copies and identity.

A library's operator class (IRON's) subclasses ``Operator`` with
``library=True`` and adds building, tuning and dispatch; each class below it
is read into a ``Declaration`` as its body finishes.
"""

from __future__ import annotations

from dataclasses import MISSING
from enum import Enum
from types import FunctionType
from typing import Any, ClassVar, Self, dataclass_transform

from ...hostruntime import set_current_device
from .declaration import Declaration
from .params import param, tunable


class Unresolvable(ValueError):
    """No legal resolution exists for this operator on this device."""


@dataclass_transform(kw_only_default=True, field_specifiers=(param, tunable))
class Operator:
    """An operator. Subclass it; each subclass's body is its declaration.

    Attributes:
        declaration: What the class declares.
        reference_only: The operator has a reference and no kernel yet.
        trace: What the build traces, or None.
    """

    declaration: ClassVar[Declaration]
    reference_only: ClassVar[bool] = False
    _library: ClassVar[type] = None  # pyright: ignore[reportAssignmentType]
    trace: Any = None

    def __init_subclass__(cls, library: bool = False, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        if library:
            cls._library = cls
            return
        library_cls = cls._library or Operator
        cls.declaration = Declaration.of_class(cls, library_cls)
        if cls.reference_only:
            kernel = [
                n
                for k in cls.__mro__
                if k is not library_cls and issubclass(k, library_cls)
                for n in ("build", "kernel", "sequence", "exported_design")
                if n in vars(k)
            ]
            if kernel:
                raise TypeError(
                    f"{cls.__name__} is reference_only, so it declares no {kernel[0]}()"
                )

    def __init__(self, *, trace: Any = None, **given: Any) -> None:
        params = self.declaration.params
        unknown = [n for n in given if n not in params or not params[n].init]
        if unknown:
            raise TypeError(f"{type(self).__name__}() takes no argument {unknown}")
        missing = [
            n
            for n, p in params.items()
            if p.init and p.default is MISSING and p.derive is None and n not in given
        ]
        if missing:
            raise TypeError(f"{type(self).__name__}() needs {missing}")
        self.trace = trace
        self._set(given)

    def _set(self, given: dict[str, Any]) -> None:
        """Set every parameter: given, else its default or what computes it.

        None counts as not given. Plain defaults are set first; then each
        function default or ``derive`` is computed in declaration order. A
        given ``derive`` parameter must equal what it computes.

        Raises:
            TypeError: An ``Enum``-domain tunable holds another type.
            ValueError: A computed parameter is None, or a given ``derive``
                one disagrees.
        """
        params = self.declaration.params
        computed = {
            n
            for n, p in params.items()
            if p.derive is not None or isinstance(p.default, FunctionType)
        }
        for n, p in params.items():
            value = given.get(n)
            plain = n not in computed and p.default is not MISSING
            vars(self)[n] = p.default if value is None and plain else value
        self._resolved = False
        for n, p in params.items():
            if n in computed and vars(self)[n] is None:
                value = (p.derive or p.default)(self)
                if value is None and n not in self.declaration.nullable:
                    raise ValueError(
                        f"{type(self).__name__}: {n} (None) must be given: nothing computes it"
                    )
                vars(self)[n] = value
            domain = p.domain
            value = vars(self)[n]
            if (
                isinstance(domain, type)
                and issubclass(domain, Enum)
                and value is not None
                and not isinstance(value, domain)
            ):
                raise TypeError(
                    f"{type(self).__name__}: {n} ({value!r}) must be a {domain.__name__}"
                )
        for n in computed:
            p = params[n]
            if (
                p.derive is not None
                and given.get(n) is not None
                and given[n] != (expected := p.derive(self))
            ):
                raise ValueError(
                    f"{type(self).__name__}: {n} ({given[n]!r}) must be {expected!r}, what its other parameters make it"
                )
        self.validate()
        self._bind()

    def _bind(self) -> None:
        """Bind each present member to this operator."""
        self._bound = {
            m.name: m.of(self)
            for m in self.declaration.members
            if m.when is None or vars(self)[m.when.name]
        }

    def validate(self) -> None:
        """Check the parameters on their own; raise ``ValueError``. Runs at construction."""

    def compatible(self) -> None:
        """Check the parameters against the resolved tunables; raise ``ValueError``.

        Runs on the copy ``resolved`` returns.
        """

    def resolve(self, dev) -> Self:
        """Return a copy with every tunable filled for ``dev``.

        Raises:
            Unresolvable: No legal value exists on ``dev``.
        """
        return self.replace()

    def resolved(self, dev) -> Self:
        """Return this operator resolved for ``dev``, which is current while it resolves and is checked.

        Raises:
            TypeError: ``resolve`` returned ``self`` or another class.
            Unresolvable: ``resolve`` left a tunable None.
        """
        if self._resolved:
            return self
        with set_current_device(dev):
            new = self.resolve(dev)
            if new is self or not isinstance(new, type(self)):
                raise TypeError(
                    f"{type(self).__name__}.resolve() must return a copy, self.replace(...)"
                )
            missing = [
                n
                for n, p in self.declaration.params.items()
                if p.tunable and vars(new)[n] is None
            ]
            if missing:
                raise Unresolvable(
                    f"{type(self).__name__}.resolve() left {missing} unset for {dev}"
                )
            new.compatible()
        new._resolved = True
        return new

    def replace(self, **changes: Any) -> Self:
        """Return a copy with ``changes``, which may name any parameter.

        A ``derive`` parameter not in ``changes`` is computed again; every
        other keeps its value, so pass a function-default one again when you
        change what it reads.
        """
        params = self.declaration.params
        unknown = [n for n in changes if n not in params]
        if unknown:
            raise TypeError(f"{type(self).__name__} has no parameter {unknown}")
        kept = {n: vars(self)[n] for n, p in params.items() if p.derive is None}
        new = object.__new__(type(self))
        new.trace = self.trace
        new._set(kept | changes)
        return new

    def layout(self) -> Self:
        """Return this operator as its cores see it: the key's parameters and the members."""
        view = object.__new__(type(self))
        vars(view).update(
            {n: vars(self)[n] for n in self.declaration.key},
            _bound=self._bound,
            trace=self.trace,
        )
        return view

    def array_key(self) -> tuple:
        """Return the identity two operators share their cores by."""
        key = (type(self), *((n, _value(vars(self)[n])) for n in self.declaration.key))
        return key if self.trace is None else (*key, ("trace", self.trace))

    def design_key(self) -> tuple:
        """Return the identity two operators share a build by: every parameter."""
        return (
            type(self),
            (
                ("trace", self.trace),
                *((n, _value(vars(self)[n])) for n in self.declaration.params),
            ),
        )

    def __repr__(self) -> str:
        own = ", ".join(
            f"{n}={vars(self)[n]!r}"
            for n, p in self.declaration.params.items()
            if p.repr
        )
        return f"{type(self).__name__}({own})"


def _value(v: Any) -> Any:
    # An Enum is its value, so renaming a member moves no key.
    return v.value if isinstance(v, Enum) else v
