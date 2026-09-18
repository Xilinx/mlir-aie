# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generation-only identities for dispatch scalars, consumed by IRON Runtime."""

from typing import NoReturn


class _DispatchParameter:
    """An opaque value until Runtime replaces it with a sequence block argument."""

    __slots__ = ("name", "scalar_type", "position", "owner", "_binding")

    def __init__(self, name: str, scalar_type: type, position: int, owner: object):
        self.name = name
        self.scalar_type = scalar_type
        self.position = position
        self.owner = owner
        self._binding = None

    def __repr__(self) -> str:
        return f"DispatchTime({self.name!r}, {self.scalar_type.__name__})"

    def _misuse(self, *args, **kwargs) -> NoReturn:
        raise TypeError(
            f"DispatchTime parameter {self.name!r} has no generation-time value. "
            "Forward it directly in Runtime(seq, fn_args=[...]) and use the "
            "sequence callback's argument for runtime arithmetic/control flow. "
            "For shapes, dtypes, or worker configuration, use CompileTime or "
            f"specialize({self.name}=...)."
        )

    def _bind(self, binding: object) -> None:
        if self._binding is not None:
            raise TypeError(
                f"DispatchTime parameter {self.name!r} must be bound exactly once "
                "in one Runtime sequence."
            )
        self._binding = binding

    # Python and NumPy must not silently interpret a symbolic parameter as a
    # truthy object, a dtype, an object-array element, or a generation-time value.
    __bool__ = __int__ = __index__ = __float__ = __complex__ = __hash__ = _misuse
    __array__ = __array_ufunc__ = __getattr__ = _misuse
    __call__ = __iter__ = __len__ = __getitem__ = __contains__ = _misuse
    __eq__ = __ne__ = __lt__ = __le__ = __gt__ = __ge__ = _misuse
    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = _misuse
    __truediv__ = __rtruediv__ = __floordiv__ = __rfloordiv__ = _misuse
    __mod__ = __rmod__ = __pow__ = __rpow__ = __divmod__ = __rdivmod__ = _misuse
    __lshift__ = __rlshift__ = __rshift__ = __rrshift__ = _misuse
    __and__ = __rand__ = __or__ = __ror__ = __xor__ = __rxor__ = _misuse
    __neg__ = __pos__ = __abs__ = __invert__ = __round__ = _misuse
