# resolvable.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
"""Structural protocol for objects that lower to MLIR operations."""

from typing import Protocol, runtime_checkable

from .. import ir  # pyright: ignore[reportMissingImports]


# Structural typing via @runtime_checkable Protocol: any class with both
# .resolve() and .tiles() passes isinstance(x, Resolvable).  The two-method
# requirement is the safeguard against false positives from classes that
# happen to define an unrelated .resolve() (e.g. pathlib.Path).
@runtime_checkable
class Resolvable(Protocol):
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        """Resolve the current object into one or more MLIR operations.
        Should only be called within an MLIR context.

        Args:
            loc (ir.Location | None, optional): Location is used by MLIR object during construction in some cases. Defaults to None.
            ip (ir.InsertionPoint | None, optional): InsertionPoint is used by MLIR object during construction in some cases. Defaults to None.
        """
        ...

    def tiles(self) -> list:
        """Tiles this Resolvable depends on for code generation.

        Override this in user-side Resolvable subclasses that reference tiles
        which aren't already discoverable via Workers or ObjectFifos. The
        Program will resolve these tiles before calling :meth:`resolve`, so
        ``tile.op`` is valid by then. Default: empty list.
        """
        return []


class NotResolvedError(Exception):
    """Raised when a property or operation is accessed on a :class:`Resolvable` object
    before :meth:`resolve` has been called.
    """

    def __init__(self, message="Cannot get operation; class not resolved."):
        self.message = message
        super().__init__(self.message)
