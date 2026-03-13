# resolvable.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
"""Abstract base class for objects that lower to MLIR operations."""

from abc import ABC, abstractmethod

from .. import ir  # type: ignore


class Resolvable(ABC):
    @abstractmethod
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


class NotResolvedError(Exception):
    """Raised when a property or operation is accessed on a :class:`Resolvable` object
    before :meth:`resolve` has been called.
    """

    def __init__(self, message="Cannot get operation; class not resolved."):
        self.message = message
        super().__init__(self.message)
