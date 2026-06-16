# task.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Runtime tasks that emit lower-level ops into a runtime sequence."""

from typing import Callable

from ... import ir  # type: ignore

from ..buffer import Buffer
from ..resolvable import Resolvable


class RuntimeTask(Resolvable):
    """A unit of work emitted into the runtime sequence."""

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        pass


class InlineOpRuntimeTask(RuntimeTask):
    """Submits arbitrary lower-level IRON ops to the runtime sequence.

    Useful for tracing or hand-wired DMA programs that don't map onto the
    high-level fill/drain verbs."""

    def __init__(self, fn: Callable, args: list):
        """Construct an InlineOpRuntimeTask.

        Args:
            fn (Callable): The function that will generate ops. It runs within
                the runtime-sequence MLIR context.
            args (list): Arguments for the function (e.g. Buffers it uses).
        """
        self._fn = fn
        self._args = args

    @staticmethod
    def _resolve_buffers(obj, loc, ip):
        """Recursively resolve any Buffer instances found in obj (handles nested lists/tuples)."""
        if isinstance(obj, Buffer):
            obj.resolve(loc=loc, ip=ip)
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                InlineOpRuntimeTask._resolve_buffers(item, loc, ip)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        for arg in self._args:
            InlineOpRuntimeTask._resolve_buffers(arg, loc, ip)
        self._fn(*self._args)
