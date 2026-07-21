# task.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from typing import Callable

from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

from ..buffer import Buffer
from ..resolvable import Resolvable
from .taskgroup import TaskGroup


class RuntimeTask(Resolvable):
    """A RuntimeTask is a task to be performed during runtime. A task may be synchronous or asynchronous."""

    def __init__(self, task_group: TaskGroup | None = None):
        """Construct a RuntimeTask. It may be associated with a TaskGroup.

        Args:
            task_group (TaskGroup | None, optional): The TaskGroup associated with this task. Defaults to None.
        """
        self._task_group = task_group

    @property
    def task_group(self) -> TaskGroup | None:
        """The TaskGroup associated with this task."""
        return self._task_group

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        pass


class InlineOpRuntimeTask(RuntimeTask):
    """An InlineOpRuntimeTask is a way of submitting arbitrary operations to a runtime that are defined
    in a lower-level style of IRON. This can be especially useful for tracing."""

    def __init__(
        self, fn: Callable, args: list, task_group: TaskGroup | None = None
    ):
        """Construct an InlineOpRuntimeTask.

        Args:
            fn (Callable): The function that will generate ops. It will be run within an MLIR module context.
            args (list): The arguments for the task: this should included objects such as Buffers used by the function.
            task_group (TaskGroup | None, optional): The TaskGroup to associated these operation with. Defaults to None.
        """
        # TODO: should validate somehow?
        self._fn = fn
        self._args = args
        RuntimeTask.__init__(self, task_group)

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
