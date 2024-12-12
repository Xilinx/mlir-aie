# task.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from typing import Callable

from ... import ir  # type: ignore

from ..resolvable import Resolvable
from ..worker import Worker
from .taskgroup import RuntimeTaskGroup


class RuntimeTask(Resolvable):
    """A RuntimeTask is a task to be performed during runtime. A task may be synchronous or asynchronous."""

    def __init__(self, task_group: RuntimeTaskGroup | None = None):
        """Construct a RuntimeTask. It may be associated with a RuntimeTaskGroup.

        Args:
            task_group (RuntimeTaskGroup | None, optional): The TaskGroup associated with this task. Defaults to None.
        """
        self._task_group = task_group

    @property
    def task_group(self) -> RuntimeTaskGroup | None:
        """The RuntimeTaskGroup associated with this task."""
        return self._task_group

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        pass


class FinishTaskGroupTask(RuntimeTask):
    """A Task indicating the task_group should be finished(), which generally means it's tasks should be waited on and freed."""

    def __init__(self, task_group: RuntimeTaskGroup):
        RuntimeTask.__init__(self, task_group)


class RuntimeStartTask(RuntimeTask):
    """A StartTask is a placeholder to indicated that a Worker should be started"""

    def __init__(self, worker: Worker, task_group: RuntimeTaskGroup | None = None):
        self._worker = worker
        RuntimeTask.__init__(self, task_group)

    @property
    def worker(self) -> Worker:
        """The worker associated with this RuntimeStartTask"""
        return self._worker


class InlineOpRuntimeTask(RuntimeTask):
    """An InlineOpRuntimeTask is a way of submitting arbitrary operations to a runtime that are defined
    in a lower-level style of IRON. This can be especially useful for tracing."""

    def __init__(
        self, fn: Callable, args: list, task_group: RuntimeTaskGroup | None = None
    ):
        """Construct an InlineOpRuntimeTask.

        Args:
            fn (Callable): The function that will generate ops. It will be run within an MLIR module context.
            args (list): The arguments for the task: this should included objects such as Buffers used by the function.
            task_group (RuntimeTaskGroup | None, optional): The TaskGroup to associated these operation with. Defaults to None.
        """
        # TODO: should validate somehow?
        self._fn = fn
        self._args = args
        RuntimeTask.__init__(self, task_group)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._fn(*self._args)
