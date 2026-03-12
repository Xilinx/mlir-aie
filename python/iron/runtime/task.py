# task.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.
from typing import Callable

from ... import ir  # type: ignore
from ...dialects.aiex import npu_rtp_write

from ..resolvable import Resolvable
from ..worker import Worker
from .data import RuntimeScalar
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


class RtpWriteTask(RuntimeTask):
    """A RuntimeTask that writes a value to a Runtime Parameter (RTP) buffer."""

    def __init__(self, buf_name: str, index: int, value):
        """Construct an RtpWriteTask.

        Args:
            buf_name (str): The name of the RTP buffer.
            index (int): The index within the RTP buffer to write.
            value: The value to write (int or RuntimeScalar op).
        """
        self._buf_name = buf_name
        self._index = index
        self._value = value
        RuntimeTask.__init__(self, None)

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        val = self._value.op if isinstance(self._value, RuntimeScalar) else self._value
        npu_rtp_write(self._buf_name, self._index, val)
