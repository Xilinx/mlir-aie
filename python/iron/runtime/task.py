# task.py -*- Python -*-
#
# Copyright (C) 2024-2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
from ... import ir  # pyright: ignore[reportMissingImports, reportAttributeAccessIssue]

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
