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
    def __init__(self, task_group: RuntimeTaskGroup | None = None):
        self._task_group = task_group

    @property
    def task_group(self) -> RuntimeTaskGroup | None:
        return self._task_group

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        pass


class FinishTaskGroupTask(RuntimeTask):
    def __init__(self, task_group: RuntimeTaskGroup):
        RuntimeTask.__init__(self, task_group)


class RuntimeStartTask(RuntimeTask):
    def __init__(self, worker: Worker, task_group: RuntimeTaskGroup | None = None):
        self._worker = worker
        RuntimeTask.__init__(self, task_group)

    @property
    def worker(self) -> Worker:
        return self._worker


class InlineOpRuntimeTask(RuntimeTask):
    def __init__(
        self, fn: Callable, args: list, task_group: RuntimeTaskGroup | None = None
    ):
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
