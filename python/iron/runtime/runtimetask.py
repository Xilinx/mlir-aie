# runtimetask.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.

from ... import ir  # type: ignore

from ..resolvable import Resolvable
from ..worker import Worker


class RuntimeTask(Resolvable):
    pass


class RuntimeStartTask(RuntimeTask):
    def __init__(self, worker: Worker):
        self._worker = worker

    @property
    def worker(self) -> Worker:
        return self._worker

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        pass


class InlineOpRuntimeTask(RuntimeTask):
    def __init__(self, fn, args):
        # TODO: should validate somehow?
        self._fn = fn
        self._args = args

    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None:
        self._fn(*self._args)
