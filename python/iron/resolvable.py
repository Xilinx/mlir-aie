# resolvable.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc.


from abc import ABC, abstractmethod

from .. import ir  # type: ignore


class Resolvable(ABC):
    @abstractmethod
    def resolve(
        cls,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...
