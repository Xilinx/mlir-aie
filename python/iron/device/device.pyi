# device.pyi -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
"""Static type stubs for the per-target Device subclasses generated at
runtime in ``device.py`` (the ``for device in AIEDevice: create_class(...)``
loop at the bottom of that module).

The stubs let IDEs, mypy, and ``from aie.iron.device.device import NPU2``-style
imports resolve cleanly.  They do NOT affect runtime behaviour; the real
classes are still synthesised at import time.
"""

from .. import ir
from .device import Device

class XCVC1902(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class XCVE2302(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class XCVE2802(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU1(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU1Col1(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU1Col2(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU1Col3(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col1(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col2(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col3(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col4(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col5(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col6(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...

class NPU2Col7(Device):
    def __init__(self) -> None: ...
    def resolve(
        self,
        loc: ir.Location | None = None,
        ip: ir.InsertionPoint | None = None,
    ) -> None: ...
