# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import aie
import pytest
from aie import ir
from aie.iron import DeviceConfiguration, Program, Runtime
from aie.iron.device import NPU2Col1


@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("versioned", [False, True])
def test_program_retains_existing_version(monkeypatch, compose, versioned):
    for name, value in (("__version__", "1.2.3"), ("__commit__", "abcdef0")):
        if versioned:
            monkeypatch.setattr(aie, name, value, raising=False)
        else:
            monkeypatch.delattr(aie, name, raising=False)
    rt = Runtime(lambda: None, [])
    program = (
        Program.compose(
            [DeviceConfiguration("main", NPU2Col1(), runtimes=[rt])], entry=rt
        )
        if compose
        else Program(NPU2Col1(), rt)
    )
    module = program.resolve_program()
    with module.context:
        saved = ir.Module.parse(str(module))
        attr = ir.DictAttr(saved.operation.attributes["iron.provenance"])
        values = {field.name: ir.StringAttr(field.attr).value for field in attr}
        assert values == (
            {"version": "1.2.3", "revision": "abcdef0"} if versioned else {}
        )
        assert saved.operation.verify()
