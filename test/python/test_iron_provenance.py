# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import sys
from types import SimpleNamespace

import pytest
from aie import ir
from aie.iron import DeviceConfiguration, Program, Runtime
from aie.iron.device import NPU2Col1


@pytest.mark.parametrize("compose", [False, True])
@pytest.mark.parametrize("wheel", [False, True])
def test_program_retains_bindings_identity(monkeypatch, compose, wheel):
    monkeypatch.setitem(
        sys.modules,
        "aie._build_version",
        SimpleNamespace(version="1.2.3.dev4", revision="a" * 40, dirty=True),
    )
    monkeypatch.setitem(
        sys.modules,
        "aie._version",
        SimpleNamespace(__version__="1.2.3", __commit__="aaaaaaa") if wheel else None,
    )
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
        assert ir.StringAttr(attr["version"]).value == (
            "1.2.3" if wheel else "1.2.3.dev4"
        )
        assert ir.StringAttr(attr["revision"]).value == "a" * 40
        assert ir.BoolAttr(attr["dirty"]).value
        assert saved.operation.verify()


def test_unversioned_bindings_still_mark_iron(monkeypatch):
    monkeypatch.setitem(sys.modules, "aie._build_version", None)
    monkeypatch.setitem(sys.modules, "aie._version", None)
    module = Program(NPU2Col1(), Runtime(lambda: None, [])).resolve_program()
    assert len(ir.DictAttr(module.operation.attributes["iron.provenance"])) == 0


def test_legacy_wheel_identity(monkeypatch):
    monkeypatch.setitem(sys.modules, "aie._build_version", None)
    monkeypatch.setitem(
        sys.modules,
        "aie._version",
        SimpleNamespace(__version__="1.0", __commit__="abcdef0"),
    )
    module = Program(NPU2Col1(), Runtime(lambda: None, [])).resolve_program()
    attr = ir.DictAttr(module.operation.attributes["iron.provenance"])
    assert ir.StringAttr(attr["version"]).value == "1.0"
    assert ir.StringAttr(attr["revision"]).value == "abcdef0"
    assert "dirty" not in attr
