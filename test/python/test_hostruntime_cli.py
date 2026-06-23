# test_hostruntime_cli.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2022-2026 Advanced Micro Devices, Inc. All rights reserved.

# RUN: %pytest %s

from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from aie.utils.hostruntime.argparse import add_compile_args, device_from_args
from aie.utils.hostruntime import cli


class _Design:
    def __init__(self):
        self.compile_kwargs = None
        self.compile_options = None

    def specialize(self, **kwargs):
        self.compile_kwargs = kwargs
        return self

    def compile(self, **kwargs):
        self.compile_options = kwargs


def _automatic_options(*, emit_mlir=False, xclbin_path=None, insts_path=None):
    return SimpleNamespace(
        dev=None,
        emit_mlir=emit_mlir,
        xclbin_path=xclbin_path,
        insts_path=insts_path,
    )


def _mock_runtime_target(monkeypatch):
    import aie.utils as utils
    import aie.utils.hostruntime as hostruntime

    runtime_device = object()
    set_calls = []
    monkeypatch.setattr(utils, "get_current_device", lambda: runtime_device)
    monkeypatch.setattr(
        utils,
        "ensure_current_device",
        lambda: pytest.fail("automatic CLI selection must not bind the runtime device"),
    )
    monkeypatch.setattr(hostruntime, "set_current_device", set_calls.append)
    monkeypatch.setattr(cli, "_runtime_device_name", lambda device: "npu2")
    return runtime_device, set_calls


def test_compile_args_leave_target_automatic_by_default():
    parser = argparse.ArgumentParser()
    add_compile_args(parser, with_emit_mlir=True)

    assert parser.parse_args([]).dev is None
    assert parser.parse_args(["--dev", "npu2"]).dev == "npu2"

    fixed_parser = argparse.ArgumentParser()
    add_compile_args(fixed_parser, default_dev="npu2")
    assert fixed_parser.parse_args([]).dev == "npu2"


def test_device_from_args_defers_an_omitted_target():
    assert device_from_args(SimpleNamespace(dev=None)) is None


def test_run_design_cli_uses_runtime_family_with_the_declared_profile(monkeypatch):
    _runtime_device, set_calls = _mock_runtime_target(monkeypatch)
    profile_device = object()
    opts = _automatic_options()
    observed = []

    cli.run_design_cli(
        _Design(),
        opts,
        compile_kwargs={},
        run_and_verify=lambda resolved_opts: observed.append(resolved_opts.dev),
        device=lambda resolved_opts: (
            observed.append(f"selector:{resolved_opts.dev}") or profile_device
        ),
    )

    assert set_calls == [profile_device]
    assert opts.dev == "npu2"
    assert observed == ["selector:npu2", "npu2"]


def test_run_design_cli_uses_the_default_profile_for_an_automatic_family(monkeypatch):
    _runtime_device, set_calls = _mock_runtime_target(monkeypatch)
    import aie.iron.device as iron_device

    profile_device = object()
    selected_names = []
    monkeypatch.setattr(
        iron_device,
        "from_name",
        lambda name: selected_names.append(name) or profile_device,
    )
    opts = _automatic_options()

    cli.run_design_cli(
        _Design(),
        opts,
        compile_kwargs={},
        run_and_verify=lambda _: None,
    )

    assert selected_names == ["npu2"]
    assert set_calls == [profile_device]


def test_run_design_cli_respects_an_explicit_target(monkeypatch):
    import aie.utils as utils
    import aie.utils.hostruntime as hostruntime

    target_device = object()
    set_calls = []
    monkeypatch.setattr(
        utils,
        "get_current_device",
        lambda: pytest.fail("explicit targets must not probe the runtime"),
    )
    monkeypatch.setattr(
        utils,
        "ensure_current_device",
        lambda: pytest.fail("explicit targets must not bind the runtime device"),
    )
    monkeypatch.setattr(hostruntime, "set_current_device", set_calls.append)
    opts = SimpleNamespace(
        dev="npu2", emit_mlir=False, xclbin_path=None, insts_path=None
    )

    cli.run_design_cli(
        _Design(),
        opts,
        compile_kwargs={},
        run_and_verify=lambda _: None,
        device=lambda _: target_device,
    )

    assert set_calls == [target_device]


def test_run_design_cli_uses_runtime_target_for_local_compile_only(monkeypatch):
    _runtime_device, set_calls = _mock_runtime_target(monkeypatch)
    profile_device = object()
    opts = _automatic_options(xclbin_path="out.xclbin", insts_path="out.insts")
    design = _Design()

    cli.run_design_cli(
        design,
        opts,
        compile_kwargs={},
        device=lambda resolved_opts: (
            profile_device
            if resolved_opts.dev == "npu2"
            else pytest.fail("wrong target")
        ),
    )

    assert set_calls == [profile_device]
    assert opts.dev == "npu2"
    assert design.compile_options == {
        "xclbin_path": "out.xclbin",
        "inst_path": "out.insts",
    }


def test_emit_mlir_requires_an_explicit_target():
    opts = _automatic_options(emit_mlir=True)

    with pytest.raises(SystemExit, match=r"--emit-mlir requires an explicit target"):
        cli.run_design_cli(_Design(), opts, compile_kwargs={})


def test_emit_mlir_accepts_a_concrete_device_without_dev(monkeypatch):
    """A programmatic fixed target is sufficient for offline MLIR emission."""
    import aie.utils as utils
    import aie.utils.hostruntime as hostruntime

    target_device = object()
    set_calls = []
    emitted = []
    monkeypatch.setattr(
        utils,
        "get_current_device",
        lambda: pytest.fail("a concrete target must not probe the runtime"),
    )
    monkeypatch.setattr(
        utils,
        "ensure_current_device",
        lambda: pytest.fail("a concrete target must not bind the runtime device"),
    )
    monkeypatch.setattr(hostruntime, "set_current_device", set_calls.append)
    opts = _automatic_options(emit_mlir=True)

    cli.run_design_cli(
        _Design(),
        opts,
        compile_kwargs={},
        device=target_device,
        emit_mlir=lambda resolved_opts: emitted.append(resolved_opts),
    )

    assert set_calls == [target_device]
    assert emitted == [opts]


def test_compile_only_without_runtime_requires_an_explicit_target(monkeypatch):
    import aie.utils as utils

    monkeypatch.setattr(utils, "get_current_device", lambda: None)
    opts = _automatic_options(xclbin_path="out.xclbin", insts_path="out.insts")

    with pytest.raises(SystemExit, match=r"compile-only mode requires --dev"):
        cli.run_design_cli(_Design(), opts, compile_kwargs={})
