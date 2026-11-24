# RUN: %pytest %s
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""The environment LitConfigHelper.setup_standard_environment gives lit tests."""

import os
from pathlib import Path
from types import SimpleNamespace

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]


class _LLVMConfig:
    """The two lit.llvm.config.LLVMConfig methods the helper calls."""

    def __init__(self, config):
        self.config = config

    def with_environment(self, name, value, append_path=False):
        self.config.environment[name] = value

    def with_system_environment(self, names, append_path=False):
        for name in [names] if isinstance(names, str) else names:
            if os.environ.get(name):
                self.with_environment(name, os.environ[name], append_path)


@pytest.fixture
def helper(monkeypatch):
    monkeypatch.syspath_prepend(str(_REPO_ROOT / "python"))
    from aie_lit_utils.lit_config_helpers import LitConfigHelper

    return LitConfigHelper


def _environment(helper, **config_attrs):
    config = SimpleNamespace(environment={}, **config_attrs)
    helper.setup_standard_environment(
        _LLVMConfig(config), config, os.path.join("obj", "root"), ""
    )
    return config.environment


@pytest.mark.parametrize("name", ["AIECC_JOBS", "AIE_KERNEL_COMPILE_JOBS"])
def test_compile_job_caps_reach_tests(helper, monkeypatch, name):
    monkeypatch.setenv(name, "3")
    assert _environment(helper)[name] == "3"
    monkeypatch.delenv(name)
    assert name not in _environment(helper)


def _pythonpath(helper, **config_attrs):
    return _environment(helper, **config_attrs)["PYTHONPATH"].split(os.pathsep)


def test_configured_xrt_python_dir_follows_the_ambient_pythonpath(
    helper, monkeypatch, tmp_path
):
    xrt = tmp_path / "xrt"
    (xrt / "python").mkdir(parents=True)
    sourced = str(tmp_path / "sourced_xrt" / "python")
    monkeypatch.setenv("PYTHONPATH", sourced)
    assert _pythonpath(helper, xrt_dir=str(xrt))[1:] == [
        sourced,
        str(xrt / "python"),
    ]


def test_configured_xrt_python_dir_is_added_once(helper, monkeypatch, tmp_path):
    xrt_python = tmp_path / "xrt" / "python"
    xrt_python.mkdir(parents=True)
    monkeypatch.setenv("PYTHONPATH", str(xrt_python))
    assert _pythonpath(helper, xrt_dir=str(tmp_path / "xrt"))[1:] == [str(xrt_python)]


@pytest.mark.parametrize("xrt_dir", [None, "", "missing"])
def test_no_xrt_python_dir_without_an_xrt_install(
    helper, monkeypatch, tmp_path, xrt_dir
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    monkeypatch.chdir(tmp_path)
    (tmp_path / "python").mkdir()
    attrs = {} if xrt_dir is None else {"xrt_dir": xrt_dir}
    assert _pythonpath(helper, **attrs) == [os.path.join("obj", "root", "python")]
