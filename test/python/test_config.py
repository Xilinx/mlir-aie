# test_config.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""Unit tests for aie.utils.config's AIECC_PATH override."""

import os

import pytest

import aie.utils.config as config


def test_aiecc_path_env_override(tmp_path, monkeypatch):
    fake_aiecc = tmp_path / "aiecc"
    fake_aiecc.touch()
    monkeypatch.setenv("AIECC_PATH", str(fake_aiecc))
    assert config.aiecc_path() == str(fake_aiecc)


def test_aiecc_path_env_override_missing_file_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("AIECC_PATH", str(tmp_path / "does-not-exist"))
    with pytest.raises(RuntimeError, match="AIECC_PATH"):
        config.aiecc_path()


def test_host_cxx_skips_peano_on_path(tmp_path, monkeypatch):
    peano = tmp_path / "peano"
    host = tmp_path / "host"
    for directory in (peano / "bin", host):
        directory.mkdir(parents=True)
        compiler = directory / config._executable_name("clang++")
        compiler.touch(mode=0o755)
    monkeypatch.setattr(config.config, "peano_install_dir", str(peano))
    monkeypatch.delenv("CXX", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([str(peano / "bin"), str(host)]))
    assert config.host_cxx_path() == str(host / config._executable_name("clang++"))

    monkeypatch.setenv("PATH", str(peano / "bin"))
    with pytest.raises(RuntimeError, match="Could not find a host C\\+\\+ compiler"):
        config.host_cxx_path()


def test_host_cxx_honors_explicit_override(tmp_path, monkeypatch):
    compiler = tmp_path / config._executable_name("custom-cxx")
    compiler.touch(mode=0o755)
    monkeypatch.setenv("CXX", str(compiler))
    assert config.host_cxx_path() == str(compiler)

    monkeypatch.setenv("CXX", str(tmp_path / "missing"))
    with pytest.raises(RuntimeError, match="CXX is set"):
        config.host_cxx_path()
