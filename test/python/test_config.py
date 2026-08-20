# test_config.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""Unit tests for aie.utils.config's AIECC_PATH override."""

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
