# test_peano_provenance.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""A benchmark row names the Peano that compiled it, not the installed wheel.

``PEANO_INSTALL_DIR`` selects the compiler, so a study that sweeps nightlies
through it must see each nightly in its rows. The fake compiler is a shebang
script placed where the resolver looks, so the resolver and the subprocess
both run for real.
"""

import os
import subprocess

import aie.utils.config as config
import pytest
from aie.utils.benchmark import peano_version, provenance

posix_only = pytest.mark.skipif(
    os.name == "nt", reason="the fake compiler is a shebang shell script"
)


def _fake_peano(root, banner):
    bin_dir = root / "bin"
    bin_dir.mkdir(parents=True)
    cxx = bin_dir / "clang++"
    cxx.write_text(f"#!/bin/sh\necho '{banner}'\n")
    cxx.chmod(0o755)
    return root


@posix_only
def test_provenance_follows_the_selected_compiler(tmp_path, monkeypatch):
    peano = _fake_peano(
        tmp_path / "peano",
        "clang version 22.0.0 (https://github.com/Xilinx/llvm-aie "
        "6dc4d6dd71558b553448df97254408ec3d800eb5)",
    )
    monkeypatch.setattr(config.config, "peano_install_dir", str(peano))
    assert peano_version() == "22.0.0+6dc4d6dd"
    assert "peano 22.0.0+6dc4d6dd" in provenance()


@posix_only
def test_an_unrecognized_banner_is_kept_whole(tmp_path, monkeypatch):
    peano = _fake_peano(tmp_path / "peano", "clang version 23.0.0git")
    monkeypatch.setattr(config.config, "peano_install_dir", str(peano))
    assert peano_version() == "clang version 23.0.0git"


def test_a_missing_compiler_is_left_out(tmp_path, monkeypatch):
    monkeypatch.setattr(config.config, "peano_install_dir", str(tmp_path))
    assert peano_version() is None
    assert not any(f.startswith("peano ") for f in provenance().split(" | "))


def test_the_installed_compiler_reports_its_own_commit():
    banner = subprocess.run(
        [config.peano_cxx_path(), "--version"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.splitlines()[0]
    version = peano_version()
    assert version is not None
    number, commit = version.split("+")
    assert f"clang version {number} " in banner
    assert commit in banner
