# test_hsa_discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Discovery of a ROCm install: explicit root, pip wheel, then system."""

import pytest
from aie.utils.hostruntime.hsaruntime import discovery


@pytest.fixture(autouse=True)
def _no_ambient_rocm(monkeypatch):
    """Neutralize the host's own ROCm so tests see only what they set up."""
    for var in discovery._ENV_ROOT_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(discovery, "_SYSTEM_ROOTS", ())
    monkeypatch.setattr(discovery, "_wheel_root", lambda: None)


def _make_rocm_root(base, libname="libhsa-runtime64.so", with_header=False):
    lib_dir = base / "lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    lib = lib_dir / libname
    lib.write_bytes(b"\x7fELF")
    if with_header:
        hsa = base / "include" / "hsa"
        hsa.mkdir(parents=True, exist_ok=True)
        (hsa / "hsa.h").write_text("/* stub */")
    return lib


@pytest.mark.parametrize("var", discovery._ENV_ROOT_VARS)
def test_explicit_root_is_honored(tmp_path, monkeypatch, var):
    """The user-facing override names an install root, not a library file."""
    lib = _make_rocm_root(tmp_path / "rocm")
    monkeypatch.setenv(var, str(tmp_path / "rocm"))
    assert discovery.find_libhsa() == str(lib)
    assert discovery.hsa_available() is True


def test_explicit_root_beats_wheel_and_system(tmp_path, monkeypatch):
    """Priority: explicit root, then wheel, then system."""
    explicit = _make_rocm_root(tmp_path / "explicit")
    _make_rocm_root(tmp_path / "wheel")
    _make_rocm_root(tmp_path / "system")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "explicit"))
    monkeypatch.setattr(discovery, "_wheel_root", lambda: tmp_path / "wheel")
    monkeypatch.setattr(discovery, "_SYSTEM_ROOTS", (tmp_path / "system",))
    assert discovery.find_libhsa() == str(explicit)


def test_wheel_beats_system(tmp_path, monkeypatch):
    wheel = _make_rocm_root(tmp_path / "wheel")
    _make_rocm_root(tmp_path / "system")
    monkeypatch.setattr(discovery, "_wheel_root", lambda: tmp_path / "wheel")
    monkeypatch.setattr(discovery, "_SYSTEM_ROOTS", (tmp_path / "system",))
    assert discovery.find_libhsa() == str(wheel)


def test_system_root_is_last_resort(tmp_path, monkeypatch):
    lib = _make_rocm_root(tmp_path / "system")
    monkeypatch.setattr(discovery, "_SYSTEM_ROOTS", (tmp_path / "system",))
    assert discovery.find_libhsa() == str(lib)


def test_soname_only_install_is_found(tmp_path, monkeypatch):
    """A pip-installed ROCm ships no unversioned symlink, only the SONAME.

    TheRock's runtime wheels contain no symlinks at all, so requiring a bare
    libhsa-runtime64.so would miss every wheel install.
    """
    lib = _make_rocm_root(tmp_path / "wheel", libname="libhsa-runtime64.so.1")
    monkeypatch.setattr(discovery, "_wheel_root", lambda: tmp_path / "wheel")
    assert discovery.find_libhsa() == str(lib)


def test_soname_preferred_over_fully_versioned(tmp_path, monkeypatch):
    """Among versioned files, the SONAME is the one whose deps resolve."""
    root = tmp_path / "rocm"
    _make_rocm_root(root, libname="libhsa-runtime64.so.1.21.0")
    soname = _make_rocm_root(root, libname="libhsa-runtime64.so.1")
    monkeypatch.setenv("ROCM_PATH", str(root))
    assert discovery.find_libhsa() == str(soname)


def test_unversioned_preferred_when_present(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    _make_rocm_root(root, libname="libhsa-runtime64.so.1")
    unversioned = _make_rocm_root(root, libname="libhsa-runtime64.so")
    monkeypatch.setenv("ROCM_PATH", str(root))
    assert discovery.find_libhsa() == str(unversioned)


def test_root_without_the_library_is_skipped(tmp_path, monkeypatch):
    """An empty or unrelated root must fall through, not shadow later ones."""
    (tmp_path / "empty" / "lib").mkdir(parents=True)
    system = _make_rocm_root(tmp_path / "system")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(discovery, "_SYSTEM_ROOTS", (tmp_path / "system",))
    assert discovery.find_libhsa() == str(system)


def test_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "nonexistent"))
    assert discovery.find_libhsa() is None
    assert discovery.hsa_available() is False


def test_include_dir_follows_the_same_roots(tmp_path, monkeypatch):
    root = tmp_path / "rocm"
    _make_rocm_root(root, with_header=True)
    monkeypatch.setenv("ROCM_PATH", str(root))
    assert discovery.find_hsa_include_dir() == str((root / "include").resolve())


def test_wheel_root_is_none_when_not_installed(monkeypatch):
    """_wheel_root must return None (not raise) when no ROCm wheel is present."""
    monkeypatch.setattr(discovery.importlib.util, "find_spec", lambda name: None)
    assert discovery._wheel_root() is None
