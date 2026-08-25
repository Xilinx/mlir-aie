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


def test_soname_preferred_over_fully_versioned(tmp_path, monkeypatch):
    """A versioned-only install is found, and the SONAME is the one picked.

    TheRock's runtime wheels contain no symlinks at all, so requiring a bare
    libhsa-runtime64.so would miss every wheel install. Among the versioned
    files the SONAME is the one whose dependencies resolve.
    """
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


def _discovery_as_platform(monkeypatch, system):
    """Re-import discovery privately, as if running on ``system``.

    The platform gate is evaluated at import time, so it can only be exercised
    by re-executing the module. This loads a throwaway copy under its own name
    rather than reloading the shared one, so nothing leaks into other tests.
    """
    import importlib.util
    import platform

    monkeypatch.setattr(platform, "system", lambda: system)
    spec = importlib.util.spec_from_file_location("_disc_probe", discovery.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_unsupported_platform_finds_nothing(tmp_path, monkeypatch):
    """Off Linux there is no library layout to probe, so discovery declines.

    ROCR's AIE agent is Linux-only. Discovery must report "no ROCm" even with a
    plausible-looking tree in place, rather than matching a name that platform
    would never use -- and ``/opt/rocm`` must not be probed there either.
    """
    _make_rocm_root(tmp_path / "rocm")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    windows = _discovery_as_platform(monkeypatch, "Windows")

    assert windows._LIBHSA_PATTERNS == (), "no library names to probe off Linux"
    assert windows._SYSTEM_ROOTS == (), "/opt/rocm is POSIX-only"
    assert windows.find_libhsa() is None
    assert windows.hsa_available() is False


def test_supported_platform_probes_the_linux_layout(tmp_path, monkeypatch):
    """The same gate must actually admit Linux, or the backend never loads."""
    lib = _make_rocm_root(tmp_path / "rocm")
    monkeypatch.setenv("ROCM_PATH", str(tmp_path / "rocm"))
    linux = _discovery_as_platform(monkeypatch, "Linux")

    assert linux._SYSTEM_ROOTS, "/opt/rocm must still be probed on Linux"
    assert linux.find_libhsa() == str(lib)
