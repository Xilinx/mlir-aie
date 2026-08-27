# test_hrx_discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Discovery of libhrx: env hints, pip layout, then FindHRX roots."""

import sys
from pathlib import Path

import pytest
from aie.utils.hostruntime.hrxruntime import discovery


@pytest.fixture(autouse=True)
def _no_ambient_hrx(monkeypatch):
    """Neutralize the host's own HRX so tests see only what they set up."""
    for var in ("HRX_LIBHRX", "LIBHRX_DIR", "HRX_DIR"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(discovery, "_HRX_ROOT_CANDIDATES", [])
    monkeypatch.setattr(discovery, "_SYSTEM_LIBS", ())
    monkeypatch.setattr(discovery, "_pip_roots", lambda: ())


def _lib_name() -> str:
    return discovery._LIBHRX_NAMES[0]


def _make_lib(directory: Path, name: str | None = None) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    lib = directory / (name or _lib_name())
    lib.write_bytes(b"stub")
    return lib


def _make_pip_package(site: Path, name: str = "hrx_wheel") -> Path:
    """Conventional pip layout: ``<site>/<name>/{lib,bin}/libhrx``."""
    pkg = site / name
    sub = "bin" if discovery._IS_WINDOWS else "lib"
    lib = _make_lib(pkg / sub)
    (pkg / "__init__.py").write_text("raise RuntimeError('must not import')\n")
    return lib


def test_explicit_lib_path_is_honored(tmp_path, monkeypatch):
    lib = _make_lib(tmp_path / "opt")
    monkeypatch.setenv("HRX_LIBHRX", str(lib))
    assert discovery.find_libhrx() == str(lib)
    assert discovery.hrx_available() is True


def test_libhrx_dir_beats_wheel(tmp_path, monkeypatch):
    hinted = _make_lib(tmp_path / "hint")
    site = tmp_path / "site"
    _make_pip_package(site)
    monkeypatch.setenv("LIBHRX_DIR", str(tmp_path / "hint"))
    monkeypatch.setattr(discovery, "_pip_roots", lambda: (site,))
    assert Path(discovery.find_libhrx()).resolve() == hinted.resolve()


def test_wheel_beats_sibling_and_system(tmp_path, monkeypatch):
    site = tmp_path / "site"
    wheel = _make_pip_package(site)
    sibling = tmp_path / "sibling"
    sibling.mkdir()
    (sibling / "include" / "hrx").mkdir(parents=True)
    (sibling / "include" / "hrx" / "hrx_runtime.h").write_text("/* stub */")
    _make_lib(sibling / "lib")
    system = _make_lib(tmp_path / "system")
    monkeypatch.setattr(discovery, "_pip_roots", lambda: (site,))
    monkeypatch.setattr(discovery, "_HRX_ROOT_CANDIDATES", [sibling])
    monkeypatch.setattr(discovery, "_SYSTEM_LIBS", (str(system),))
    assert Path(discovery.find_libhrx()).resolve() == wheel.resolve()


def test_system_lib_is_last_resort(tmp_path, monkeypatch):
    lib = _make_lib(tmp_path / "system")
    monkeypatch.setattr(discovery, "_SYSTEM_LIBS", (str(lib),))
    assert discovery.find_libhrx() == str(lib)


def test_missing_returns_none():
    assert discovery.find_libhrx() is None
    assert discovery.hrx_available() is False


def test_pip_package_without_libhrx_falls_through(tmp_path, monkeypatch):
    site = tmp_path / "site"
    pkg = site / "unrelated"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    system = _make_lib(tmp_path / "system")
    monkeypatch.setattr(discovery, "_pip_roots", lambda: (site,))
    monkeypatch.setattr(discovery, "_SYSTEM_LIBS", (str(system),))
    assert discovery.find_libhrx() == str(system)


def test_wheel_libhrx_is_none_when_not_installed():
    assert discovery._wheel_libhrx() is None


def test_pip_layout_does_not_import_the_package(tmp_path, monkeypatch):
    site = tmp_path / "site"
    lib = _make_pip_package(site, name="hrx_wheel")
    monkeypatch.setattr(discovery, "_pip_roots", lambda: (site,))
    found = discovery._wheel_libhrx()
    assert found is not None
    assert Path(found).resolve() == lib.resolve()
    assert "hrx_wheel" not in sys.modules
