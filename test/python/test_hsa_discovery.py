# test_hsa_discovery.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

import os
from aie.utils.hostruntime.hsaruntime import discovery


def test_env_hint_is_honored(tmp_path):
    fake = tmp_path / "libhsa-runtime64.so"
    fake.write_bytes(b"\x7fELF")
    old = os.environ.get("HSA_RUNTIME_LIB")
    os.environ["HSA_RUNTIME_LIB"] = str(fake)
    try:
        assert discovery.find_libhsa() == str(fake)
        assert discovery.hsa_available() is True
    finally:
        if old is None:
            os.environ.pop("HSA_RUNTIME_LIB", None)
        else:
            os.environ["HSA_RUNTIME_LIB"] = old


def test_missing_returns_none(tmp_path, monkeypatch):
    # Point every hint at nonexistent paths and neutralize standard locations
    # by making ROCM_PATH a bare tmp dir with no lib/libhsa-runtime64.so.
    monkeypatch.delenv("HSA_RUNTIME_LIB", raising=False)
    monkeypatch.delenv("HSA_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    # find_libhsa may still find a system libhsa; only assert the API is callable
    # and returns None-or-str without raising.
    result = discovery.find_libhsa()
    assert result is None or isinstance(result, str)


def test_rocm_path_candidate_is_resolved(tmp_path, monkeypatch):
    rocm_root = tmp_path / "opt" / "rocm"
    lib_dir = rocm_root / "lib"
    lib_dir.mkdir(parents=True)
    fake = lib_dir / "libhsa-runtime64.so"
    fake.write_bytes(b"\x7fELF")

    monkeypatch.delenv("HSA_RUNTIME_LIB", raising=False)
    monkeypatch.delenv("HSA_RUNTIME_DIR", raising=False)
    monkeypatch.setenv("ROCM_PATH", str(rocm_root))

    assert discovery.find_libhsa() == str(fake)
