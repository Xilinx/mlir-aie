# test_hsa_runtime_selection.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-side selection/contract tests for the HSA runtime. No NPU dispatch."""

import os
import subprocess
import sys


def _run_import(env_overrides):
    env = dict(os.environ)
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run(
        [sys.executable, "-c", "import aie.utils"],
        env=env, capture_output=True, text=True,
    )


def test_invalid_iron_runtime_is_hard_error():
    res = _run_import({"IRON_RUNTIME": "bogus"})
    assert res.returncode != 0, res.stdout + res.stderr
    assert "Invalid IRON_RUNTIME" in res.stderr, res.stderr


def test_unset_iron_runtime_imports_cleanly():
    res = _run_import({"IRON_RUNTIME": None})
    assert res.returncode == 0, res.stdout + res.stderr


def test_auto_never_selects_hsa():
    res = _run_import(
        {
            "IRON_RUNTIME": "auto",
            "IRON_PRINT_BACKEND": "1",
        }
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "hsa" not in res.stdout.lower(), res.stdout


def test_iron_runtime_hsa_without_libhsa_raises():
    import aie.utils as u

    if u.has_hsa:
        import pytest
        pytest.skip("HSA is discoverable on this host; missing-HSA path untestable")
    res = _run_import(
        {
            "IRON_RUNTIME": "hsa",
            "HSA_RUNTIME_LIB": None,
            "HSA_RUNTIME_DIR": None,
            "ROCM_PATH": None,
            "LD_LIBRARY_PATH": "",
        }
    )
    assert res.returncode != 0, res.stdout + res.stderr
    assert "libhsa" in res.stderr.lower() or "hsa" in res.stderr.lower(), res.stderr
