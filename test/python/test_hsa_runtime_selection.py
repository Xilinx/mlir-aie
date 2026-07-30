# test_hsa_runtime_selection.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Host-side selection/contract tests for the HSA runtime. No NPU dispatch.

The HSA backend is selected via the shared ``NPU_RUNTIME`` env var (xrt|hrx|hsa|
auto). These tests cover the negative-behavior contract: an invalid value is a
hard error, HSA is opt-in (``auto`` never selects it), and ``NPU_RUNTIME=hsa``
without libhsa raises.
"""

import os
import subprocess
import sys


def _run_import(env_overrides, body="import aie.utils"):
    env = dict(os.environ)
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return subprocess.run(
        [sys.executable, "-c", body],
        env=env,
        capture_output=True,
        text=True,
    )


def test_invalid_npu_runtime_is_hard_error():
    """An explicitly invalid NPU_RUNTIME must fail the import loudly."""
    res = _run_import({"NPU_RUNTIME": "bogus"})
    assert res.returncode != 0, res.stdout + res.stderr
    assert "Invalid NPU_RUNTIME" in res.stderr, res.stderr


def test_unset_npu_runtime_imports_cleanly():
    """Unset NPU_RUNTIME defaults to 'auto' and must import fine."""
    res = _run_import({"NPU_RUNTIME": None})
    assert res.returncode == 0, res.stdout + res.stderr


def test_auto_never_selects_hsa():
    """auto must resolve to XRT or CPU, never HSA (opt-in only)."""
    res = _run_import(
        {"NPU_RUNTIME": "auto"},
        # DEFAULT_TENSOR_CLASS lives in tensor_factory; aie.utils re-exports the
        # factory functions but not the class itself.
        body=(
            "from aie.utils.tensor_factory import DEFAULT_TENSOR_CLASS; "
            "print(DEFAULT_TENSOR_CLASS.__name__)"
        ),
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "HSATensor" not in res.stdout, res.stdout


def test_npu_runtime_hsa_without_libhsa_raises():
    """NPU_RUNTIME=hsa with no libhsa must raise ImportError.

    Only meaningful when HSA is not discoverable on this host; if it is, the
    contract can't be exercised, so skip.
    """
    import aie.utils as u

    if u.has_hsa:
        import pytest

        pytest.skip("HSA is discoverable on this host; missing-HSA path untestable")
    res = _run_import(
        {
            "NPU_RUNTIME": "hsa",
            "ROCM_PATH": None,
            "ROCM_HOME": None,
            "LD_LIBRARY_PATH": "",
        }
    )
    assert res.returncode != 0, res.stdout + res.stderr
    assert "libhsa" in res.stderr.lower() or "hsa" in res.stderr.lower(), res.stderr
