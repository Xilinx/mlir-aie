# test_hrx_runtime_selection.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""Negative / contract tests for the HRX runtime selection and guards.

These are host-side unit tests: none of them dispatch on the NPU. They cover the
review's "negative behavior" gap:
  * an invalid IRON_RUNTIME is a hard error (not a silent fallback);
  * IRON_RUNTIME=hrx with no libhrx raises (only asserted when HRX is absent);
  * HRX rejects trace_config instead of silently ignoring it;
  * the IRON_HRX_TIMEOUT parser and device-generation detection helpers.
"""

from __future__ import annotations

import os
import subprocess
import sys

import pytest


def _run_import(env_overrides: dict) -> subprocess.CompletedProcess:
    """Import aie.utils in a fresh interpreter with the given env overrides."""
    env = dict(os.environ)
    env.update({k: v for k, v in env_overrides.items() if v is not None})
    for k, v in env_overrides.items():
        if v is None:
            env.pop(k, None)
    return subprocess.run(
        [sys.executable, "-c", "import aie.utils"],
        env=env,
        capture_output=True,
        text=True,
    )


def test_invalid_iron_runtime_is_hard_error():
    """An explicitly invalid IRON_RUNTIME must fail the import loudly."""
    res = _run_import({"IRON_RUNTIME": "bogus"})
    assert res.returncode != 0, res.stdout + res.stderr
    assert "Invalid IRON_RUNTIME" in res.stderr, res.stderr


def test_unset_iron_runtime_imports_cleanly():
    """Unset IRON_RUNTIME defaults to 'auto' and must import fine."""
    res = _run_import({"IRON_RUNTIME": None})
    assert res.returncode == 0, res.stdout + res.stderr


def test_iron_runtime_hrx_without_libhrx_raises():
    """IRON_RUNTIME=hrx with no libhrx must raise ImportError.

    Only meaningful when HRX is not discoverable on this host; if it is, the
    contract can't be exercised, so skip.
    """
    import aie.utils as u

    if u.has_hrx:
        pytest.skip("HRX is discoverable on this host; missing-HRX path untestable")
    res = _run_import(
        {
            "IRON_RUNTIME": "hrx",
            "HRX_DIR": None,
            "HRX_BUILD": None,
            "LIBHRX_DIR": None,
            "HRX_LIBHRX": None,
            "LD_LIBRARY_PATH": "",
        }
    )
    assert res.returncode != 0, res.stdout + res.stderr
    assert "libhrx" in res.stderr.lower(), res.stderr


def _make_hrx_runtime_without_init():
    """Construct an HRXHostRuntime bypassing __init__ (no device / libhrx needed)."""
    hrt = pytest.importorskip(
        "aie.utils.hostruntime.hrxruntime.hostruntime",
        reason="hrxruntime package not importable",
    )
    return hrt, hrt.HRXHostRuntime.__new__(hrt.HRXHostRuntime)


def test_hrx_run_rejects_trace_config():
    """HRXHostRuntime.run() must reject a trace_config (parity with C++)."""
    hrt, rt = _make_hrx_runtime_without_init()
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    handle = hrt.HRXKernelHandle(
        executable=None,
        export_ordinal=0,
        kernel_name="MLIR_AIE",
        xclbin_path="x",
        insts_path="i",
    )

    class _Trace:
        pass

    with pytest.raises(HostRuntimeError, match="[Tt]race"):
        rt.run(handle, [], trace_config=_Trace())


def test_hrx_load_and_run_rejects_trace_config():
    """HRXHostRuntime.load_and_run() must reject before mutating run_args."""
    hrt, rt = _make_hrx_runtime_without_init()
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    class _Kernel:
        trace_config = object()

    run_args = [1, 2, 3]
    with pytest.raises(HostRuntimeError, match="[Tt]race"):
        rt.load_and_run(_Kernel(), run_args)
    # run_args must be untouched on the error path.
    assert run_args == [1, 2, 3]


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, 0.0),
        ("", 0.0),
        ("0", 0.0),
        ("2.5", 2.5),
        ("-3", 0.0),
        ("notanumber", 0.0),
    ],
)
def test_hrx_sync_timeout_parsing(monkeypatch, value, expected):
    pkg = pytest.importorskip("aie.utils.hostruntime.hrxruntime")
    if value is None:
        monkeypatch.delenv("IRON_HRX_TIMEOUT", raising=False)
    else:
        monkeypatch.setenv("IRON_HRX_TIMEOUT", value)
    assert pkg._hrx_sync_timeout_s() == expected


def test_hrx_device_gen_env_override(monkeypatch):
    hrt = pytest.importorskip("aie.utils.hostruntime.hrxruntime.hostruntime")
    monkeypatch.setenv("IRON_HRX_DEVICE", "npu1")
    assert hrt._detect_hrx_device_gen() == "npu1"


def test_hrx_device_gen_returns_known_value(monkeypatch):
    """Without an override, detection returns one of the known generations."""
    hrt = pytest.importorskip("aie.utils.hostruntime.hrxruntime.hostruntime")
    monkeypatch.delenv("IRON_HRX_DEVICE", raising=False)
    assert hrt._detect_hrx_device_gen() in ("npu1", "npu2")
