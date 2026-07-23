# test_hrx_runtime_selection.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s

"""Negative / contract tests for the HRX runtime selection and guards.

These are host-side unit tests: none of them dispatch on the NPU. They cover the
review's "negative behavior" gap:
  * an invalid NPU_RUNTIME is a hard error (not a silent fallback);
  * NPU_RUNTIME=hrx with no libhrx raises (only asserted when HRX is absent);
  * HRX rejects trace_config instead of silently ignoring it;
  * the IRON_HRX_TIMEOUT parser and device-generation detection helpers.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time

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
    """An explicitly invalid NPU_RUNTIME must fail the import loudly."""
    res = _run_import({"NPU_RUNTIME": "bogus"})
    assert res.returncode != 0, res.stdout + res.stderr
    assert "Invalid NPU_RUNTIME" in res.stderr, res.stderr


def test_unset_iron_runtime_imports_cleanly():
    """Unset NPU_RUNTIME defaults to 'auto' and must import fine."""
    res = _run_import({"NPU_RUNTIME": None})
    assert res.returncode == 0, res.stdout + res.stderr


def test_iron_runtime_hrx_without_libhrx_raises():
    """NPU_RUNTIME=hrx with no libhrx must raise ImportError.

    Only meaningful when HRX is not discoverable on this host; if it is, the
    contract can't be exercised, so skip.
    """
    import aie.utils as u

    if u.has_hrx:
        pytest.skip("HRX is discoverable on this host; missing-HRX path untestable")
    res = _run_import(
        {
            "NPU_RUNTIME": "hrx",
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


# ---------------------------------------------------------------------------
# Thread-safety of lazy initialization (no device / libhrx needed).
#
# HRXContext.get() and _HrxLib.ensure() use double-checked locking so a
# multithreaded process cannot double-init the device/stream or observe a
# half-bound libhrx. These tests replace the device-touching bodies with cheap
# counters that sleep *inside* the guarded section to widen the race window, then
# assert the body ran exactly once under heavy concurrent first-touch.
# ---------------------------------------------------------------------------


def test_hrx_context_get_is_thread_safe(monkeypatch):
    """Concurrent first-touch of HRXContext.get() must build exactly one context."""
    ctx_mod = pytest.importorskip("aie.utils.hostruntime.hrxruntime.context")

    builds = []

    def fake_init(self):
        # Runs inside HRXContext.get()'s lock; sleep widens the check->assign
        # window so a missing lock would let several threads build.
        builds.append(1)
        time.sleep(0.005)

    monkeypatch.setattr(ctx_mod.HRXContext, "_instance", None, raising=False)
    monkeypatch.setattr(ctx_mod.HRXContext, "__init__", fake_init)

    n_threads = 16
    results = []
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()  # release all threads at once -> maximize contention
        results.append(ctx_mod.HRXContext.get())

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(builds) == 1, f"HRXContext built {len(builds)}x, expected exactly 1"
    assert len({id(r) for r in results}) == 1, "threads saw >1 HRXContext instance"


def test_hrx_lib_ensure_binds_once_under_threads(monkeypatch):
    """_HrxLib.ensure() must bind libhrx exactly once under concurrent threads."""
    b = pytest.importorskip("aie.utils.hostruntime.hrxruntime._bindings")

    # Fresh instance so the test is independent of the process-wide `lib`
    # (which may already be bound) and never dlopen()s a real library.
    hrx_lib = b._HrxLib()
    binds = []

    def fake_bind(self):
        binds.append(1)
        time.sleep(0.005)
        self._ready = True

    monkeypatch.setattr(b._HrxLib, "_bind", fake_bind)

    n_threads = 16
    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        hrx_lib.ensure()

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(binds) == 1, f"libhrx bound {len(binds)}x, expected exactly 1"
    assert hrx_lib._ready is True
