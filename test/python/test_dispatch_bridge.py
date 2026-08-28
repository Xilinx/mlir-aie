# test_dispatch_bridge.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for DispatchBridge -- no NPU, no MLIR pipeline required.

Exercises the ctypes call convention and the -2 (guard-failed) sentinel
against a hand-built fixture ``.so`` that mimics ``dispatch_generate``'s ABI
directly (thread-local buffer owned by the callee, exact word count
returned), so these tests do not depend on aiecc/aie-opt/aie-translate at
all -- only a host C compiler.
"""

import subprocess

import pytest
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# One fixture .cpp exercising every path DispatchBridge needs to handle:
#   normal value       -> exact-size result via the thread-local buffer
#   value == 0          -> "guard failed" (std::nullopt-equivalent), returns -2
_FIXTURE_SRC = r"""
#include <cstddef>
#include <cstdint>
#include <vector>

thread_local static std::vector<uint32_t> g_result;

extern "C" int64_t dispatch_generate(int32_t scale, size_t n_tiles,
                                      uint32_t **out_ptr) {
  if (scale == 0) return -2;
  g_result.assign(n_tiles, 0);
  for (size_t i = 0; i < n_tiles; ++i) g_result[i] = static_cast<uint32_t>(scale) + i;
  *out_ptr = g_result.data();
  return static_cast<int64_t>(g_result.size());
}
"""


@pytest.fixture(scope="module")
def fixture_so(tmp_path_factory):
    """Compile the fixture .cpp once per test module; skip if no host compiler."""
    from aie.utils import config

    try:
        cxx = config.host_cxx_path()
    except RuntimeError:
        pytest.skip("no host C++ compiler available")

    tmp_dir = tmp_path_factory.mktemp("dispatch_bridge_fixture")
    src_path = tmp_dir / "fixture.cpp"
    src_path.write_text(_FIXTURE_SRC)
    so_path = tmp_dir / "fixture.so"
    result = subprocess.run(
        [
            cxx,
            "-shared",
            "-fPIC",
            "-O0",
            "-std=c++17",
            str(src_path),
            "-o",
            str(so_path),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return so_path


def _bridge(fixture_so):
    return DispatchBridge(
        fixture_so,
        dispatch_params=["scale", "n_tiles"],
        param_ctypes=["int32_t", "size_t"],
    )


def test_generate_returns_exact_size_result(fixture_so):
    bridge = _bridge(fixture_so)
    words = bridge.generate({"scale": 10, "n_tiles": 4})
    assert list(words) == [10, 11, 12, 13]


def test_generate_handles_varying_sizes_across_calls(fixture_so):
    """Successive calls with different sizes must each return correctly.

    Regression guard for the thread-local buffer being reused/overwritten
    correctly rather than stale data leaking between calls.
    """
    bridge = _bridge(fixture_so)
    small = bridge.generate({"scale": 1, "n_tiles": 2})
    large = bridge.generate({"scale": 100, "n_tiles": 6})
    assert list(small) == [1, 2]
    assert list(large) == [100, 101, 102, 103, 104, 105]


def test_generate_raises_on_guard_failed(fixture_so):
    bridge = _bridge(fixture_so)
    with pytest.raises(HostRuntimeError, match="overflowed a hardware BD field"):
        bridge.generate({"scale": 0, "n_tiles": 1})


def test_mismatched_param_lengths_rejected(fixture_so):
    with pytest.raises(ValueError, match="same length"):
        DispatchBridge(
            fixture_so,
            dispatch_params=["scale", "n_tiles"],
            param_ctypes=["int32_t"],
        )


def test_unrecognized_ctype_rejected(fixture_so):
    with pytest.raises(HostRuntimeError, match="unrecognized generated C type"):
        DispatchBridge(
            fixture_so,
            dispatch_params=["scale"],
            param_ctypes=["not_a_real_ctype"],
        )
