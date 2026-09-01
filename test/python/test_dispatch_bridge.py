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

import numpy as np
import pytest
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.compile.jit._dispatch_compile import (
    _DYNAMIC_LOWERING_PASSES,
    DispatchCompileError,
    _check_built_abi,
)
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# One fixture .cpp exercising every path DispatchBridge needs to handle:
#   normal value       -> exact-size result via the thread-local buffer
#   value == 0          -> "guard failed" (std::nullopt-equivalent), returns -2
# Hand-written, so it pins the shape ConvertAIEXToEmitC must emit from this
# side; test/Conversion/AIEXToEmitC/dispatch_shim.mlir pins the other side.
_FIXTURE_BODY = r"""
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

_FIXTURE_ABI = 'extern "C" const char *dispatch_abi() { return "int32_t,size_t"; }\n'


def _compile_fixture(tmp_dir, source, name):
    """Compile *source* into ``<name>.so``; skip the module if no host compiler."""
    from aie.utils import config

    try:
        cxx = config.host_cxx_path()
    except RuntimeError:
        pytest.skip("no host C++ compiler available")

    src_path = tmp_dir / f"{name}.cpp"
    src_path.write_text(source)
    so_path = tmp_dir / f"{name}.so"
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


@pytest.fixture(scope="module")
def fixture_so(tmp_path_factory):
    """Build a self-describing fixture .so, as ConvertAIEXToEmitC would emit one."""
    tmp_dir = tmp_path_factory.mktemp("dispatch_bridge_fixture")
    return _compile_fixture(tmp_dir, _FIXTURE_ABI + _FIXTURE_BODY, "fixture")


@pytest.fixture(scope="module")
def fixture_so_no_abi(tmp_path_factory):
    """Build a .so with no dispatch_abi(), as an unusable cache entry has."""
    tmp_dir = tmp_path_factory.mktemp("dispatch_bridge_fixture_no_abi")
    return _compile_fixture(tmp_dir, _FIXTURE_BODY, "fixture_no_abi")


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


def test_param_ctypes_read_from_the_so(fixture_so):
    """With no param_ctypes given, the signature comes from dispatch_abi()."""
    bridge = DispatchBridge(fixture_so, dispatch_params=["scale", "n_tiles"])
    assert list(bridge.generate({"scale": 10, "n_tiles": 3})) == [10, 11, 12]


def test_so_without_abi_rejected(fixture_so_no_abi):
    """A .so with no ABI to report is unusable and must be rebuilt."""
    with pytest.raises(HostRuntimeError, match="exports no dispatch_abi"):
        DispatchBridge(fixture_so_no_abi, dispatch_params=["scale", "n_tiles"])


def test_param_count_mismatch_with_so_rejected(fixture_so):
    """The .so takes two scalars, so a design declaring one does not match it."""
    with pytest.raises(HostRuntimeError, match="cached artifact is stale"):
        DispatchBridge(fixture_so, dispatch_params=["scale"])


def test_unrecognized_ctype_rejected(fixture_so):
    with pytest.raises(HostRuntimeError, match="unrecognized generated C type"):
        DispatchBridge(
            fixture_so,
            dispatch_params=["scale"],
            param_ctypes=["not_a_real_ctype"],
        )


def test_dynamic_lowering_invokes_the_shared_pipeline():
    """The dynamic path must run aiecc's pipeline, not a copy of its pass list.

    aiecc's getNpuDmaLoweringPipeline and this flag both resolve to
    buildNpuDmaLoweringPipeline (lib/Dialect/AIEX/Transforms/AIEXNpuPipelines.cpp),
    which is what keeps the two paths lowering identically. Spelling out
    individual passes here would break that: aie-opt accepts a short list
    happily, so a skipped pass is wrong hardware behavior, not a compile error.
    """
    assert _DYNAMIC_LOWERING_PASSES == ["--aie-npu-dma-lowering"]


@pytest.mark.parametrize(
    "value", [2**31, -(2**31) - 1, 2**70, -1], ids=["hi", "lo", "huge", "neg-unsigned"]
)
def test_out_of_range_value_rejected(fixture_so, value):
    """A value that would silently wrap must raise, not dispatch.

    ctypes truncates without complaint -- c_int32(2**31) is -2147483648 and
    c_int32(2**70) is 0 -- so an unchecked value produces a valid-looking
    instruction stream built from a number the caller never passed.
    """
    bridge = _bridge(fixture_so)
    # scale is int32_t, n_tiles is size_t (unsigned): -1 fits neither.
    param = "n_tiles" if value == -1 else "scale"
    other = {"n_tiles": 2} if param == "scale" else {"scale": 1}
    with pytest.raises(HostRuntimeError, match="does not fit its generated C"):
        bridge.generate({param: value, **other})


@pytest.fixture(scope="module")
def transposed_so(tmp_path_factory):
    """Build a .so reporting (int64_t, int32_t); only dispatch_abi() is needed."""
    tmp_dir = tmp_path_factory.mktemp("dispatch_bridge_transposed")
    src = 'extern "C" const char *dispatch_abi() { return "int64_t,int32_t"; }\n'
    return _compile_fixture(tmp_dir, src, "transposed")


def test_param_type_mismatch_rejected(transposed_so):
    """A declared/generated type mismatch means the values are transposed.

    The generated parameter order is the Runtime(inputs=[...]) order the
    author wrote by hand; the declared order is the Python signature. Nothing
    ties them together, so this is the only signal available when a design
    threads its scalars in a different order than it declares them.
    """
    with pytest.raises(DispatchCompileError, match="declared as int32"):
        _check_built_abi(transposed_so, ["rows", "cols"], [np.int32, np.int64])

    # Correctly ordered: no complaint.
    _check_built_abi(transposed_so, ["rows", "cols"], [np.int64, np.int32])

    # Unknown declared type is left unchecked rather than guessed at.
    _check_built_abi(transposed_so, ["rows", "cols"], [None, None])

    # Arity is checked even with no declared types to compare.
    with pytest.raises(DispatchCompileError, match="the design declares 1"):
        _check_built_abi(transposed_so, ["rows"], None)
