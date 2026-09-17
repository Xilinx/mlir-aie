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
    DispatchCompileError,
    _check_runtime_sequence_abi,
    dispatch_scalar_c_type,
)
from aie.ir import Context, Module
from aie.utils.compile.utils import SHARED_LIB_SUFFIX, host_shared_lib_cmd
from aie.utils.hostruntime.hostruntime import HostRuntimeError

# Mirrors TxnEncoding.h, which the generated source gets the real macro from.
# A Windows DLL exports nothing without it, so every fixture is prefixed with it.
_EXPORT_MACRO = r"""
#ifdef _WIN32
#define AIE_DISPATCH_EXPORT __declspec(dllexport)
#else
#define AIE_DISPATCH_EXPORT
#endif
"""

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

extern "C" AIE_DISPATCH_EXPORT int64_t dispatch_generate(int32_t scale, size_t n_tiles,
                                      uint32_t **out_ptr) {
  if (scale == 0) return -2;
  g_result.assign(n_tiles, 0);
  for (size_t i = 0; i < n_tiles; ++i) g_result[i] = static_cast<uint32_t>(scale) + i;
  *out_ptr = g_result.data();
  return static_cast<int64_t>(g_result.size());
}
"""

_FIXTURE_ABI = (
    'extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() '
    '{ return "int32_t,size_t"; }\n'
)


def _compile_fixture(tmp_dir, source, name):
    """Compile *source* into a shared library; skip the module if no host compiler."""
    src_path = tmp_dir / f"{name}.cpp"
    src_path.write_text(_EXPORT_MACRO + source)
    so_path = tmp_dir / f"{name}{SHARED_LIB_SUFFIX}"
    try:
        cmd = host_shared_lib_cmd(src_path, so_path, opt="-O0")
    except RuntimeError:
        pytest.skip("no host C++ compiler available")

    result = subprocess.run(cmd, capture_output=True, text=True)
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
    return DispatchBridge(fixture_so, dispatch_params=["scale", "n_tiles"])


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


def test_so_without_abi_rejected(fixture_so_no_abi):
    """A .so with no ABI to report is unusable and must be rebuilt."""
    with pytest.raises(HostRuntimeError, match="exports no dispatch_abi"):
        DispatchBridge(fixture_so_no_abi, dispatch_params=["scale", "n_tiles"])


def test_param_count_mismatch_with_so_rejected(fixture_so):
    """The .so takes two scalars, so a design declaring one does not match it."""
    with pytest.raises(HostRuntimeError, match="cached artifact is stale"):
        DispatchBridge(fixture_so, dispatch_params=["scale"])


@pytest.fixture(scope="module")
def bogus_ctype_so(tmp_path_factory):
    """Build a .so naming a C type no ctypes type corresponds to."""
    tmp_dir = tmp_path_factory.mktemp("dispatch_bridge_bogus")
    src = (
        'extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() '
        '{ return "not_a_real_ctype"; }\n'
    )
    return _compile_fixture(tmp_dir, src, "bogus")


def test_unrecognized_ctype_rejected(bogus_ctype_so):
    """A C type the bridge cannot marshal must be named, not silently coerced."""
    with pytest.raises(HostRuntimeError, match="unrecognized generated C type"):
        DispatchBridge(bogus_ctype_so, dispatch_params=["scale"])


def test_unloadable_so_rejected(tmp_path):
    """A truncated/corrupt .so must name the kernel cache, not raise a bare OSError."""
    bad = tmp_path / "not-an-elf.so"
    bad.write_text("this is not a shared object\n")
    with pytest.raises(HostRuntimeError, match="could not be loaded"):
        DispatchBridge(bad, dispatch_params=["scale"])


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


def test_param_type_mismatch_rejected():
    """A declared/generated type mismatch means the values are transposed.

    The generated parameter order is the Runtime(seq, fn_args=[...]) order the
    author wrote by hand; the declared order is the Python signature. Nothing
    ties them together, so this is the only signal available when a design
    threads its scalars in a different order than it declares them.
    """
    with Context():
        module = Module.parse("""module { aie.device(npu1_1col) {
              aie.runtime_sequence @seq(%rows: i64, %a: memref<8xi32>, %cols: i32) {}
            }}""")
        with pytest.raises(DispatchCompileError, match="declared as int32"):
            _check_runtime_sequence_abi(module, ["rows", "cols"], [np.int32, np.int64])
        _check_runtime_sequence_abi(module, ["rows", "cols"], [np.int64, np.int32])
        with pytest.raises(TypeError, match="Unsupported DispatchTime"):
            _check_runtime_sequence_abi(module, ["rows", "cols"], [None, None])
        with pytest.raises(DispatchCompileError, match="the design declares 1"):
            _check_runtime_sequence_abi(module, ["rows"], [np.int64])


@pytest.mark.parametrize("value", [1.9, 1.0, "2", None, np.float32(2)])
def test_non_integer_value_rejected(fixture_so, value):
    with pytest.raises(HostRuntimeError, match="must be an integer"):
        _bridge(fixture_so).generate({"scale": value, "n_tiles": 1})


def test_integer_protocol_evaluated_once(fixture_so):
    class Integer:
        calls = 0

        def __index__(self):
            self.calls += 1
            return self.calls

    value = Integer()
    assert list(
        _bridge(fixture_so).generate({"scale": value, "n_tiles": np.int64(2)})
    ) == [1, 2]
    assert value.calls == 1


@pytest.mark.parametrize(
    "ctype,value,accepted",
    [
        ("bool", 0, True),
        ("bool", 1, True),
        ("bool", 2, False),
        ("bool", -1, False),
        ("bool", 255, False),
        ("int8_t", -128, True),
        ("int8_t", 127, True),
        ("int8_t", -129, False),
        ("int8_t", 128, False),
        ("uint64_t", 2**64 - 1, True),
        ("uint64_t", 2**64, False),
    ],
)
def test_scalar_bounds(tmp_path, ctype, value, accepted):
    source = f"""
#include <cstdint>
extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() {{ return "{ctype}"; }}
extern "C" AIE_DISPATCH_EXPORT int64_t dispatch_generate({ctype} value, uint32_t **out) {{
  static uint32_t word;
  word = static_cast<uint32_t>(value);
  *out = &word;
  return 1;
}}
"""
    bridge = DispatchBridge(_compile_fixture(tmp_path, source, "bounds"), ["value"])
    if accepted:
        assert list(bridge.generate({"value": value})) == [value & 0xFFFFFFFF]
    else:
        with pytest.raises(HostRuntimeError, match="does not fit"):
            bridge.generate({"value": value})


@pytest.mark.parametrize(
    "status,error", [(-1, "unexpected status"), (0, None), (1, "null pointer")]
)
def test_invalid_or_empty_result(tmp_path, status, error):
    source = f"""
#include <cstdint>
extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() {{ return ""; }}
extern "C" AIE_DISPATCH_EXPORT int64_t dispatch_generate(uint32_t **out) {{
  *out = nullptr;
  return {status};
}}
"""
    bridge = DispatchBridge(_compile_fixture(tmp_path, source, "result"), [])
    if error:
        with pytest.raises(HostRuntimeError, match=error):
            bridge.generate({})
    else:
        result = bridge.generate({})
        assert result.dtype == np.uint32
        assert result.size == 0


@pytest.mark.parametrize(
    "abi,error",
    [("nullptr", "null dispatch ABI"), ('"\\xff"', "non-ASCII dispatch ABI")],
)
def test_malformed_abi(tmp_path, abi, error):
    source = (
        f'extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() {{ return {abi}; }}'
    )
    with pytest.raises(HostRuntimeError, match=error):
        DispatchBridge(_compile_fixture(tmp_path, source, "abi"), [])


def test_missing_generate_symbol(tmp_path):
    source = 'extern "C" AIE_DISPATCH_EXPORT const char *dispatch_abi() { return ""; }'
    with pytest.raises(HostRuntimeError, match="exports no dispatch_generate"):
        DispatchBridge(_compile_fixture(tmp_path, source, "missing"), [])


def test_numpy_index_aliases_follow_runtime_mapping():
    assert dispatch_scalar_c_type(np.uintp) == "size_t"
    assert dispatch_scalar_c_type(np.longlong) == (
        "int64_t" if np.longlong is np.int64 else "size_t"
    )
