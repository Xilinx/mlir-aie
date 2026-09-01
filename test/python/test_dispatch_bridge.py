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

import re
import subprocess
from pathlib import Path

import pytest
from aie.utils.compile.jit._dispatch_bridge import DispatchBridge
from aie.utils.compile.jit._dispatch_compile import _DYNAMIC_LOWERING_PASSES
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


# aiecc C++ symbol -> aie-opt flag, in getNpuDmaLoweringPipeline order. The
# dynamic path spells this pipeline as CLI flags; the static path builds it
# in-process. They must stay identical or a dispatch design lowers differently
# from the same design compiled statically.
_AIECC_PIPELINE = [
    ("createAIEMaterializeBDChainsPass", "--aie-materialize-bd-chains"),
    (
        "createAIESubstituteShimDMAAllocationsPass",
        "--aie-substitute-shim-dma-allocations",
    ),
    ("createAIEUnrollRuntimeSequenceLoopsPass", "--aie-unroll-runtime-sequence-loops"),
    ("createCanonicalizerPass", "--canonicalize"),
    ("createAIEDecomposeLargeDmaBdPass", "--aie-decompose-large-dma-bd"),
    ("createAIELowerDynamicBDPoolPass", "--aie-lower-dynamic-bd-pool"),
    ("createCanonicalizerPass", "--canonicalize"),
    ("createAIEAssignRuntimeSequenceBDIDsPass", "--aie-assign-runtime-sequence-bd-ids"),
    ("createAIEDMATasksToNPUPass", "--aie-dma-tasks-to-npu"),
    ("createAIELowerDmaChannelResetPass", "--aie-lower-dma-channel-reset"),
    ("createAIEDmaToNpuPass", "--aie-dma-to-npu"),
    ("createAIELowerSetLockPass", "--aie-lower-set-lock"),
    ("createAIELowerCoreResetPass", "--aie-lower-core-reset"),
]


def _aiecc_pipeline_symbols():
    """The create*Pass() calls in aiecc's getNpuDmaLoweringPipeline, in order."""
    header = Path(__file__).resolve().parents[2] / "tools" / "aiecc" / "IRTransforms.h"
    text = header.read_text()
    start = text.index("getNpuDmaLoweringPipeline")
    body = text[start : text.index("\n}", start)]
    return re.findall(r"create[A-Za-z0-9]+Pass", body)


def test_dynamic_lowering_matches_aiecc_pipeline():
    """The dynamic pass list must not drift from aiecc's static pipeline.

    If this fails, aiecc's pipeline changed: mirror the change into
    _DYNAMIC_LOWERING_PASSES and update _AIECC_PIPELINE here. Silently
    skipping a pass on the dynamic path produces wrong hardware behavior, not
    a compile error -- aie-opt happily accepts a shorter list.
    """
    symbols = _aiecc_pipeline_symbols()
    assert symbols == [sym for sym, _flag in _AIECC_PIPELINE], (
        "aiecc's getNpuDmaLoweringPipeline changed; _DYNAMIC_LOWERING_PASSES in "
        "_dispatch_compile.py and _AIECC_PIPELINE here both need updating."
    )
    assert _DYNAMIC_LOWERING_PASSES == [flag for _sym, flag in _AIECC_PIPELINE]


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

    The generated parameter order is the Runtime(inputs=[...]) order the
    author wrote by hand; the declared order is the Python signature. Nothing
    ties them together, so this is the only signal available when a design
    threads its scalars in a different order than it declares them.
    """
    import numpy as np
    from aie.utils.compile.jit._dispatch_compile import (
        DispatchCompileError,
        _check_param_types,
    )

    params = [("int64_t", "v1"), ("int32_t", "v2")]
    with pytest.raises(DispatchCompileError, match="declared as int32"):
        _check_param_types(params, ["rows", "cols"], [np.int32, np.int64])

    # Correctly ordered: no complaint.
    _check_param_types(params, ["rows", "cols"], [np.int64, np.int32])

    # Unknown declared type is left unchecked rather than guessed at.
    _check_param_types(params, ["rows", "cols"], [None, None])
