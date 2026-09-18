# test_npukernel_unit.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Kernel keyword routing regressions requiring neither compilation nor an NPU."""

from unittest.mock import Mock

import aie.utils as utils
import numpy as np
import pytest
from aie.utils.compile.jit.markers import DispatchTime, In
from aie.utils.jit import jit
from aie.utils.npukernel import NPUKernel


@pytest.fixture
def runtime(monkeypatch):
    runtime = Mock()
    monkeypatch.setitem(utils.__dict__, "DefaultNPURuntime", runtime)
    return runtime


def test_dispatch_signature_cannot_be_mutated(monkeypatch):
    import aie.utils.npukernel as npukernel
    from aie.utils.hostruntime.hostruntime import HostRuntimeError

    bridge = Mock()
    bridge_factory = Mock(return_value=bridge)
    monkeypatch.setattr(npukernel, "DispatchBridge", bridge_factory)
    names = ["count"]
    kernel = NPUKernel(dispatch_params=names, dispatch_lib_path="unused.so")
    names.append("extra")
    kernel.dispatch_params.clear()
    kernel._generate_dispatch_insts({"count": 3})
    kernel.dispatch_params.append("extra")
    with pytest.raises(HostRuntimeError, match="dispatch scalar mismatch"):
        kernel._generate_dispatch_insts({"count": 3, "extra": 4})
    assert kernel.dispatch_params == ["count"]
    bridge_factory.assert_called_once()
    bridge.generate.assert_called_once_with({"count": 3})


@pytest.mark.parametrize("dispatch_params", [[], ["n_tiles"]])
@pytest.mark.parametrize("unknown", ["n_tile", "rety", "dispatch_scalars"])
def test_unknown_keyword_rejected_before_runtime(runtime, dispatch_params, unknown):
    kernel = NPUKernel(dispatch_params=dispatch_params)
    scalars = {name: 3 for name in dispatch_params}

    with pytest.raises(TypeError, match=f"unexpected keyword.*'{unknown}'"):
        kernel(object(), **scalars, **{unknown: 6})

    runtime.load_and_run.assert_not_called()


def test_unknown_keyword_does_not_initialize_runtime(monkeypatch):
    def fail_runtime_probe():
        pytest.fail("unknown keywords must be rejected before runtime initialization")

    monkeypatch.delitem(utils.__dict__, "DefaultNPURuntime", raising=False)
    monkeypatch.setattr(utils, "_get_default_npu_runtime", fail_runtime_probe)

    with pytest.raises(TypeError, match="unexpected keyword.*'n_tile'"):
        NPUKernel(dispatch_params=["n_tiles"])(n_tiles=3, n_tile=6)


@pytest.mark.parametrize("dispatch_params", [[], ["n_tiles"], ["retry"]])
@pytest.mark.parametrize("retry", [None, False, True])
def test_valid_keywords_forwarded(runtime, dispatch_params, retry):
    kernel = NPUKernel(dispatch_params=dispatch_params)
    tensor = object()
    scalars = {name: 3 for name in dispatch_params}
    options = {} if retry is None or "retry" in scalars else {"retry": retry}

    result = kernel(tensor, **scalars, **options)

    runtime.load_and_run.assert_called_once_with(
        kernel, [tensor], dispatch_scalars=scalars or None, **options
    )
    assert result is runtime.load_and_run.return_value


@pytest.mark.parametrize("specialized", [False, True])
def test_jit_default_does_not_hide_unknown_keyword(
    monkeypatch, runtime, npu2_device, specialized
):
    @jit
    def design(a: In, *, n_tiles: DispatchTime[np.int32] = 3):
        pass

    if specialized:
        design = design.specialize(n_tiles=3)
    kernel = NPUKernel(dispatch_params=design.compilable.dispatch_params)
    monkeypatch.setattr(design, "_compile_and_build_kernel", lambda *args: kernel)

    with pytest.raises(TypeError, match="unexpected keyword.*'n_tile'"):
        design(object(), n_tile=6)

    runtime.load_and_run.assert_not_called()


@pytest.mark.parametrize("kwargs,expected", [({}, 3), ({"n_tiles": 6}, 6)])
def test_jit_dispatch_default_and_override(
    monkeypatch, runtime, npu2_device, kwargs, expected
):
    @jit
    def design(a: In, *, n_tiles: DispatchTime[np.int32] = 3):
        pass

    kernel = NPUKernel(dispatch_params=design.compilable.dispatch_params)
    tensor = object()
    monkeypatch.setattr(design, "_compile_and_build_kernel", lambda *args: kernel)

    design(tensor, retry=False, **kwargs)

    runtime.load_and_run.assert_called_once_with(
        kernel, [tensor], dispatch_scalars={"n_tiles": expected}, retry=False
    )
