# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %pytest %s
# REQUIRES: xrt_python_bindings

from collections import OrderedDict

import pytest
from aie.utils.hostruntime.hostruntime import HostRuntimeError
from aie.utils.hostruntime.xrtruntime.hostruntime import (
    CachedXRTRuntime,
    XRTHostRuntime,
    XRTKernelHandle,
)
import pyxrt


@pytest.mark.parametrize("runtime_type", [XRTHostRuntime, CachedXRTRuntime])
def test_failed_init_requires_previous_context_to_reinitialize(
    monkeypatch, runtime_type
):
    runtime = runtime_type.__new__(runtime_type)
    runtime._dispatched_context = None
    runtime._tensor_class = object
    runtime._context_cache = OrderedDict()
    runtime._insts_cache = OrderedDict()
    monkeypatch.setattr(runtime, "check_device_consistency", lambda: None)

    contexts = [object(), object()]
    handles = [
        XRTKernelHandle(
            kernel=None,
            xclbin=None,
            context=context,
            insts=None,
            name="main:sequence",
            is_full_elf=True,
            init_name="main:reconfig_init",
        )
        for context in contexts
    ]
    for context in contexts:
        runtime._context_cache[context] = {
            "context": context,
            "handles": [],
            "runs": {},
            "kernels": {},
        }

    initialized = []
    dispatched = []

    def initialize(self, context, init_name, kernel_name):
        initialized.append(context)
        if context is contexts[1]:
            raise HostRuntimeError("initialization failed after resetting the array")

    monkeypatch.setattr(XRTHostRuntime, "_run_init_entry", initialize)
    monkeypatch.setattr(
        runtime,
        "_run_full_elf",
        lambda handle, *args: dispatched.append(handle.context),
    )

    runtime.run(handles[0], [])
    runtime.run(handles[0], [])
    assert initialized == [contexts[0]]
    with pytest.raises(HostRuntimeError, match="initialization failed"):
        runtime.run(handles[1], [])
    assert runtime._dispatched_context is None
    if runtime_type is CachedXRTRuntime:
        assert contexts[1] not in runtime._context_cache
        assert contexts[0] in runtime._context_cache
    runtime.run(handles[0], [])
    assert initialized == [contexts[0], contexts[1], contexts[0]]
    assert dispatched == [contexts[0], contexts[0], contexts[0]]


@pytest.mark.parametrize("runtime_type", [XRTHostRuntime, CachedXRTRuntime])
@pytest.mark.parametrize("failure", ["raises", "times_out"])
def test_failed_dispatch_reinitializes(monkeypatch, runtime_type, failure):
    runtime = runtime_type.__new__(runtime_type)
    runtime._dispatched_context = None
    runtime._tensor_class = object
    monkeypatch.setattr(runtime, "check_device_consistency", lambda: None)

    completed = pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    states = iter([completed, pyxrt.ert_cmd_state.ERT_CMD_STATE_TIMEOUT, completed])

    class Run:
        def dispatch(self, buffers, parameters):
            state = next(states)
            if failure == "raises" and state != completed:
                raise RuntimeError("dispatch failed")
            return state, 0

    context = object()
    handle = XRTKernelHandle(
        kernel=None,
        xclbin=None,
        context=context,
        insts=None,
        name="main:sequence",
        is_full_elf=True,
        init_name="main:reconfig_init",
        kernel_run=Run(),
    )
    initialized = []
    monkeypatch.setattr(
        XRTHostRuntime,
        "_run_init_entry",
        lambda self, context, init_name, kernel_name: initialized.append(context),
    )

    runtime.run(handle, [])
    with pytest.raises((RuntimeError, HostRuntimeError)):
        runtime.run(handle, [])
    assert runtime._dispatched_context is None
    runtime.run(handle, [])
    assert initialized == [context, context]
