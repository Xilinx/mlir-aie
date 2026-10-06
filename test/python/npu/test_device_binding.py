# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1_xrt% %pytest %s
# RUN: %run_on_npu2_xrt% %pytest %s
# REQUIRES: xrt_python_bindings
"""A design with no device bound targets the device the runtime reports."""

import numpy as np
import pytest

import aie.iron as iron
import aie.utils as utils
from aie.iron import In, ObjectFifo, Out, Program, Runtime
from aie.utils import get_current_device, set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign


@pytest.fixture(autouse=True)
def unbound():
    previous = get_current_device(probe_runtime=False)
    set_current_device(None)
    try:
        yield
    finally:
        set_current_device(previous)


def copy():
    ty = np.ndarray[(32,), np.dtype[np.int32]]
    of_in = ObjectFifo(ty)
    of_out = of_in.cons().forward()

    def sequence(a, b, a_in, b_out):
        a_in.fill(a)
        b_out.drain(b, wait=True)

    rt = Runtime(sequence, [ty, ty, of_in.prod(), of_out.cons()])
    return Program(get_current_device(), rt).resolve_program()


@iron.jit
def copy_jit(a: In, b: Out):
    return copy()


def runtime_device():
    return type(utils.DefaultNPURuntime.device())


def test_generation_binds_the_runtime_device_before_its_cache_key():
    generated_for = []

    def gen():
        generated_for.append(type(get_current_device(probe_runtime=False)))
        return copy()

    cd = CompilableDesign(gen)
    cd._generated
    assert generated_for == [runtime_device()]
    (key,) = cd._generated_cache
    assert runtime_device().__name__ in key[2]


def test_compile_binds_the_runtime_device_before_its_cache_lookup():
    cd = CompilableDesign(copy)
    cd.compile()
    assert type(get_current_device(probe_runtime=False)) is runtime_device()
    assert cd.get_cache_entry().directory.name == cd._compute_cache_hash()


def test_cleanup_npu_runtime_releases_the_default_runtime_contexts():
    a = iron.arange(32, dtype=np.int32)
    b = iron.zeros(32, dtype=np.int32)
    copy_jit(a, b)
    np.testing.assert_array_equal(b.numpy(), a.numpy())
    runtime = utils.DefaultNPURuntime
    assert runtime._context_cache

    utils.cleanup_npu_runtime()
    assert not runtime._context_cache
