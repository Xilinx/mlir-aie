# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""A kernel's declaration lives in the IR it is resolved into, never on the kernel."""

import numpy as np
import pytest

import aie.iron as iron
from aie.dialects.aie import AIEDevice, buffer, device, object_fifo, tile
from aie.extras.context import mlir_mod_ctx
from aie.iron import (
    CompileTime,
    ExternalFunction,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_

SOURCE = 'extern "C" { void add_one(int* i, int* o, int n) {} }'


@pytest.fixture(autouse=True)
def _npu2(npu2_device):
    yield


def _add_one(tile_size=16, **kwargs):
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    return ExternalFunction(
        "add_one",
        source_string=SOURCE,
        arg_types=[tile_ty, tile_ty, np.int32],
        **kwargs,
    )


def _declarations(text, name="add_one"):
    return [line for line in text.splitlines() if f"func.func private @{name}" in line]


def _calls(text, name="add_one"):
    return [line for line in text.splitlines() if f"func.call @{name}" in line]


def _module(*bodies):
    """One ``aie.device`` per body, each run inside its device."""
    with mlir_mod_ctx() as ctx:
        for i, body in enumerate(bodies):
            device(AIEDevice.npu2_1col, sym_name=f"dev{i}")(body)
    return str(ctx.module)


@iron.jit
def _chain(
    input: In,
    output: Out,
    *,
    kernels: CompileTime[object],
    num_elements: CompileTime[int],
):
    tile_size = kernels[0].tile_size(0)
    tensor_ty = np.ndarray[(num_elements,), np.dtype[np.int32]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[np.int32]]
    fifos = [ObjectFifo(tile_ty, name=f"f{i}") for i in range(len(kernels) + 1)]

    def core_body(of_in, of_out, kernel):
        for _ in range_(num_elements // tile_size):
            a = of_in.acquire(1)
            b = of_out.acquire(1)
            kernel(a, b, tile_size)
            of_in.release(1)
            of_out.release(1)

    workers = [
        Worker(core_body, fn_args=[fifos[i].cons(), fifos[i + 1].prod(), kernel])
        for i, kernel in enumerate(kernels)
    ]

    def seq(a, b, of_in, of_out):
        of_in.fill(a)
        of_out.drain(b, wait=True)

    rt = Runtime(seq, [tensor_ty, tensor_ty, fifos[0].prod(), fifos[-1].cons()])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


def test_one_kernel_is_declared_in_every_generation():
    kernel = _add_one()
    for num_elements in (32, 64, 32):
        text = _chain.as_mlir(None, None, kernels=(kernel,), num_elements=num_elements)
        assert len(_declarations(text)) == 1
        assert len(_calls(text)) == 1


def test_equal_kernels_share_one_declaration():
    first, second = _add_one(), _add_one()
    assert first == second and first is not second
    text = _chain.as_mlir(None, None, kernels=(first, second), num_elements=32)
    assert len(_declarations(text)) == 1
    assert len(_calls(text)) == 2


def test_each_device_declares_its_own_copy():
    kernel = _add_one()
    text = _module(lambda: kernel.resolve(), lambda: kernel.resolve())
    assert len(_declarations(text)) == 2


def test_resolving_twice_in_one_scope_declares_once():
    kernel = _add_one()

    def body():
        kernel.resolve()
        kernel.resolve()

    assert len(_declarations(_module(body))) == 1


def test_same_name_different_signature_conflicts():
    def body():
        _add_one(16).resolve()
        _add_one(32).resolve()

    with pytest.raises(ValueError, match="conflicts.*signature"):
        _module(body)


def test_same_name_different_object_conflicts():
    def body():
        _add_one(object_file_name="first.o").resolve()
        _add_one(object_file_name="second.o").resolve()

    with pytest.raises(ValueError, match="conflicts.*link_with"):
        _module(body)


def test_same_name_different_stack_size_conflicts():
    def body():
        _add_one().resolve()
        _add_one(stack_size_override=2048).resolve()

    with pytest.raises(ValueError, match="conflicts.*stack_size_override"):
        _module(body)


def test_symbol_owned_by_another_op_is_rejected():
    def body():
        tile_ty = np.ndarray[(16,), np.dtype[np.int32]]
        object_fifo("add_one", tile(0, 0), tile(0, 2), 2, tile_ty)
        _add_one().resolve()

    with pytest.raises(ValueError, match="already names a aie.objectfifo"):
        _module(body)


def test_call_before_resolve_is_rejected():
    def body():
        data = buffer(tile(0, 2), np.ndarray[(16,), np.dtype[np.int32]])
        _add_one()(data, data, 16)

    with pytest.raises(ValueError, match="must be resolved"):
        _module(body)
