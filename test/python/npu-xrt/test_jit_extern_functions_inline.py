# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

# ExternalFunction(inline=True): the kernel is emitted as alwaysinline LLVM IR,
# llvm-link'd into the core by aiecc and inlined -- so there is no surviving
# func.call and no separately object-linked kernel .o.  This exercises that path
# end-to-end and asserts the result matches the object-linked path exactly.
# Inline is the Peano front-end path (Chess cannot llvm-link); @jit uses Peano
# by default.  See issue #3396.

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import CompileTime, ExternalFunction, In, Out, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.controlflow import range_


@jit
def transform(
    input: In,
    output: Out,
    *,
    func: CompileTime[object],
    num_elements: CompileTime[int] = 1024,
    dtype: CompileTime[object] = np.int32,
):
    """Apply ``func`` to each tile of ``input``, once per tile."""
    tile_size = func.tile_size(0)
    if num_elements % tile_size != 0:
        raise ValueError(
            f"Number of elements ({num_elements}) must be a multiple of {tile_size}."
        )
    num_tiles = num_elements // tile_size

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out, fn):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            fn(elem_in, elem_out, fn.tile_size(0))
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod(), func])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


@pytest.fixture(autouse=True)
def _clear_state():
    transform._kernel_cache.clear()
    ExternalFunction._instances.clear()
    yield
    transform._kernel_cache.clear()
    ExternalFunction._instances.clear()


def _add_one(inline: bool) -> ExternalFunction:
    return ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* input, int* output, int tile_size) {
                for (int i = 0; i < tile_size; i++) {
                    output[i] = input[i] + 1;
                }
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
        inline=inline,
    )


def test_inline_add_one_is_numerically_correct():
    """inline=True compiles and runs, producing correct results."""
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    output_tensor = iron.zeros((1024,), dtype=np.int32, device="npu")
    expected = input_tensor.numpy() + 1

    transform(input_tensor, output_tensor, func=_add_one(inline=True))

    np.testing.assert_array_equal(output_tensor.numpy(), expected)


def test_inline_matches_object_linked():
    """inline=True and the default object-linked path give identical results."""
    input_tensor = iron.randint(0, 100, (1024,), dtype=np.int32, device="npu")
    ref_out = iron.zeros((1024,), dtype=np.int32, device="npu")
    inl_out = iron.zeros((1024,), dtype=np.int32, device="npu")

    transform(input_tensor, ref_out, func=_add_one(inline=False))
    # Reset caches so the second build is independent of the first.
    transform._kernel_cache.clear()
    ExternalFunction._instances.clear()
    transform(input_tensor, inl_out, func=_add_one(inline=True))

    np.testing.assert_array_equal(inl_out.numpy(), ref_out.numpy())
