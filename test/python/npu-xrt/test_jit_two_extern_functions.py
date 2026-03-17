# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

# End-to-end test for the core new capability: a single Worker calling TWO
# distinct ExternalFunction instances, each compiled to its own object file.
# This exercises the full multi-.o JIT pipeline:
#   1. Two separate source compilations (two .o files in the cache dir)
#   2. aie-assign-core-link-files traces both func.call ops and emits
#      link_files = ["add_one.o", "scale_by_two.o"] on the CoreOp
#   3. Two INPUT() directives in the linker script (Peano path)
#   4. Successful lld link with both object files
#   5. Core executes both functions: output[i] = (input[i] + 1) * 2

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import ExternalFunction, jit
from aie.iron import ObjectFifo, Worker, Runtime, Program
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_


@jit
def add_then_scale(input, output, add_func, scale_func):
    """Apply add_func then scale_func sequentially on each tile."""
    num_elements = np.size(input)
    tile_size = add_func.tile_size(0)
    num_tiles = num_elements // tile_size
    dtype = input.dtype

    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out, add_fn, scale_fn):
        for _ in range_(num_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            # Apply add_fn first, writing result into elem_out as a temporary,
            # then apply scale_fn in-place on elem_out.
            add_fn(elem_in, elem_out, tile_size)
            scale_fn(elem_out, elem_out, tile_size)
            of_in.release(1)
            of_out.release(1)

    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), add_func, scale_func],
    )

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, B):
        rt.start(worker)
        rt.fill(of_in.prod(), A)
        rt.drain(of_out.cons(), B, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def test_two_external_functions_different_objects():
    """
    One core calls two ExternalFunctions compiled to separate object files.
    Expected result: output[i] = (input[i] + 1) * 2.
    """
    add_one = ExternalFunction(
        "add_one",
        source_string="""extern "C" {
            void add_one(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] + 1;
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    scale_by_two = ExternalFunction(
        "scale_by_two",
        source_string="""extern "C" {
            void scale_by_two(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] * 2;
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    input_tensor = iron.arange(32, dtype=np.int32)
    output_tensor = iron.zeros((32,), dtype=np.int32)

    add_then_scale(input_tensor, output_tensor, add_one, scale_by_two)

    expected = (np.arange(32, dtype=np.int32) + 1) * 2
    np.testing.assert_array_equal(output_tensor.numpy(), expected)


def test_two_external_functions_same_object():
    """
    One core calls two ExternalFunctions that share the same compiled object
    file. The aie-assign-core-link-files pass must deduplicate the .o path
    so it appears only once in link_files and is linked exactly once.
    Expected result: output[i] = (input[i] + 1) * 2 (same computation, shared .o).
    """
    # Both functions come from the same translation unit / object file name.
    add_one = ExternalFunction(
        "add_one_shared",
        object_file_name="shared_kernel.o",
        source_string="""extern "C" {
            void add_one_shared(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] + 1;
            }
            void scale_by_two_shared(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] * 2;
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    scale_by_two = ExternalFunction(
        "scale_by_two_shared",
        object_file_name="shared_kernel.o",
        source_string="""extern "C" {
            void add_one_shared(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] + 1;
            }
            void scale_by_two_shared(int* in, int* out, int n) {
                for (int i = 0; i < n; i++) out[i] = in[i] * 2;
            }
        }""",
        arg_types=[
            np.ndarray[(16,), np.dtype[np.int32]],
            np.ndarray[(16,), np.dtype[np.int32]],
            np.int32,
        ],
    )

    input_tensor = iron.arange(32, dtype=np.int32)
    output_tensor = iron.zeros((32,), dtype=np.int32)

    add_then_scale(input_tensor, output_tensor, add_one, scale_by_two)

    expected = (np.arange(32, dtype=np.int32) + 1) * 2
    np.testing.assert_array_equal(output_tensor.numpy(), expected)
