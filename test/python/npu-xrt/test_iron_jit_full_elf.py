# test_iron_jit_full_elf.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""End-to-end tests for the full-ELF @iron.jit path (issue #3148).

``@iron.jit(full_elf=True)`` compiles a design to a single self-contained ELF
(PDIs + TXN control code) instead of an xclbin + insts pair, and runs it through
the full-ELF ``XRTHostRuntime`` path (``pyxrt.hw_context(dev, pyxrt.elf(...))``
+ ``run.set_arg`` + ``run.start``).  Full ELF is an NPU2 feature.

Coverage:
- @iron.jit(full_elf=True) transparent compile + run + verify
- npu.load_pdi is auto-injected into the runtime sequence
- AOT compile(full_elf_path=...) writes a single ELF and no xclbin/insts
- Full-ELF result matches the same design on the default xclbin path
- Trace on the full-ELF path
"""

import os

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import (
    CompileTime,
    In,
    Out,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.utils import tensor
from aie.utils.trace import TraceConfig, parse_trace

_TILE_SIZE = 16


def _add_const_design(input_buf, output_buf, N, add_value):
    """Add ``add_value`` to every element of a length-N int32 vector."""
    tile_ty = np.ndarray[(_TILE_SIZE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE_SIZE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE_SIZE):
                elem_out[i] = elem_in[i] + add_value
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


_N = 1024


@pytest.fixture(scope="session")
def input_array():
    return iron.arange(_N, dtype=np.int32)


@iron.jit(full_elf=True)
def add_const_full_elf(
    input_buf: In, output_buf: Out, *, N: CompileTime[int], add_value: CompileTime[int]
):
    return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)


@pytest.mark.parametrize("add_value", [1, 5, 100])
def test_full_elf_correct_output(input_array, add_value):
    """@iron.jit(full_elf=True) compiles to one ELF and runs correctly."""
    output = iron.zeros(_N, dtype=np.int32, device="npu")
    add_const_full_elf(input_array, output, N=_N, add_value=add_value)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), input_array.numpy() + add_value)


def test_full_elf_injects_load_pdi(input_array):
    """The full-ELF path auto-injects npu.load_pdi referencing the device."""
    mlir = add_const_full_elf.as_mlir(input_array, None, N=_N, add_value=1)
    assert "npu.load_pdi" in mlir


def test_full_elf_aot_single_elf(tmp_path):
    """AOT compile(full_elf_path=...) writes one ELF and returns (elf, None)."""
    elf_path = tmp_path / "design.elf"
    design = add_const_full_elf.specialize(N=_N, add_value=3)
    elf, insts = design.compile(full_elf_path=str(elf_path))
    assert insts is None
    assert elf_path.exists()
    # A full ELF is self-contained: no sibling xclbin / insts.bin.
    assert not (tmp_path / "final.xclbin").exists()
    assert not (tmp_path / "insts.bin").exists()


def test_full_elf_matches_xclbin_path(input_array):
    """Full-ELF output is identical to the default xclbin+insts path."""

    @iron.jit(N=_N, add_value=9)
    def add_nine_xclbin(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    @iron.jit(N=_N, add_value=9, full_elf=True)
    def add_nine_full_elf(
        input_buf: In,
        output_buf: Out,
        *,
        N: CompileTime[int],
        add_value: CompileTime[int],
    ):
        return _add_const_design(input_buf, output_buf, N=N, add_value=add_value)

    out_xclbin = iron.zeros(_N, dtype=np.int32, device="npu")
    add_nine_xclbin(input_array, out_xclbin)
    out_xclbin.to("cpu")

    out_elf = iron.zeros(_N, dtype=np.int32, device="npu")
    add_nine_full_elf(input_array, out_elf)
    out_elf.to("cpu")

    np.testing.assert_array_equal(out_elf.numpy(), out_xclbin.numpy())


@iron.jit(full_elf=True)
def add_const_full_elf_trace(
    input_buf: In,
    output_buf: Out,
    *,
    N: CompileTime[int],
    add_value: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    tile_ty = np.ndarray[(_TILE_SIZE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        for _ in range_(N // _TILE_SIZE):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            for i in range_(_TILE_SIZE):
                elem_out[i] = elem_in[i] + add_value
            of_in.release(1)
            of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (a, b):
        if trace_config:
            rt.enable_trace(trace_config.trace_size, workers=[worker])
        rt.start(worker)
        rt.fill(of_in.prod(), a)
        rt.drain(of_out.cons(), b, wait=True)
    return Program(iron.get_current_device(), rt).resolve_program()


@pytest.mark.parametrize("trace_size", [8192])
def test_full_elf_trace(trace_size):
    """Tracing works on the full-ELF path: output correct + trace collected."""
    ref = np.arange(_N, dtype=np.int32)
    a = tensor(ref, dtype=np.int32)
    c = tensor(np.zeros(_N, dtype=np.int32), dtype=np.int32)

    trace_config = TraceConfig(trace_size=trace_size)
    add_const_full_elf_trace(a, c, N=_N, add_value=0, trace_config=trace_config)

    c.to("cpu")
    np.testing.assert_array_equal(c.numpy(), ref)

    assert os.path.exists(trace_config.trace_file)
    assert trace_config.physical_mlir_path is not None
    assert os.path.exists(trace_config.physical_mlir_path)

    with open(trace_config.physical_mlir_path, "r") as f:
        physical_mlir_str = f.read()
    trace_buffer = trace_config.read_trace()
    trace_events = parse_trace(trace_buffer, physical_mlir_str)
    assert len(trace_events) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
