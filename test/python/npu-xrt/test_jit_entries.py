# test_jit_entries.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""A Program with several entry runtimes, each dispatched by name from one build.

One worker adds 1 to each tile it is sent. Entry ``whole`` sends the whole
vector; entry ``half`` sends only its first half, so the second half of the
output keeps what the host put there. Both entries come from one compile, on
the xclbin path (one instruction stream per entry beside one xclbin) and on
the full-ELF path (one kernel per entry in one ELF).
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    CompilableDesign,
    CompileTime,
    DeviceConfiguration,
    In,
    InOut,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.utils.compile import NPU_CACHE_HOME
from aie.utils.trace import TraceConfig

_TILE = 16
_N = 1024
_SENTINEL = -7


def _entries_design(N, trace_config=None):
    tile_ty = np.ndarray[(_TILE,), np.dtype[np.int32]]
    tensor_ty = np.ndarray[(N,), np.dtype[np.int32]]
    of_in = ObjectFifo(tile_ty, name="in")
    of_out = ObjectFifo(tile_ty, name="out")

    def core_body(of_in, of_out):
        elem_in = of_in.acquire(1)
        elem_out = of_out.acquire(1)
        for i in range_(_TILE):
            elem_out[i] = elem_in[i] + 1
        of_in.release(1)
        of_out.release(1)

    worker = Worker(core_body, fn_args=[of_in.cons(), of_out.prod()])
    first_half = TensorAccessPattern((N,), 0, [1, 1, 1, N // 2], [0, 0, 0, 1])

    def whole(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    def half(a, b, in_h, out_h):
        in_h.fill(a, tap=first_half)
        out_h.drain(b, tap=first_half, wait=True)

    types = [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()]
    entries = [Runtime(whole, types, name="whole"), Runtime(half, types, name="half")]
    program = Program(iron.get_current_device(), entries, workers=[worker])
    if trace_config is not None:
        program.enable_trace(trace_config.trace_size, workers=[worker])
    return program.resolve_program()


@iron.jit
def entries_xclbin(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _entries_design(N)


@iron.jit(full_elf=True)
def entries_full_elf(input_buf: In, output_buf: Out, *, N: CompileTime[int]):
    return _entries_design(N)


@iron.jit
def traced_entries(
    input_buf: In,
    output_buf: Out,
    *,
    N: CompileTime[int],
    trace_config: CompileTime[TraceConfig | None] = None,
):
    return _entries_design(N, trace_config)


def entries_program():
    return _entries_design(_N)


Chunk = np.ndarray[(4,), np.dtype[np.int32]]


def _adding_configuration(name, value):
    in_fifo = ObjectFifo(Chunk, name=f"{name}_in")
    out_fifo = ObjectFifo(Chunk, name=f"{name}_out")

    def core(in_h, out_h):
        elem_in = in_h.acquire(1)
        elem_out = out_h.acquire(1)
        for i in range_(4):
            elem_out[i] = elem_in[i] + value
        in_h.release(1)
        out_h.release(1)

    def run(data, in_h, out_h):
        in_h.fill(data)
        out_h.drain(data, wait=True)

    runtime = Runtime(run, [Chunk, in_fifo.prod(), out_fifo.cons()], name="run")
    worker = Worker(core, [in_fifo.cons(), out_fifo.prod()])
    configuration = DeviceConfiguration(
        name, NPU2Col1(), workers=[worker], runtimes=[runtime]
    )
    return configuration, runtime


@iron.jit
def same_named_entries(data: InOut):
    left, left_run = _adding_configuration("left", 2)
    right, right_run = _adding_configuration("right", 3)
    return Program.compose([left, right], entry=[left_run, right_run]).resolve_program()


def _builds() -> set:
    if not NPU_CACHE_HOME.is_dir():
        return set()
    return {
        p.name
        for p in NPU_CACHE_HOME.iterdir()
        if p.is_dir() and p.name not in ("devices", "objects", "pch")
    }


def _expected(data, entry):
    if entry == "whole":
        return data + 1
    return np.concatenate([data[: _N // 2] + 1, np.full(_N // 2, _SENTINEL, np.int32)])


@pytest.mark.parametrize("design", [entries_xclbin, entries_full_elf])
def test_each_entry_runs_from_one_build(design):
    before = _builds()
    data = iron.arange(_N, dtype=np.int32)
    for entry, runs in (
        ("half", "half"),
        ("whole", "whole"),
        ("half", "half"),
        ("main:whole", "whole"),
    ):
        output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
        design(data, output, N=_N, entry=entry)
        output.to("cpu")
        np.testing.assert_array_equal(output.numpy(), _expected(data.numpy(), runs))
    # The default entry is the first, as an entry-less call dispatches it.
    output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
    design(data, output, N=_N)
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), _expected(data.numpy(), "whole"))
    assert len(_builds() - before) <= 1


@pytest.mark.parametrize("design", [entries_xclbin, entries_full_elf])
def test_an_unknown_entry_is_refused(design):
    data = iron.arange(_N, dtype=np.int32)
    output = iron.zeros(_N, dtype=np.int32, device="npu")
    with pytest.raises(ValueError, match="has no entry 'third'"):
        design(data, output, N=_N, entry="third")


@pytest.mark.parametrize("design", [entries_xclbin, entries_full_elf])
def test_a_wrongly_sized_buffer_is_refused(design):
    output = iron.zeros(_N, dtype=np.int32, device="npu")
    with pytest.raises(RuntimeError, match="covers 8192 bytes"):
        design(iron.arange(2 * _N, dtype=np.int32), output, N=_N, entry="half")


@pytest.mark.parametrize("full_elf", [False, True])
@pytest.mark.parametrize("entry, runs", [(None, "whole"), ("half", "half")])
def test_a_traced_entry_decodes_against_its_own_sequence(
    full_elf, entry, runs, tmp_path
):
    trace_config = TraceConfig(trace_size=8192)
    data = iron.arange(_N, dtype=np.int32)
    output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
    traced_entries.specialize(full_elf=full_elf)(
        data, output, N=_N, trace_config=trace_config, entry=entry
    )
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), _expected(data.numpy(), runs))
    assert trace_config.kernel == f"main:{runs}"
    assert trace_config.trace_to_json(
        trace_config.physical_mlir_path, str(tmp_path / f"{runs}.json")
    )


def test_same_named_entries_are_dispatched_by_qualified_name():
    for entry, value in (("left:run", 2), ("right:run", 3)):
        data = iron.arange(4, dtype=np.int32, device="npu")
        same_named_entries(data, entry=entry)
        data.to("cpu")
        np.testing.assert_array_equal(data.numpy(), np.arange(4) + value)
    with pytest.raises(ValueError, match="several entries named 'run'"):
        same_named_entries(iron.arange(4, dtype=np.int32, device="npu"), entry="run")


def test_a_missing_entry_stream_is_rebuilt():
    data = iron.arange(_N, dtype=np.int32)
    output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
    entries_xclbin(data, output, N=_N, entry="whole")
    tensor_ty = np.ndarray[(_N,), np.dtype[np.int32]]
    built = entries_xclbin.specialize(
        input_buf=tensor_ty, output_buf=tensor_ty, N=_N
    ).compilable
    built.compile()
    built.entry_insts_path("half").unlink()
    output = iron.tensor(np.full(_N, _SENTINEL, np.int32), device="npu")
    entries_xclbin(data, output, N=_N, entry="half")
    output.to("cpu")
    np.testing.assert_array_equal(output.numpy(), _expected(data.numpy(), "half"))
    assert built.entry_insts_path("half").is_file()


def test_an_insts_only_build_of_several_entries_is_refused():
    with pytest.raises(ValueError, match="lowers one runtime sequence"):
        CompilableDesign(entries_program, insts_only=True).compile()
