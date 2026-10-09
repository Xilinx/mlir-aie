# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""Designs compiled from several threads each build with their own kernels.

A generator registers its kernels in the process-wide
`ExternalFunction._instances`, which the next generation clears, so two
generations must not overlap. The first generator here waits, its kernel
registered, for the second to start; were they to overlap, the second would
clear the first's kernel.
"""

import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from aie.iron import ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.utils import set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign

TILE = 64
tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]


# Set by each generator as it starts, by symbol.
entered = {"fill_seven": threading.Event(), "fill_nine": threading.Event()}
# Whether fill_seven's generation saw fill_nine's start before it finished.
overlapped: list[bool] = []


def fill(symbol: str, value: int):
    """A design whose one core fills its output with `value` by a kernel of
    its own, `symbol`."""

    def generator():
        entered[symbol].set()
        kernel = ExternalFunction(
            symbol,
            source_string=f"""
                extern "C" void {symbol}(int *out) {{
                    for (int i = 0; i < {TILE}; i++)
                        out[i] = {value};
                }}""",
            arg_types=[tile_ty],
        )
        of_out = ObjectFifo(tile_ty, name="out")

        def core(of_out, fn):
            for _ in range_(1):
                elem = of_out.acquire(1)
                fn(elem)
                of_out.release(1)

        worker = Worker(core, [of_out.prod(), kernel])

        def sequence(out, out_h):
            out_h.drain(out, wait=True)

        rt = Runtime(sequence, [tile_ty, of_out.cons()])
        module = Program(NPU2Col1(), rt, workers=[worker]).resolve_program()
        if symbol == "fill_seven":
            overlapped.append(entered["fill_nine"].wait(timeout=1.0))
        return module

    return generator


@pytest.fixture(autouse=True)
def _device():
    set_current_device(NPU2Col1())
    ExternalFunction._instances.clear()
    for event in entered.values():
        event.clear()
    overlapped.clear()
    yield
    ExternalFunction._instances.clear()


def test_designs_compiled_from_threads_keep_their_own_kernels(tmp_path):
    designs = {
        "fill_seven": CompilableDesign(fill("fill_seven", 7), use_cache=False),
        "fill_nine": CompilableDesign(fill("fill_nine", 9), use_cache=False),
    }

    def build(name):
        # The second build starts once the first is generating.
        if name == "fill_nine":
            assert entered["fill_seven"].wait(timeout=60)
        out = tmp_path / name
        return designs[name].compile(
            xclbin_path=out / "final.xclbin", inst_path=out / "insts.bin"
        )

    with ThreadPoolExecutor(2) as pool:
        built = dict(zip(designs, pool.map(build, designs)))

    assert overlapped == [False]
    for name, (xclbin, insts) in built.items():
        assert xclbin.exists() and insts.exists()
        assert [k.name for k in designs[name]._generated[1]] == [name]
