# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
# REQUIRES: peano
"""An instructions-only build addresses the buffers where the image's build
placed them.

Placement reserves the banks a kernel's static data needs, which only the
compiled kernel shows, so the RTP buffer the sequence writes moves with the
kernel's table. The `--no-measure-data-size` build places it without that
reservation, which keeps the comparison from passing vacuously.
"""

import numpy as np
import pytest

from aie.iron import Buffer, ExternalFunction, ObjectFifo, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU2Col1
from aie.utils import set_current_device
from aie.utils.compile.jit.compilabledesign import CompilableDesign

TILE = 64
tile_ty = np.ndarray[(TILE,), np.dtype[np.int32]]
rtp_ty = np.ndarray[(4,), np.dtype[np.int32]]


def design():
    lookup = ExternalFunction(
        "lookup",
        source_string="""
            static int table[2048] = {1, 2, 3, 4, 5, 6, 7, 8};
            extern "C" void lookup(int *out, int k) {
                for (int i = 0; i < 64; i++)
                    out[i] = table[(i * k) & 2047]++;
            }""",
        arg_types=[tile_ty, np.int32],
    )
    of_out = ObjectFifo(tile_ty, name="out")
    rtp = Buffer(rtp_ty, name="rtp", use_write_rtp=True)

    def core(of_out, rtp, fn):
        for _ in range_(1):
            elem = of_out.acquire(1)
            fn(elem, rtp[0])
            of_out.release(1)

    worker = Worker(core, [of_out.prod(), rtp, lookup])

    def sequence(out, out_h):
        rtp[0] = 3
        out_h.drain(out, wait=True)

    rt = Runtime(sequence, [tile_ty, of_out.cons()])
    return Program(NPU2Col1(), rt, workers=[worker]).resolve_program()


@pytest.fixture(autouse=True)
def _device():
    set_current_device(NPU2Col1())
    ExternalFunction._instances.clear()
    yield
    ExternalFunction._instances.clear()


def test_insts_only_matches_the_full_build(tmp_path):
    full = tmp_path / "full"
    CompilableDesign(design, use_cache=False).compile(
        xclbin_path=full / "final.xclbin", inst_path=full / "insts.bin"
    )
    unmeasured = tmp_path / "unmeasured"
    CompilableDesign(
        design, use_cache=False, aiecc_flags=["--no-measure-data-size"]
    ).compile(
        xclbin_path=unmeasured / "final.xclbin", inst_path=unmeasured / "insts.bin"
    )
    only = tmp_path / "only" / "insts.bin"
    CompilableDesign(design, use_cache=False, insts_only=True).compile(inst_path=only)

    expected = (full / "insts.bin").read_bytes()
    assert (unmeasured / "insts.bin").read_bytes() != expected
    assert only.read_bytes() == expected
