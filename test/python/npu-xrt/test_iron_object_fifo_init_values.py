# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# RUN: %run_on_npu1% %pytest %s
# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""End-to-end NPU test for IRON ObjectFifo's `init_values` constructor arg.

Builds an IRON program whose only data source is a MemTile buffer
pre-populated with `arange(1, N+1)` via init_values, copies through a
compute-tile worker, drains to the host, and verifies the output bytes
match the init data.

Cycling the static buffer multiple times via set_iter_count is NOT
exercised here — empirically that path hangs on hardware in this setup,
which is a separate issue. This test covers the single-transfer case
(R=1), which is sufficient to prove the init_values attribute reaches
the hardware and the static buffer is populated correctly at design
startup.
"""

import numpy as np
import pytest

import aie.iron as iron
from aie.iron import ObjectFifo, Out, Program, Runtime, Worker
from aie.iron.controlflow import range_
from aie.iron.device import AnyMemTile
from aie.iron.dataflow.endpoint import ObjectFifoEndpoint

N = 16  # elements per buffer
R = 1  # single transfer; cycling via set_iter_count is a separate concern


@iron.jit
def init_values_design(out: Out):
    """Build an IRON program whose only data source is a MemTile buffer
    pre-populated with `arange(1, N+1)` via init_values. Cycle the buffer
    R times via set_iter_count, copy through a compute-tile worker, drain
    R*N int32 to `out`."""
    dev = iron.get_current_device()
    tile_ty = np.ndarray[(N,), np.dtype[np.int32]]

    of_init = ObjectFifo(
        tile_ty,
        depth=1,
        name="of_init",
        init_values=[np.arange(1, N + 1, dtype=np.int32)],
    )
    # No Worker/rt.fill targets the producer side; pin it to a MemTile.
    of_init.prod().endpoint = ObjectFifoEndpoint(AnyMemTile)

    of_out = ObjectFifo(tile_ty, depth=2, name="of_out")

    def copy_body(of_in_c, of_out_p):
        for _ in range_(R):
            elem_in = of_in_c.acquire(1)
            elem_out = of_out_p.acquire(1)
            for i in range_(N):
                elem_out[i] = elem_in[i]
            of_in_c.release(1)
            of_out_p.release(1)

    worker = Worker(copy_body, fn_args=[of_init.cons(), of_out.prod()])

    rt = Runtime()
    out_ty = np.ndarray[(N * R,), np.dtype[np.int32]]

    def sequence(a):
        of_out.cons().drain(a, wait=True)

    rt.sequence(sequence, [out_ty])

    return Program(dev, rt, workers=[worker]).resolve_program()


def test_iron_object_fifo_init_values_e2e():
    out = iron.zeros([N * R], dtype=np.int32)
    init_values_design(out)

    out.to("cpu")
    actual = out.numpy()
    expected = np.tile(np.arange(1, N + 1, dtype=np.int32), R)
    np.testing.assert_array_equal(
        actual,
        expected,
        err_msg=(
            "Output did not match init_values tiled R times. "
            f"got first 4: {actual[:4]}, expected first 4: {expected[:4]}"
        ),
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
