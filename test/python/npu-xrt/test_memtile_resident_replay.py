# test_memtile_resident_replay.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %run_on_npu2% %pytest %s
# REQUIRES: xrt_python_bindings

"""On-device test of a mem tile buffer loaded once and read a dispatch-time
number of times.

The fill's lock values, the re-arm of its producer lock and the replay's start
count all come from one DispatchTime scalar, `uses`, so one compiled design
serves every count. The replay is started twice, the second start overriding
the task's count, as an operand held resident across a slab would be.
"""

import aie.iron as iron
import numpy as np
import pytest
from aie.dialects._aie_enum_gen import AIETileType
from aie.extras.dialects import arith
from aie.iron import (
    Acquire,
    Bd,
    Buffer,
    DispatchTime,
    Flow,
    In,
    Lock,
    Out,
    Program,
    Release,
    Runtime,
)
from aie.iron.device import Tile
from aie.utils.hostruntime.hostruntime import HostRuntimeError

CHUNK = 128
MAX_USES = 63


@iron.jit
def resident_replay(a: In, c: Out, *, uses: DispatchTime[np.int32] = 2):
    in_ty = np.ndarray[(CHUNK,), np.dtype[np.int32]]
    out_ty = np.ndarray[(CHUNK * MAX_USES,), np.dtype[np.int32]]
    shim = Tile(col=0, row=0, tile_type=AIETileType.ShimNOCTile)
    mem = Tile(col=0, row=1, tile_type=AIETileType.MemTile)
    resident = Buffer(type=in_ty, tile=mem, name="resident")
    empty = Lock(tile=mem, name="empty")
    full = Lock(tile=mem, name="full")
    into = Flow(shim, mem, src_channel=0, dst_channel=0)
    out = Flow(mem, shim, src_channel=0, dst_channel=0)

    def seq(A, C, n):
        one = arith.constant(1)
        empty.set(n)
        into.fill(A)
        into.endpoint(mem).task(
            Bd(
                resident,
                acquires=[Acquire(empty, value=n)],
                releases=[Release(full, value=n)],
            )
        ).start()
        first = arith.divui(n, arith.constant(2))
        replay = out.endpoint(mem).task(
            Bd(resident, acquires=[Acquire(full)], releases=[Release(empty)]),
            runs=first,
        )
        replay.start()
        replay.start(repeat_count=n - first - one).free()
        out.drain(C, sizes=[1, 1, n, CHUNK], strides=[0, 0, CHUNK, 1], wait=True)

    rt = Runtime(seq, [in_ty, out_ty, uses])
    rt.add_lock(empty)
    rt.add_flow(into)
    rt.add_flow(out)
    return Program(iron.get_current_device(), rt).resolve_program()


def test_memtile_resident_replay():
    design = resident_replay.specialize()
    a = iron.tensor(
        np.random.default_rng(0).integers(0, 2**16, size=(CHUNK,), dtype=np.int32),
        dtype=np.int32,
        device="npu",
    )
    for uses in (2, 5, 17, MAX_USES):
        c = iron.zeros((CHUNK * MAX_USES,), dtype=np.int32, device="npu")
        design(a, c, uses=uses)
        expected = np.zeros((CHUNK * MAX_USES,), dtype=np.int32)
        expected[: CHUNK * uses] = np.tile(a.numpy(), uses)
        np.testing.assert_array_equal(c.numpy(), expected)
    assert len(design._kernel_cache) == 1


# 0 would let the fill acquire `empty` unconditionally, and 64 is past the lock's
# range: the instruction stream is refused before anything reaches the device.
@pytest.mark.parametrize("uses", [0, MAX_USES + 1])
def test_memtile_resident_replay_rejects(uses):
    design = resident_replay.specialize()
    a = iron.zeros((CHUNK,), dtype=np.int32, device="npu")
    c = iron.zeros((CHUNK * MAX_USES,), dtype=np.int32, device="npu")
    with pytest.raises(HostRuntimeError, match="overflowed a hardware BD field"):
        design(a, c, uses=uses)
