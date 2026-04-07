# Copyright (C) 2025, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Regression tests for Buffer placement and resolution behaviour.

Covers the bug reported in https://github.com/Xilinx/mlir-aie/issues/3011:
  - A Buffer passed to a Worker is placed automatically by the placer and
    resolved before inline_ops callbacks fire, so indexing inside the callback
    works correctly.
  - A Buffer that is created but never given to any Worker has no tile and
    therefore cannot be resolved; InlineOpRuntimeTask must raise a clear
    ValueError rather than a confusing AttributeError from __setitem__.
  - Multiple RTP buffers (one per worker) in a list can all be written inside a
    single inline_ops callback, reflecting the common RTP-initialisation pattern
    seen in ML examples such as resnet layers_conv2_x.
"""

import numpy as np

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1Col1, NPU2

rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
data_ty = np.ndarray[(64,), np.dtype[np.int32]]


# ---------------------------------------------------------------------------
# Test 1: Buffer given to a Worker is resolved before inline_ops fires,
#         so element writes inside the callback produce correct rtp_write ops.
# CHECK-LABEL: TEST: rtp_buffer_written_in_inline_ops
# CHECK: aiex.npu.rtp_write(@my_rtp, 0, 7)
# CHECK: aiex.npu.rtp_write(@my_rtp, 1, 3)
# ---------------------------------------------------------------------------
print("\nTEST: rtp_buffer_written_in_inline_ops")

of_in = ObjectFifo(data_ty, name="in")
of_out = ObjectFifo(data_ty, name="out")
rtp_buf = Buffer(rtp_ty, name="my_rtp", use_write_rtp=True)


def core_fn(of_in, of_out, rtp):
    scale = rtp[0]
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    of_in.release(1)
    of_out.release(1)


worker = Worker(core_fn, [of_in.cons(), of_out.prod(), rtp_buf])

rt = Runtime()
with rt.sequence(data_ty, data_ty) as (inp, out):

    def set_rtp(buf):
        buf[0] = 7
        buf[1] = 3

    rt.inline_ops(set_rtp, [rtp_buf])
    rt.start(worker)
    rt.fill(of_in.prod(), inp)
    rt.drain(of_out.cons(), out, wait=True)

module = Program(NPU1Col1(), rt).resolve_program(SequentialPlacer())
print(module)


# ---------------------------------------------------------------------------
# Test 2: Multiple RTP buffers (one per worker) in a list, all written in one
#         inline_ops callback — mirrors the resnet layers_conv2_x pattern.
# CHECK-LABEL: TEST: multiple_rtp_buffers_in_inline_ops
# CHECK: aiex.npu.rtp_write(@rtp_w0, 0, 1)
# CHECK: aiex.npu.rtp_write(@rtp_w1, 0, 2)
# CHECK: aiex.npu.rtp_write(@rtp_w2, 0, 3)
# ---------------------------------------------------------------------------
print("\nTEST: multiple_rtp_buffers_in_inline_ops")

n_workers = 3
of_ins = [ObjectFifo(data_ty, name=f"in{i}") for i in range(n_workers)]
of_outs = [ObjectFifo(data_ty, name=f"out{i}") for i in range(n_workers)]
rtps = [Buffer(rtp_ty, name=f"rtp_w{i}", use_write_rtp=True) for i in range(n_workers)]


def core_fn_rtp(of_in, of_out, rtp):
    scale = rtp[0]
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    of_in.release(1)
    of_out.release(1)


workers = [
    Worker(core_fn_rtp, [of_ins[i].cons(), of_outs[i].prod(), rtps[i]])
    for i in range(n_workers)
]

rt2 = Runtime()
with rt2.sequence(data_ty, data_ty, data_ty, data_ty, data_ty, data_ty) as (
    i0,
    i1,
    i2,
    o0,
    o1,
    o2,
):

    def set_rtps(rtps):
        rtps[0][0] = 1
        rtps[1][0] = 2
        rtps[2][0] = 3

    rt2.inline_ops(set_rtps, [rtps])
    rt2.start(*workers)
    rt2.fill(of_ins[0].prod(), i0)
    rt2.fill(of_ins[1].prod(), i1)
    rt2.fill(of_ins[2].prod(), i2)
    rt2.drain(of_outs[0].cons(), o0, wait=True)
    rt2.drain(of_outs[1].cons(), o1, wait=True)
    rt2.drain(of_outs[2].cons(), o2, wait=True)

module2 = Program(NPU2(), rt2).resolve_program(SequentialPlacer())
print(module2)


# ---------------------------------------------------------------------------
# Test 3: A Buffer never given to any Worker raises ValueError (not the
#         confusing AttributeError from __setitem__) when inline_ops fires.
#         This is the exact failure mode of GitHub issue #3011.
# CHECK-LABEL: TEST: unplaced_buffer_in_inline_ops_raises
# CHECK: PASSED
# ---------------------------------------------------------------------------
print("\nTEST: unplaced_buffer_in_inline_ops_raises")

of_in3 = ObjectFifo(data_ty, name="in3")
of_out3 = ObjectFifo(data_ty, name="out3")
placed_rtp = Buffer(rtp_ty, name="placed_rtp", use_write_rtp=True)
orphan_rtp = Buffer(
    rtp_ty, name="orphan_rtp", use_write_rtp=True
)  # never given to a Worker


def core_fn3(of_in, of_out, rtp):
    scale = rtp[0]
    elem_in = of_in.acquire(1)
    elem_out = of_out.acquire(1)
    of_in.release(1)
    of_out.release(1)


worker3 = Worker(core_fn3, [of_in3.cons(), of_out3.prod(), placed_rtp])

rt3 = Runtime()
with rt3.sequence(data_ty, data_ty) as (inp3, out3):

    def write_both(placed, orphan):
        placed[0] = 1
        orphan[0] = 1  # orphan has no tile → should raise ValueError

    rt3.inline_ops(write_both, [placed_rtp, orphan_rtp])
    rt3.start(worker3)
    rt3.fill(of_in3.prod(), inp3)
    rt3.drain(of_out3.cons(), out3, wait=True)

try:
    Program(NPU1Col1(), rt3).resolve_program(SequentialPlacer())
    print("FAILED: expected ValueError but no exception was raised")
except ValueError as e:
    assert "placed" in str(e).lower(), f"unexpected message: {e}"
    print("PASSED")
except Exception as e:
    print(f"FAILED: expected ValueError, got {type(e).__name__}: {e}")
