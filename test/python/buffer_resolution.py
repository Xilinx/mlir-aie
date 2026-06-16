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
from aie.iron.device import NPU1Col1, NPU2

rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
data_ty = np.ndarray[(64,), np.dtype[np.int32]]


# ---------------------------------------------------------------------------
# Test 1: Buffer given to a Worker is resolved before inline_ops fires,
#         so element writes inside the callback produce correct rtp_write ops.
# CHECK-LABEL: TEST: rtp_buffer_written_in_inline_ops
# CHECK: %[[RV0:.+]] = arith.constant 7 : i32
# CHECK: aiex.npu.rtp_write(@my_rtp, 0, %[[RV0]]) : i32
# CHECK: %[[RV1:.+]] = arith.constant 3 : i32
# CHECK: aiex.npu.rtp_write(@my_rtp, 1, %[[RV1]]) : i32
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


def sequence(inp, out):

    def set_rtp(buf):
        buf[0] = 7
        buf[1] = 3

    rt.inline_ops(set_rtp, [rtp_buf])
    of_in.prod().fill(inp)
    of_out.cons().drain(out, wait=True)


rt.sequence(sequence, [data_ty, data_ty])

module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
print(module)


# ---------------------------------------------------------------------------
# Test 2: Multiple RTP buffers (one per worker) in a list, all written in one
#         inline_ops callback — mirrors the resnet layers_conv2_x pattern.
# CHECK-LABEL: TEST: multiple_rtp_buffers_in_inline_ops
# CHECK: %[[RV2:.+]] = arith.constant 1 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w0, 0, %[[RV2]]) : i32
# CHECK: %[[RV3:.+]] = arith.constant 2 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w1, 0, %[[RV3]]) : i32
# CHECK: %[[RV4:.+]] = arith.constant 3 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w2, 0, %[[RV4]]) : i32
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


def sequence(i0, i1, i2, o0, o1, o2):

    def set_rtps(rtps):
        rtps[0][0] = 1
        rtps[1][0] = 2
        rtps[2][0] = 3

    rt2.inline_ops(set_rtps, [rtps])
    of_ins[0].prod().fill(i0)
    of_ins[1].prod().fill(i1)
    of_ins[2].prod().fill(i2)
    of_outs[0].cons().drain(o0, wait=True)
    of_outs[1].cons().drain(o1, wait=True)
    of_outs[2].cons().drain(o2, wait=True)


rt2.sequence(sequence, [data_ty, data_ty, data_ty, data_ty, data_ty, data_ty])

module2 = Program(NPU2(), rt2, workers=list(workers)).resolve_program()
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


def sequence(inp3, out3):

    def write_both(placed, orphan):
        placed[0] = 1
        orphan[0] = 1  # orphan has no tile → should raise ValueError

    rt3.inline_ops(write_both, [placed_rtp, orphan_rtp])
    of_in3.prod().fill(inp3)
    of_out3.cons().drain(out3, wait=True)


rt3.sequence(sequence, [data_ty, data_ty])

try:
    Program(NPU1Col1(), rt3, workers=[worker3]).resolve_program()
    print("FAILED: expected ValueError but no exception was raised")
except ValueError as e:
    assert "placed" in str(e).lower(), f"unexpected message: {e}"
    print("PASSED")
except Exception as e:
    print(f"FAILED: expected ValueError, got {type(e).__name__}: {e}")
