# Copyright (C) 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""Regression tests for Buffer placement and resolution behaviour.

Covers the bug reported in https://github.com/Xilinx/mlir-aie/issues/3011:
  - A Buffer passed to a Worker is placed automatically by the placer and
    resolved before the runtime sequence body runs, so indexing it inside the
    body (an RTP write) works correctly.
  - A Buffer that is created but never given to any Worker has no tile and
    therefore cannot be resolved; indexing it in the body raises a clear error
    rather than a confusing one.
  - Multiple RTP buffers (one per worker) in a list can all be written in the
    body, reflecting the common RTP-initialisation pattern seen in ML examples
    such as resnet layers_conv2_x.
"""

import numpy as np

from aie.iron import Buffer, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU1Col1, NPU2

rtp_ty = np.ndarray[(16,), np.dtype[np.int32]]
data_ty = np.ndarray[(64,), np.dtype[np.int32]]


# ---------------------------------------------------------------------------
# Test 1: Buffer given to a Worker is resolved before the sequence body runs,
#         so element writes in the body produce correct rtp_write ops.
# CHECK-LABEL: TEST: rtp_buffer_written_in_body
# CHECK: %[[RTP0:.*]] = arith.constant 7 : i32
# CHECK: aiex.npu.rtp_write(@my_rtp, 0, %[[RTP0]]) : i32
# CHECK: %[[RTP1:.*]] = arith.constant 3 : i32
# CHECK: aiex.npu.rtp_write(@my_rtp, 1, %[[RTP1]]) : i32
# ---------------------------------------------------------------------------
print("\nTEST: rtp_buffer_written_in_body")

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


def sequence(inp, out, in_h, out_h):
    # rtp_buf is placed via the Worker; the body runs after that, so the RTP
    # writes below resolve correctly.
    rtp_buf[0] = 7
    rtp_buf[1] = 3
    in_h.fill(inp)
    out_h.drain(out, wait=True)


rt = Runtime(sequence, [data_ty, data_ty], fn_args=[of_in.prod(), of_out.cons()])

module = Program(NPU1Col1(), rt, workers=[worker]).resolve_program()
print(module)


# ---------------------------------------------------------------------------
# Test 2: Multiple RTP buffers (one per worker) in a list, all written in the
#         sequence body — mirrors the resnet layers_conv2_x pattern.
# CHECK-LABEL: TEST: multiple_rtp_buffers_in_body
# CHECK: %[[RTPW0:.*]] = arith.constant 1 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w0, 0, %[[RTPW0]]) : i32
# CHECK: %[[RTPW1:.*]] = arith.constant 2 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w1, 0, %[[RTPW1]]) : i32
# CHECK: %[[RTPW2:.*]] = arith.constant 3 : i32
# CHECK: aiex.npu.rtp_write(@rtp_w2, 0, %[[RTPW2]]) : i32
# ---------------------------------------------------------------------------
print("\nTEST: multiple_rtp_buffers_in_body")

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


def sequence2(i0, i1, i2, o0, o1, o2, in_hs, out_hs):
    rtps[0][0] = 1
    rtps[1][0] = 2
    rtps[2][0] = 3
    in_hs[0].fill(i0)
    in_hs[1].fill(i1)
    in_hs[2].fill(i2)
    out_hs[0].drain(o0, wait=True)
    out_hs[1].drain(o1, wait=True)
    out_hs[2].drain(o2, wait=True)


rt2 = Runtime(
    sequence2,
    [data_ty, data_ty, data_ty, data_ty, data_ty, data_ty],
    fn_args=[
        [of_ins[i].prod() for i in range(n_workers)],
        [of_outs[i].cons() for i in range(n_workers)],
    ],
)

module2 = Program(NPU2(), rt2, workers=workers).resolve_program()
print(module2)


# ---------------------------------------------------------------------------
# Test 3: A Buffer never given to any Worker has no tile, so indexing it in the
#         sequence body raises a clear "not resolved" error rather than a
#         confusing one. This is the failure mode of GitHub issue #3011.
# CHECK-LABEL: TEST: unplaced_buffer_in_body_raises
# CHECK: PASSED
# ---------------------------------------------------------------------------
print("\nTEST: unplaced_buffer_in_body_raises")

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


def sequence3(inp3, out3, in_h, out_h):
    placed_rtp[0] = 1
    orphan_rtp[0] = 1  # orphan has no tile → should raise a clear error
    in_h.fill(inp3)
    out_h.drain(out3, wait=True)


rt3 = Runtime(sequence3, [data_ty, data_ty], fn_args=[of_in3.prod(), of_out3.cons()])

try:
    Program(NPU1Col1(), rt3, workers=[worker3]).resolve_program()
    print("FAILED: expected an error but no exception was raised")
except (AttributeError, ValueError) as e:
    assert "resolved" in str(e).lower(), f"unexpected message: {e}"
    print("PASSED")
except Exception as e:
    print(f"FAILED: expected a clear resolve error, got {type(e).__name__}: {e}")
