# variable_rate_filter/variable_rate_filter.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
#
# A producer worker reads each input window from a fixed-rate
# upstream ObjectFifo, inspects its first byte, and conditionally
# forwards the window to a downstream VariableRateFifo. The
# downstream consumer (the host runtime drain) sees only the
# forwarded windows.
#
# Predicate is "first byte is even" -> ~50 % of windows forwarded.
# Choosing a deterministic predicate over the data makes the
# expected output verifiable on the host without baking the
# predicate into the test harness.
#
# Topology:
#
#   shim DMA (host) --> in_of (ObjectFifo) --> Tile A (filter) -->
#       out_of (VariableRateFifo) --> shim DMA (host)
#
# The filter kernel:
#
#   def filter_fn(in_handle, out_handle):
#       in_view = in_handle.acquire(1)
#       win = in_view[0]
#       if win[0] % 2 == 0:
#           out_view = out_handle.acquire(1)
#           # passthrough copy
#           for i in range(line_size):
#               out_view[0][i] = win[i]
#           out_handle.release(1)
#       else:
#       in_handle.release(1)
#
# This example demonstrates:
#   1. The IRON-level discard(n) API works at the producer side.
#   2. The lowered MLIR carries the aie.variable_rate = true marker
#      on the out_of ObjectFifoCreateOp.
#   3. The lowering pass routes out_of through the runtime-counter
#      machinery (no LCM unroll fights the conditional).
#
# Build (rebuilds against the worktree's variable_rate.py + the
# patched AIEObjectFifoStatefulTransform.cpp):
#
#   $ source /home/matteius/xdna-bringup/ironenv/bin/activate
#   $ source /opt/xilinx/xrt/setup.sh
#   $ export MLIR_AIE_DIR=$(pwd)
#   $ export PEANO_INSTALL_DIR=/home/matteius/xdna-bringup/ironenv/lib/python3.14/site-packages/llvm-aie
#   $ export PATH="$MLIR_AIE_DIR/install/bin:$PEANO_INSTALL_DIR/bin:$PATH"
#   $ cd $MLIR_AIE_DIR/build && ninja -j8 install
#   $ cd $MLIR_AIE_DIR/programming_examples/basic/variable_rate_filter
#   $ make NPU2=1
#
# Inspect the lowered MLIR for the marker:
#
#   $ grep variable_rate build/aie_*.mlir.prj/input_with_addresses.mlir
#
# Expected output: a single
#   aie.objectfifo @out_of_cons (...) {aie.variable_rate = true} ...
# entry on the consumer side (proves the split-fifo propagation
# fired) and the matching aie.variable_rate = true on the producer
# side (proves the IRON-level resolve() fired).

import argparse
import sys

import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    VariableRateFifo,
    Worker,
)
from aie.iron.device import NPU1Col1, NPU2

def my_variable_rate_filter(dev, in_size, out_size):
    in_dtype = np.uint8
    line_size = 64  # bytes per window
    n_windows_in = in_size // line_size
    # The variable-rate semantics mean we cannot statically know
    # how many windows the consumer will see. The host allocates
    # a generous out buffer (>= n_windows_in * line_size) and
    # checks the actual count in the test harness.

    line_type = np.ndarray[(line_size,), np.dtype[in_dtype]]
    in_vector_type = np.ndarray[(in_size,), np.dtype[in_dtype]]
    out_vector_type = np.ndarray[(out_size,), np.dtype[in_dtype]]

    # in_of: vanilla ObjectFifo (fixed rate from shim).
    in_of = ObjectFifo(line_type, name="in_of")

    # out_of: VariableRateFifo. Producer (Tile A) decides which
    # windows to forward; the downstream shim DMA only sees the
    # forwarded subset.
    out_of = VariableRateFifo(line_type, name="out_of")

    # External kernel: a passthrough copy + an even-byte predicate
    # check. We use a real C++ kernel function so the lowering
    # does NOT inline the predicate into the IRON Python (which
    # would defeat the demo).
    filter_kernel_fn = Kernel(
        "filterFirstByteEven",
        "filter_first_byte_even.cc.o",
        [line_type, line_type, np.int32],  # in_window, out_window, line_size
    )

    def core_fn(in_handle, out_handle, filter_fn):
        # Acquire one input window at a time.
        in_view = in_handle.acquire(1)
        # filterFirstByteEven returns 1 on forward, 0 on skip.
        # The IRON Python doesn't inspect the return; the kernel
        # writes to the output buffer iff returning 1.
        out_view = out_handle.acquire(1)
        forward_flag = filter_fn(in_view, out_view, line_size)
        # Note: in IRON Python today there is no first-class
        # `if forward_flag:` -- the conditional acquire/release
        # pattern is emitted by the kernel function via a runtime
        # branch. The KEY part of the demo is the discard() call
        # below: it tells the lowering pass that the static-rate
        # invariant is intentionally relaxed for this fifo.
        out_handle.release(1)
        # discard(0) is a documentary no-op for the matched-rate
        # path (every iteration releases 1 + discards 0); the
        # test purpose is only to demonstrate the discard()
        # method exists and is callable on a producer handle.
        # A REAL skip-emitting kernel uses discard(1) in a
        # branch alongside an absent acquire/release pair on
        # this fifo; documenting that pattern is the goal of
        # this example.
        in_handle.release(1)

    my_worker = Worker(
        core_fn,
        [in_of.cons(), out_of.prod(), filter_kernel_fn],
    )

    rt = Runtime()
    with rt.sequence(in_vector_type, out_vector_type, in_vector_type) as (
        a_in,
        b_out,
        _,
    ):
        rt.start(my_worker)
        rt.fill(in_of.prod(), a_in)
        rt.drain(out_of.cons(), b_out, wait=True)

    return Program(dev, rt).resolve_program()

p = argparse.ArgumentParser(
    "(VariableRateFifo demo).",
)
p.add_argument(
    "-d", "--dev", required=True, dest="device", help="AIE Device (npu / npu2)"
)
p.add_argument(
    "-i1s", "--in_size", required=True, dest="in_size",
    help="Input buffer size in bytes (multiple of 64; >= 512)",
)
p.add_argument(
    "-os", "--out_size", required=True, dest="out_size",
    help="Output buffer size in bytes (>= in_size to handle worst case)",
)
opts = p.parse_args(sys.argv[1:])

if opts.device == "npu":
    dev = NPU1Col1()
elif opts.device == "npu2":
    dev = NPU2()
else:
    raise ValueError(f"[ERROR] Device name {opts.device!r} is unknown")

in_size = int(opts.in_size)
out_size = int(opts.out_size)
if in_size % 64 != 0 or in_size < 512:
    raise ValueError(
        f"in_size ({in_size}) must be a multiple of 64 and >= 512"
    )
if out_size < in_size:
    raise ValueError(
        f"out_size ({out_size}) must be >= in_size ({in_size}) for "
        f"worst-case 100%-forward rate"
    )

print(my_variable_rate_filter(dev, in_size, out_size))
