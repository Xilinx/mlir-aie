# variable_rate_filter/variable_rate_filter.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
#
# A producer worker reads input windows from a fixed-rate upstream
# ObjectFifo. On every other window it forwards the window to a
# downstream VariableRateFifo via acquire/release; on the alternate
# window it calls ``discard(1)`` on the producer handle, telling the
# variable-rate fifo to skip that slot without forwarding. The
# downstream consumer (the host runtime drain) sees only the
# forwarded windows.
#
# Topology:
#
#   shim DMA (host) --> in_of (ObjectFifo) --> Tile A (filter) -->
#       out_of (VariableRateFifo) --> shim DMA (host)
#
# This example demonstrates:
#   1. The IRON-level discard(n) API works at the producer side and
#      is invoked in a branch where the producer chooses not to forward.
#   2. The lowered MLIR carries the aie.variable_rate = true marker
#      on the out_of ObjectFifoCreateOp.
#   3. The lowering pass routes out_of through the runtime-counter
#      machinery (no LCM unroll fights the asymmetric rates).
#
# Note: a richer "predicate decided in the C++ kernel per window"
# variant is not expressible at the current IRON-Python layer without
# a first-class scf.if lowering on the conditional acquire/release.
# This example uses a Python-level deterministic skip pattern, which
# still exercises the discard() API and the variable_rate marker
# end-to-end.
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

    # External C++ kernel: a window-copy callable invoked on the
    # forwarded iterations. The skip decision is made at the IRON
    # Python layer by the alternating pattern in ``core_fn`` below;
    # this kernel is intentionally a pure copy so the producer-side
    # state machine (acquire+release on forward, discard(1) on skip)
    # is the visible mechanism in the example.
    filter_kernel_fn = Kernel(
        "filterFirstByteEven",
        "filter_first_byte_even.cc.o",
        [line_type, line_type, np.int32],  # in_window, out_window, line_size
    )

    # Producer worker. Uses a Python-level deterministic skip pattern
    # — every other window is forwarded, the alternate window is
    # discarded via the producer-side ``discard(1)`` API. This demonstrates
    # the variable-rate codepath: the producer's loop has asymmetric
    # acquire / release counts on the variable-rate output fifo (acquire+
    # release on forwarded iterations, discard on skipped ones) which the
    # ``aie.variable_rate = true`` marker tells the lowering pass to
    # accept (LCM-based loop unrolling is bypassed; the runtime-counter
    # machinery handles the asymmetric rates).
    #
    # Predicate-based skip (where the C++ kernel decides per window) is
    # sketched in the module-level comment but isn't expressible at the
    # current IRON-Python layer without a first-class ``scf.if`` lowering.
    def core_fn(in_handle, out_handle, filter_fn):
        # Two iterations per worker invocation: forward, then discard.
        # The Worker's outer while-true loop drives the cycle.
        for skip in range(2):
            in_view = in_handle.acquire(1)
            if skip == 0:
                out_view = out_handle.acquire(1)
                # Kernel performs the copy on the forwarded iteration.
                filter_fn(in_view, out_view, line_size)
                out_handle.release(1)
            else:
                # Skipped iteration: tell the variable-rate fifo we
                # intentionally chose not to forward this slot. The
                # consumer's count is unaffected; only the producer's
                # discard counter increments.
                out_handle.discard(1)
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
