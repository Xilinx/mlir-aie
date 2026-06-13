# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc.
"""Host-harness helpers shared by the ``test.py``-style example drivers.

CLI flag wiring lives in :mod:`aie.utils.hostruntime.argparse` —
specifically :func:`add_runtime_args`.  This module only exposes the
NPU-kernel construction helper.
"""

from aie.utils import NPUKernel, TraceConfig


def create_npu_kernel(opts):
    """Build an :class:`NPUKernel` (with optional trace config) from parsed CLI opts.

    Reads ``opts.xclbin``, ``opts.instr``, ``opts.kernel``, and the
    optional trace fields (``trace_size``, ``trace_file``, ``ddr_id``,
    ``enable_ctrl_pkts``) — all produced by
    :func:`aie.utils.hostruntime.argparse.add_runtime_args`.

    Stashes the resulting kernel on ``opts.npu_kernel`` and returns
    ``opts`` for chaining.
    """
    trace_config = None
    trace_size = getattr(opts, "trace_size", 0)
    if trace_size > 0:
        trace_config = TraceConfig(
            trace_size=trace_size,
            trace_file=getattr(opts, "trace_file", "trace.txt"),
            ddr_id=getattr(opts, "ddr_id", 4),
            enable_ctrl_pkts=getattr(opts, "enable_ctrl_pkts", False),
        )
    opts.npu_kernel = NPUKernel(
        xclbin_path=opts.xclbin,
        insts_path=opts.instr,
        kernel_name=opts.kernel,
        trace_config=trace_config,
    )
    return opts
