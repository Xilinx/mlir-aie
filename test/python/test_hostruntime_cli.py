# test_hostruntime_cli.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

# RUN: %pytest %s
"""Unit tests for the shared IRON example CLI dispatcher."""

from argparse import Namespace

from aie.utils import get_current_device
from aie.utils.hostruntime import set_current_device
from aie.utils.hostruntime.cli import run_design_cli


def test_run_design_cli_binds_device_before_validate_and_run():
    """The selected CLI device is current before any user callback runs."""
    opts = Namespace(dev="npu2", emit_mlir=False, xclbin_path=None, insts_path=None)
    seen = []

    def validate(_opts):
        seen.append(type(get_current_device(probe_runtime=False)).__name__)

    def run_and_verify(_opts):
        seen.append(type(get_current_device(probe_runtime=False)).__name__)

    try:
        run_design_cli(
            object(),
            opts,
            compile_kwargs={},
            run_and_verify=run_and_verify,
            validate=validate,
        )
    finally:
        set_current_device(None)

    assert seen == ["NPU2Col1", "NPU2Col1"]
