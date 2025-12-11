# npukernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
from pathlib import Path
from . import DEFAULT_IRON_RUNTIME
from .hostruntime import TraceConfig


class NPUKernel:
    def __init__(
        self,
        xclbin_path,
        insts_path,
        device_index=0,
        kernel_name="PP_FD_PRE",
        trace_config: TraceConfig | None = None,
    ):
        self._xclbin_path = xclbin_path
        self._insts_path = insts_path
        self._kernel_name = kernel_name
        self._trace_config = trace_config

    # Blocking call.
    def __call__(self, *args):
        return DEFAULT_IRON_RUNTIME.load_and_run(
            [Path(self._xclbin_path), Path(self._insts_path), self._kernel_name],
            list(args),
            trace_config=self._trace_config,
        )
