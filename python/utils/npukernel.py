# npukernel.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
from pathlib import Path
from . import DEFAULT_NPU_RUNTIME
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

    @property
    def trace_config(self) -> TraceConfig | None:
        return self._trace_config

    @property
    def xclbin_path(self):
        return self._xclbin_path

    @property
    def insts_path(self):
        return self._insts_path

    @property
    def kernel_name(self):
        return self._kernel_name

    # Blocking call.
    def __call__(self, *args):
        return DEFAULT_NPU_RUNTIME.load_and_run(
            [self],
            list(args),
            trace_config=self._trace_config,
        )
