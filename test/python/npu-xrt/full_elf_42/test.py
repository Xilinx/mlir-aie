#!/usr/bin/env python3
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.
#
# Minimal full-ELF flow test: read back a single i32 value (42).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: aiecc -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime

N_OUTPUT = 1


def test_output_42():
    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "main:sequence")

    out_tensor = iron.tensor((N_OUTPUT,), dtype=np.int32, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, out_tensor.buffer_object())

    out_tensor.data.fill(0)
    out_tensor.to("npu")

    run.start()
    run.wait2()

    out_tensor.to("cpu")
    assert out_tensor.numpy().tolist() == [42]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
