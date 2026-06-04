# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Python test for scratchpad parameter passing using ParameterScratchpad.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: aiecc.py -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir
# RUN: cp aie.mlir.prj/params.txt .
# RUN: %run_on_npu2% %python %s
#
# This is the Python equivalent of the C++ test in ../scratchpad_params/.
# It exercises the full flow:
#   1. aiecc.py compiles aie.mlir → aie.elf + params.txt
#   2. This script loads the ELF, creates a ParameterScratchpad from params.txt,
#      writes bf16 parameters, and verifies the core computes foo * bar.
#   3. A second run with different values tests parameter re-use across runs.

import sys

import pyxrt
from ml_dtypes import bfloat16

from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor


def main():
    FOO_1, BAR_1 = bfloat16(3.0), bfloat16(4.0)
    FOO_2, BAR_2 = bfloat16(2.0), bfloat16(5.0)

    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    # Output buffer: 2 x bf16 (only the first element is written by the core)
    out_tensor = XRTTensor((2,), dtype=bfloat16)

    run = pyxrt.run(kernel)
    run.set_arg(0, out_tensor.buffer_object())

    params = ParameterScratchpad(run, "params.txt")

    def run_once(foo, bar):
        out_tensor.data.fill(0)
        out_tensor.to("npu")

        params.write("foo", foo)
        params.write("bar", bar)
        params.sync()

        run.start()
        run.wait2()

        out_tensor.to("cpu")
        return float(out_tensor.numpy()[0])

    # --- Run 1: foo=3.0, bar=4.0 → expect 12.0 ---
    result1 = run_once(FOO_1, BAR_1)
    expected1 = float(FOO_1) * float(BAR_1)
    print(f"Run 1 — Expected: {expected1}, Got: {result1}")

    # --- Run 2: foo=2.0, bar=5.0 → expect 10.0 ---
    result2 = run_once(FOO_2, BAR_2)
    expected2 = float(FOO_2) * float(BAR_2)
    print(f"Run 2 — Expected: {expected2}, Got: {result2}")

    if result1 == expected1 and result2 == expected2:
        print("PASS!")
        return 0
    else:
        print("FAIL.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
