# (c) Copyright 2025 Advanced Micro Devices, Inc.
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

import struct
import sys

import pyxrt
from ml_dtypes import bfloat16

from aie.utils.parameter_scratchpad import ParameterScratchpad


def read_bf16(bo, offset):
    """Read a bfloat16 value from a buffer object at the given byte offset."""
    mv = bo.map()
    raw = bytes(mv[offset : offset + 2])
    # Reconstruct float32 from bfloat16 (upper 16 bits of float32)
    f32_bytes = b"\x00\x00" + raw
    return struct.unpack("<f", f32_bytes)[0]


def main():
    FOO_1, BAR_1 = bfloat16(3.0), bfloat16(4.0)
    FOO_2, BAR_2 = bfloat16(2.0), bfloat16(5.0)

    device = pyxrt.device(0)
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    # Output buffer: 2 x bf16 = 4 bytes
    bo_out = pyxrt.ext.bo(device, 4)
    bo_out.write(bytes(4), 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run = pyxrt.run(kernel)
    run.set_arg(0, bo_out)

    params = ParameterScratchpad(run, "params.txt")

    # --- Run 1: foo=3.0, bar=4.0 → expect 12.0 ---
    params.write("foo", FOO_1)
    params.write("bar", BAR_1)
    params.sync()

    run.start()
    run.wait2()
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    result1 = read_bf16(bo_out, 0)
    expected1 = float(FOO_1) * float(BAR_1)
    print(f"Run 1 — Expected: {expected1}, Got: {result1}")

    # --- Run 2: foo=2.0, bar=5.0 → expect 10.0 ---
    params.write("foo", FOO_2)
    params.write("bar", BAR_2)
    params.sync()
    bo_out.write(bytes(4), 0)
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    run.start()
    run.wait2()
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    result2 = read_bf16(bo_out, 0)
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
