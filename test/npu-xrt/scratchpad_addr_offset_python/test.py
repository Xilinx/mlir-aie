# (c) Copyright 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Test for DMA address offset patching via offset_parameter (IRON flow).
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: aiecc.py -v --generate-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir
# RUN: cp aie.mlir.prj/params.txt .
# RUN: %run_on_npu2% %python %s
#
# Setup:
#   - Input buffer: 32 i32 values [0, 1, 2, ..., 31]
#   - Core: passthrough of 8 elements
#   - offset_parameter @input_offset controls the DMA read start position
#
# We run three times with different offsets and verify the output each time.

import sys

import numpy as np
import pyxrt

from aie.utils.parameter_scratchpad import ParameterScratchpad


def main():
    N_INPUT = 32
    N_OUTPUT = 8

    device = pyxrt.device(0)
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    # Input buffer: [0, 1, 2, ..., 31] as i32
    input_data = np.arange(N_INPUT, dtype=np.int32)
    bo_in = pyxrt.ext.bo(device, N_INPUT * 4)
    bo_in.write(input_data.tobytes(), 0)
    bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Output buffer: 8 x i32
    bo_out = pyxrt.ext.bo(device, N_OUTPUT * 4)

    run = pyxrt.run(kernel)
    run.set_arg(0, bo_in)
    run.set_arg(1, bo_out)

    params = ParameterScratchpad(run, "params.txt")

    test_cases = [
        (0, list(range(0, 8))),
        (8, list(range(8, 16))),
        (16, list(range(16, 24))),
    ]

    all_pass = True
    for run_idx, (offset, expected) in enumerate(test_cases, 1):
        # Clear output
        bo_out.write(bytes(N_OUTPUT * 4), 0)
        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

        # Write offset parameter (in elements)
        params.write("input_offset", np.int32(offset))
        params.sync()

        run.start()
        run.wait2()

        bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        mv = bo_out.map()
        result = np.frombuffer(bytes(mv[: N_OUTPUT * 4]), dtype=np.int32).tolist()

        status = "PASS" if result == expected else "FAIL"
        if result != expected:
            all_pass = False
        print(
            f"Run {run_idx} — offset={offset:2d}  expected={expected}  got={result}  {status}"
        )

    if all_pass:
        print("PASS!")
        return 0
    else:
        print("FAIL.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
