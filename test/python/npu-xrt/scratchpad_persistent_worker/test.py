# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# A persistent Worker reads each dispatch's ScratchpadParameter value, not a
# previous dispatch's: Worker(wait_for_parameters=True) acquires the core's
# parameter-sync lock at the top of every iteration, which the sequence's sync
# sets once its values are written.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pytest
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

VALUES = (1, 5, 2, 9, 7, 3, 1 << 20, 0)


def test_each_dispatch_reads_its_own_value():
    device = XRTHostRuntime()._device
    context = pyxrt.hw_context(device, pyxrt.elf("aie.elf"))
    kernel = pyxrt.ext.kernel(context, "test:sequence")
    out = iron.tensor((2,), dtype=np.int32, device="cpu")
    run = pyxrt.run(kernel)
    run.set_arg(0, out.buffer_object())
    params = ParameterScratchpad(run, "params.txt")

    seen = []
    for value in VALUES:
        out.data.fill(-1)
        out.to("npu")
        params.write("n", value)
        params.sync()
        run.start()
        run.wait2()
        out.to("cpu")
        seen.append(tuple(int(v) for v in out.numpy()))

    # The iteration count proves the core ran on across dispatches rather than
    # being reset, which is the case the fence is for.
    assert [count for _, count in seen] == list(range(1, len(VALUES) + 1)), seen
    assert [value for value, _ in seen] == list(VALUES), seen


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
