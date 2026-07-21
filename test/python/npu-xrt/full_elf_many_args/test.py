# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# End-to-end full-ELF test with MORE THAN 5 host buffers.
#
# It demonstrates the DDR address-patch bug in AIETargetNPU.cpp: appendAddressPatch
# folds a 0x80000000 AIE-aperture offset into arg_plus for arg_idx >= 5. That
# fold is correct for the xclbin + instruction-buffer runtime (HOST_ONLY BOs
# passed as kernel(opcode, insts_bo, ninsts, ...), where the firmware translates
# only the first 5 buffers), and it is exercised/passing by npu-xrt/many_buffers.
#
# But the full-ELF runtime (xrt.elf + xrt.ext.kernel + run.set_arg) translates
# all buffer addresses, so the fold double-translates the 6th and later buffers:
# their shim DMA lands ~2 GiB off and the host reads back 0.
#
# EXPECTED on today's mlir-aie: arg 0..4 read back correctly (100..104), arg 5..7
# read back 0 -> this test FAILS. With the offset suppressed on the full-ELF path
# it PASSES for all 8 buffers.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: aiecc -v --get-full-elf --no-xchesscc --no-xbridge --dynamic-objFifos aie.mlir
# RUN: %run_on_npu2% %pytest %s

import numpy as np
import pyxrt

import aie.iron as iron
from aie.utils.hostruntime.xrtruntime.hostruntime import XRTHostRuntime

N_BUFFERS = 8


def test_full_elf_many_args():
    runtime = XRTHostRuntime()
    device = runtime._device
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "main:sequence")

    tensors = [
        iron.tensor((1,), dtype=np.int32, device="cpu") for _ in range(N_BUFFERS)
    ]
    run = pyxrt.run(kernel)
    for i, t in enumerate(tensors):
        t.data.fill(0)
        t.to("npu")
        run.set_arg(i, t.buffer_object())

    run.start()
    run.wait2()

    for t in tensors:
        t.to("cpu")

    actual = [int(t.numpy()[0]) for t in tensors]
    expected = [100 + i for i in range(N_BUFFERS)]
    # Per-buffer detail makes the arg_idx >= 5 boundary obvious on failure.
    detail = "\n".join(
        f"  arg{i}: expected={expected[i]} actual={actual[i]}"
        + ("" if actual[i] == expected[i] else "  <-- MISMATCH")
        for i in range(N_BUFFERS)
    )
    assert actual == expected, f"host buffers beyond arg 4 were corrupted:\n{detail}"


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v"])
