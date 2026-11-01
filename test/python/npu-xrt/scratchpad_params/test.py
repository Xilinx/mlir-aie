# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Python test for scratchpad parameter passing using ParameterScratchpad.
#
# REQUIRES: ryzen_ai_npu2, peano, xrt_python_bindings, xrt_python_ctrl_scratchpad_bo
#
# RUN: %python %S/aie_design.py > aie.mlir
# RUN: %aiecc -v --get-full-elf --dynamic-objFifos --get-scratchpad-parameters aie.mlir
# RUN: %run_on_npu2% %pytest %s
#
# This is the Python equivalent of the C++ test in ../scratchpad_params/.
# The design's core computes foo * bar from two bf16 ScratchpadParameters.
#   1. Through iron.jit: a call's parameters= writes them through the run's
#      ctrl scratchpad, from the build's own params.txt; two calls with
#      different values reuse the run.
#   2. Through pyxrt: aiecc compiles aie.mlir to aie.elf + params.txt, and a
#      run drains into another run's scratchpad alias.

import numpy as np
import pytest
import pyxrt
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import Out
from aie.utils.hostruntime.xrtruntime.device import acquire_device
from aie.utils.hostruntime.xrtruntime.parameter_scratchpad import (
    ParameterScratchpad,
)

import aie_design


@iron.jit(full_elf=True)
def multiply(out: Out):
    return aie_design.program().resolve_program()


@pytest.mark.parametrize(
    "foo,bar",
    [
        (bfloat16(3.0), bfloat16(4.0)),
        (bfloat16(2.0), bfloat16(5.0)),
    ],
)
def test_a_call_writes_its_parameters(foo, bar):
    out = iron.zeros((2,), dtype=bfloat16)
    multiply(out, parameters={"foo": foo, "bar": bar})
    assert float(out.numpy()[0]) == float(foo) * float(bar)


def test_a_parameter_the_design_does_not_declare_is_refused():
    out = iron.zeros((2,), dtype=bfloat16)
    with pytest.raises(RuntimeError, match="baz"):
        multiply(out, parameters={"foo": bfloat16(1.0), "baz": bfloat16(2.0)})


@pytest.fixture(scope="module")
def kernel_setup():
    device = acquire_device()
    elf = pyxrt.elf("aie.elf")
    context = pyxrt.hw_context(device, elf)
    kernel = pyxrt.ext.kernel(context, "test:sequence")

    # Output buffer: 2 x bf16 (only the first element is written by the core)
    out_tensor = iron.tensor((2,), dtype=bfloat16, device="cpu")

    run = pyxrt.run(kernel)
    run.set_arg(0, out_tensor.buffer_object())

    params = ParameterScratchpad(run, "params.txt", device=device)
    return run, params, out_tensor, kernel


def test_a_run_drains_into_another_runs_scratchpad_alias(kernel_setup):
    _, target, _, kernel = kernel_setup
    run = pyxrt.run(kernel)
    run.set_arg(0, target.alias())
    params = ParameterScratchpad(run, "params.txt", device=acquire_device())
    params.write("foo", bfloat16(3.0))
    params.write("bar", bfloat16(4.0))
    params.sync()

    run.start()
    run.wait2()

    # The run's first output element lands in the low half of word 0: the
    # parameter params.txt gives index 0 (entries are "name index type kind
    # min max" after the parameter and slot counts).
    with open("params.txt") as f:
        words = f.read().split()
    entries = [words[i : i + 6] for i in range(2, 2 + 6 * int(words[0]), 6)]
    (first,) = [name for name, index, *_ in entries if index == "0"]
    target.sync_from_device()
    assert target.read(first) & 0xFFFF == int(bfloat16(12.0).view(np.uint16))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
