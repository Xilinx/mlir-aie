# vector_vector_modulo/vector_vector_modulo.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024-2026 Advanced Micro Devices, Inc. or its affiliates
"""Element-wise vector % vector — Iron API design with ``@iron.jit``.

The design body delegates to ``aie.iron.algorithms.transform_binary_typed``,
which handles the ObjectFifo / Worker / Runtime plumbing.  The entry point
supports three invocation modes so the same file drives both the Ryzen AI
NPU @iron.jit pipeline and the legacy aiecc-based vck5000 (Versal AIE1) flow:

  * standalone:  ``python3 vector_vector_modulo.py``
        JIT-compile + run + verify via ``iron.tensor``.

  * compile-only:  ``... --xclbin-path=PATH --insts-path=PATH``
        Used by the NPU ``Makefile`` to drive @iron.jit's ``compile()`` and
        hand the artifacts to the C++ ``test.cpp`` host.

  * emit-MLIR:  ``... -d {npu,npu2,xcvc1902} --emit-mlir``
        Print the resolved MLIR module to stdout for the legacy aiecc
        Makefile rule (used by the vck5000 path; aiecc consumes the
        printed MLIR and produces ``test.elf`` via Chess + HSA).
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, Out
from aie.iron.algorithms import transform_binary_typed
from aie.iron.device import NPU1Col1, NPU2Col1, XCVC1902
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    if dev_str == "npu":
        return NPU1Col1()
    if dev_str == "npu2":
        return NPU2Col1()
    if dev_str == "xcvc1902":
        return XCVC1902()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


@iron.jit
def vector_vector_modulo(
    input0: In,
    input1: In,
    output: Out,
    *,
    num_elements: Compile[int] = 256,
    dtype: Compile[type] = np.int32,
    tile_size: Compile[int] = 16,
):
    tensor_ty = np.ndarray[(num_elements,), np.dtype[dtype]]
    return transform_binary_typed(lambda a, b: a % b, tensor_ty, tile_size=tile_size)


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE Vector Vector Modulo")
    p.add_argument(
        "-d", "--dev", type=str, choices=["npu", "npu2", "xcvc1902"], default="npu"
    )
    p.add_argument("-n", "--num-elements", type=int, default=256)
    p.add_argument("--tile-size", type=int, default=16)
    p.add_argument(
        "--emit-mlir",
        action="store_true",
        help="print the resolved MLIR module to stdout (legacy aiecc / vck5000 path)",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def _emit_mlir(opts):
    set_current_device(_device_for(opts.dev))
    print(
        vector_vector_modulo.as_mlir(
            None,
            None,
            None,
            num_elements=opts.num_elements,
            dtype=np.int32,
            tile_size=opts.tile_size,
        )
    )


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = vector_vector_modulo.specialize(
        num_elements=opts.num_elements,
        dtype=np.int32,
        tile_size=opts.tile_size,
    )
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    input0 = iron.randint(1, 100, (opts.num_elements,), dtype=np.int32, device="npu")
    input1 = iron.randint(1, 100, (opts.num_elements,), dtype=np.int32, device="npu")
    output = iron.zeros_like(input0)

    vector_vector_modulo(
        input0,
        input1,
        output,
        num_elements=opts.num_elements,
        dtype=np.int32,
        tile_size=opts.tile_size,
    )

    expected = input0.numpy() % input1.numpy()
    actual = output.numpy()
    errors = int(np.sum(actual != expected))

    if opts.verbose:
        for i in range(min(opts.num_elements, 16)):
            print(
                f"{i:3}: {int(input0[i]):4} % {int(input1[i]):4} = {int(output[i]):4}"
            )

    if errors:
        print(f"\nError count: {errors}\nFailed.\n")
        sys.exit(-1)
    print("\nPASS!\n")
    sys.exit(0)


def main():
    opts = _make_argparser().parse_args()
    if opts.emit_mlir:
        _emit_mlir(opts)
        return
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
