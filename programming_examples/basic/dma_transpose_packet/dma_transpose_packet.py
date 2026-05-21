# dma_transpose_packet/dma_transpose_packet.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
"""2-D array transpose via shim DMA — Iron API with ``@iron.jit``.

The transpose is performed entirely at the input shim DMA via a
``TensorAccessPattern`` with strides ``[1, 1, 1, K]`` over an ``(M, K)``
input; the forwarding tile (selected by IRON) just relays.  Lowered
through aiecc with ``--packet-sw-objFifos`` so the ObjectFifo routes
end up as packet-switched flows — which is the lesson here, hence the
``_packet`` suffix on the directory.

Three invocation modes:

  * standalone:           ``python3 dma_transpose_packet.py``
  * compile-only:         ``... --xclbin-path=PATH --insts-path=PATH``
  * generate access map:  ``... --generate-access-map``  (writes
                           ``transpose_data.png`` and exits)
"""

import argparse
import sys

import numpy as np

import aie.iron as iron
from aie.iron import Compile, In, ObjectFifo, Out, Program, Runtime
from aie.iron.device import NPU1Col1, NPU2
from aie.helpers.taplib import TensorAccessPattern
from aie.utils.hostruntime import set_current_device


def _device_for(dev_str):
    if dev_str == "npu":
        return NPU1Col1()
    if dev_str == "npu2":
        return NPU2()
    raise ValueError(f"[ERROR] Device name {dev_str!r} is unknown")


def _transpose_tap(M: int, K: int) -> TensorAccessPattern:
    # Stride K in the innermost wrap skips a row → reads a full column.
    return TensorAccessPattern(
        (M, K), offset=0, sizes=[1, 1, K, M], strides=[1, 1, 1, K]
    )


@iron.jit(aiecc_flags=["--packet-sw-objFifos"])
def dma_transpose_packet(
    a_in: In,
    _b_unused: In,
    c_out: Out,
    *,
    M: Compile[int] = 64,
    K: Compile[int] = 32,
):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    of_in = ObjectFifo(tensor_ty, name="in")
    of_out = of_in.cons().forward()

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a, _, c):
        rt.fill(of_in.prod(), a, _transpose_tap(M, K))
        rt.drain(of_out.cons(), c, wait=True)

    return Program(iron.get_current_device(), rt).resolve_program()


def _make_argparser():
    p = argparse.ArgumentParser(prog="AIE 2-D DMA Transpose (packet-switched)")
    p.add_argument("-d", "--dev", type=str, choices=["npu", "npu2"], default="npu")
    p.add_argument("-M", "--M", type=int, default=64)
    p.add_argument("-K", "--K", type=int, default=32)
    p.add_argument(
        "--generate-access-map",
        action="store_true",
        help="write transpose_data.png and exit",
    )
    p.add_argument("--xclbin-path", type=str, default=None)
    p.add_argument("--insts-path", type=str, default=None)
    return p


def _compile_kwargs(opts):
    return dict(M=opts.M, K=opts.K)


def _compile_only(opts):
    if not opts.insts_path:
        sys.exit("--xclbin-path requires --insts-path (must be set together)")
    set_current_device(_device_for(opts.dev))
    spec = dma_transpose_packet.specialize(**_compile_kwargs(opts))
    spec.compile(xclbin_path=opts.xclbin_path, inst_path=opts.insts_path)


def _run_and_verify(opts):
    in_np = np.arange(1, opts.M * opts.K + 1, dtype=np.int32).reshape(opts.M, opts.K)
    b_np = np.zeros_like(in_np)
    out_np = np.zeros_like(in_np)

    a_t = iron.tensor(in_np.reshape(-1), dtype=np.int32, device="npu")
    b_t = iron.tensor(b_np.reshape(-1), dtype=np.int32, device="npu")
    c_t = iron.tensor(out_np.reshape(-1), dtype=np.int32, device="npu")

    dma_transpose_packet(a_t, b_t, c_t, **_compile_kwargs(opts))

    expected = in_np.T.reshape(-1)
    if not np.array_equal(c_t.numpy(), expected):
        sys.exit("FAIL! output does not match transpose(in)")
    print("PASS!")


def main():
    opts = _make_argparser().parse_args()
    if opts.generate_access_map:
        _transpose_tap(opts.M, opts.K).visualize(
            show_arrows=True,
            plot_access_count=False,
            file_path="transpose_data.png",
        )
        return
    if opts.xclbin_path:
        _compile_only(opts)
        return
    _run_and_verify(opts)


if __name__ == "__main__":
    main()
