# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# IRON design for the persistent-worker scratchpad test: a while_true Worker
# runs one loop iteration per dispatch and writes [n, iterations so far], where
# n is the ScratchpadParameter the host set for that dispatch. The core is not
# reset between dispatches (the full-ELF sequence configures the device only
# when its PDI changes), so the iteration count keeps growing.
#
# Usage:
#   python3 aie_design.py [--fence {on,off}] > aie.mlir

import argparse

import numpy as np

from aie.dialects.arith import addi, constant
from aie.iron import (
    Buffer,
    DeviceConfiguration,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    Worker,
)
from aie.iron.device import NPU2Col1
from aie.ir import IntegerType


def design(fence: bool) -> str:
    out_ty = np.ndarray[(2,), np.dtype[np.int32]]
    count_ty = np.ndarray[(1,), np.dtype[np.int32]]

    n = ScratchpadParameter("n", np.int32)
    of_out = ObjectFifo(out_ty, name="objfifo_out")
    count = Buffer(count_ty, initial_value=np.zeros(1, np.int32), name="count")

    def core_fn(of_out, n, count):
        value = n.read()
        iterations = addi(count[0], constant(IntegerType.get_signless(32), 1))
        count[0] = iterations
        elem = of_out.acquire(1)
        elem[0] = value
        elem[1] = iterations
        of_out.release(1)

    worker = Worker(core_fn, [of_out.prod(), n, count], wait_for_parameters=fence)

    def sequence(out_tensor, out_h):
        out_h.drain(out_tensor, wait=True)

    rt = Runtime(sequence, [out_ty, of_out.cons()], name="sequence")
    config = DeviceConfiguration("test", NPU2Col1(), workers=[worker], runtimes=[rt])
    return str(Program.compose([config], entry=rt).resolve_program())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fence",
        choices=("on", "off"),
        default="on",
        help="wait for each dispatch's parameters at the top of every iteration",
    )
    args = parser.parse_args()
    print(design(args.fence == "on"))


if __name__ == "__main__":
    main()
