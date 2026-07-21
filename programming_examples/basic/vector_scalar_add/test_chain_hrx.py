# test_chain_hrx.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""HRX multi-dispatch / chain (runlist) test for vector_scalar_add.

The test runs two dispatches of the same ``x + 1`` kernel as a single batched
submit (``HRXHostRuntime.run_chain``), where the second run consumes the first
run's output:

    run0:  out0 = inA  + 1        (inA[i] = i + 1  ->  out0[i] = i + 2)
    run1:  out1 = out0 + 1        (out1[i] = i + 3)

If chaining/ordering were broken, ``run1`` would read ``out0`` before ``run0``
wrote it and ``out1`` would be wrong (or zero). Validating ``out1 == i + 3``
proves the in-chain producer->consumer dependency holds.

Usage (after ``make all`` built build/final.xclbin + build/insts.bin):

    IRON_RUNTIME=hrx python3 test_chain_hrx.py \
        --xclbin build/final.xclbin --instr build/insts.bin --kernel MLIR_AIE
"""

import argparse
import sys

import numpy as np

from aie.utils.hostruntime.hrxruntime.hostruntime import HRXHostRuntime
from aie.utils.hostruntime.hrxruntime.tensor import HRXTensor
from aie.utils.npukernel import NPUKernel


def main():
    p = argparse.ArgumentParser(prog="HRX vector_scalar_add chain/runlist test")
    p.add_argument("-x", "--xclbin", default="build/final.xclbin")
    p.add_argument("-i", "--instr", default="build/insts.bin")
    p.add_argument("-k", "--kernel", default="MLIR_AIE")
    p.add_argument("-n", "--size", type=int, default=1024)
    opts = p.parse_args()

    size = opts.size
    rt = HRXHostRuntime()
    kernel = NPUKernel(opts.xclbin, opts.instr, kernel_name=opts.kernel)
    handle = rt.load(kernel)

    # inA[i] = i + 1 ; out0/out1 start zeroed.
    in_a = HRXTensor(np.arange(1, size + 1, dtype=np.int32), dtype=np.int32)
    out0 = HRXTensor((size,), dtype=np.int32)
    out1 = HRXTensor((size,), dtype=np.int32)

    # Two chained runs in one batched submit (single ERT_CMD_CHAIN).
    rt.run_chain([(handle, [in_a, out0]), (handle, [out0, out1])])

    got0 = out0.numpy()
    got1 = out1.numpy()
    ref0 = np.arange(1, size + 1, dtype=np.int32) + 1  # i + 2
    ref1 = ref0 + 1  # i + 3

    errors = 0
    if not np.array_equal(got0, ref0):
        bad = int(np.argmax(got0 != ref0))
        print(f"run0 mismatch @ {bad}: got {got0[bad]} != {ref0[bad]}")
        errors += int(np.count_nonzero(got0 != ref0))
    if not np.array_equal(got1, ref1):
        bad = int(np.argmax(got1 != ref1))
        print(f"run1 mismatch @ {bad}: got {got1[bad]} != {ref1[bad]}")
        errors += int(np.count_nonzero(got1 != ref1))

    if errors:
        print(f"\nfailed ({errors} mismatches).\n")
        return 1
    print(f"run0: out0 == in + 1  (first 4: {got0[:4]})")
    print(f"run1: out1 == out0 + 1 (first 4: {got1[:4]})")

    # Deeper chain: depth links threaded through distinct buffers in one submit;
    # stage k must equal in + (k + 1). Stresses the inter-dispatch barrier across
    # many dispatches in a single ERT_CMD_CHAIN.
    depth = 8
    stages = [HRXTensor((size,), dtype=np.int32) for _ in range(depth)]
    chain = [(handle, [in_a, stages[0]])]
    for k in range(1, depth):
        chain.append((handle, [stages[k - 1], stages[k]]))
    rt.run_chain(chain)

    base = np.arange(1, size + 1, dtype=np.int32)
    for k, st in enumerate(stages):
        ref = base + (k + 1)
        got = st.numpy()
        if not np.array_equal(got, ref):
            bad = int(np.argmax(got != ref))
            print(
                f"deep-chain stage {k} mismatch @ {bad}: got {got[bad]} != {ref[bad]}"
            )
            errors += int(np.count_nonzero(got != ref))
    if errors:
        print(f"\nfailed ({errors} mismatches in deep chain).\n")
        return 1
    print(
        f"deep chain (depth {depth}): stage k == in + (k+1)  "
        f"[stage {depth - 1} first 4: {stages[-1].numpy()[:4]}]"
    )

    print("\nPASS!\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
