#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# run.lit launches this, so lit itself does not run it.
# REQUIRES: dont_run
# RUN: echo FAIL | FileCheck %s
# CHECK: PASS

"""Runs one sequence of the head-of-line design and checks the core's sum
of everything it received against numpy."""

import argparse
import sys

import numpy as np
import pyxrt

WORDS_A = 9


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--xclbin", required=True)
    cli.add_argument("--insts", required=True)
    args = cli.parse_args()

    a = np.arange(1, WORDS_A + 1, dtype=np.int32)
    b = np.array([100, 200, 300, 400], dtype=np.int32)

    dev = pyxrt.device(0)
    xclbin = pyxrt.xclbin(args.xclbin)
    dev.register_xclbin(xclbin)
    ctx = pyxrt.hw_context(dev, xclbin.get_uuid())
    kernel = pyxrt.kernel(ctx, "MLIR_AIE")
    insts = np.fromfile(args.insts, dtype=np.uint32)

    bos = []
    for data, group, flags in (
        (insts, 1, pyxrt.bo.cacheable),
        (a, 3, pyxrt.bo.host_only),
        (b, 4, pyxrt.bo.host_only),
        (np.full(1, -1, dtype=np.int32), 5, pyxrt.bo.host_only),
    ):
        bo = pyxrt.bo(dev, data.nbytes, flags, kernel.group_id(group))
        np.frombuffer(bo.map(), dtype=data.dtype, count=data.size)[:] = data
        bo.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        bos.append(bo)

    state = kernel(3, bos[0], insts.size, *bos[1:]).wait()
    bos[3].sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
    got = int(np.frombuffer(bos[3].map(), dtype=np.int32, count=1)[0])
    want = int(a.sum() + b.sum())
    if state != pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED or got != want:
        print(f"FAIL: {state}, got {got}, want {want}")
        sys.exit(1)
    print("PASS!")


if __name__ == "__main__":
    main()
