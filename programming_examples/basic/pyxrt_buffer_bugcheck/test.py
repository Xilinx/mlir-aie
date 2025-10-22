# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import pyxrt


def config(device, xclbin_path, instr_path):
    # Register xclbin and create a context for it
    xclbin = pyxrt.xclbin(xclbin_path)
    kernels = xclbin.get_kernels()
    xkernel = [k for k in kernels if "MLIR_AIE" == k.get_name()][0]

    device.register_xclbin(xclbin)
    context = pyxrt.hw_context(device, xclbin.get_uuid())
    kernel = pyxrt.kernel(context, xkernel.get_name())

    # Load instructions
    with open(instr_path, "rb") as f:
        insts = np.frombuffer(f.read(), dtype=np.uint32)

    return (context, kernel, insts)


def run1(device, context, kernel, insts, size):
    dtype = np.uint8

    in_volume = size // np.dtype(dtype).itemsize
    out_volume = size // np.dtype(dtype).itemsize

    # Initialize data for input, output, and testing
    in_data = np.arange(0, in_volume, dtype=dtype)
    out_data = np.zeros([out_volume], dtype=dtype)

    ref = in_data

    # Create instruction buffer (always needed)
    bo_insts = pyxrt.bo(device, len(insts) * 4, pyxrt.bo.cacheable, kernel.group_id(1))

    # Create input/output buffers, and populate them with data
    bo_in = pyxrt.bo(device, in_volume, pyxrt.bo.host_only, kernel.group_id(3))
    bo_in.write(in_data.view(np.uint8), 0)
    bo_out = pyxrt.bo(device, out_volume, pyxrt.bo.host_only, kernel.group_id(4))
    bo_out.write(out_data.view(np.uint8), 0)

    # Sync input buffers to device
    bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Run the kernel
    run = kernel(
        3,
        bo_insts,
        len(insts),
        bo_in,
        bo_out,
    )
    result = run.wait()
    if hasattr(pyxrt, "ert_cmd_state") and hasattr(pyxrt, "ERT_CMD_STATE_COMPLETED"):
        success = result == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    else:
        success = int(result) == 0

    # Sync data from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Read output buffer
    output_bytes = bo_out.read(out_volume, 0)
    output_data = np.frombuffer(output_bytes, dtype=dtype)

    # Check data
    print(ref)
    print(output_data)
    assert (ref == output_data).all()


def run2(device, context, kernel, insts, size):
    dtype = np.uint8

    in_volume = size // np.dtype(dtype).itemsize
    out_volume = size * 2 // np.dtype(dtype).itemsize

    # Initialize data for input, output, and testing
    in1_data = np.arange(0, in_volume, dtype=dtype)
    in2_data = np.arange(0, in_volume, dtype=dtype)
    out_data = np.zeros([out_volume], dtype=dtype)

    ref = np.concatenate((in1_data, in2_data))

    # Create instruction buffer (always needed)
    bo_insts = pyxrt.bo(device, len(insts) * 4, pyxrt.bo.cacheable, kernel.group_id(1))

    # Create input/output buffers, and populate them with data
    bo_in1 = pyxrt.bo(device, in1_data.nbytes, pyxrt.bo.host_only, kernel.group_id(3))
    bo_in1.write(in1_data.view(np.uint8), 0)
    bo_in2 = pyxrt.bo(device, in2_data.nbytes, pyxrt.bo.host_only, kernel.group_id(4))
    bo_in2.write(in2_data.view(np.uint8), 0)
    bo_out = pyxrt.bo(device, out_data.nbytes, pyxrt.bo.host_only, kernel.group_id(5))
    bo_out.write(out_data.view(np.uint8), 0)

    # Sync input buffers to device
    bo_insts.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in1.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_in2.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Run the kernel
    run = kernel(
        3,
        bo_insts,
        len(insts),
        bo_in1,
        bo_in2,
        bo_out,
    )
    result = run.wait()
    if hasattr(pyxrt, "ert_cmd_state") and hasattr(pyxrt, "ERT_CMD_STATE_COMPLETED"):
        success = result == pyxrt.ert_cmd_state.ERT_CMD_STATE_COMPLETED
    else:
        success = int(result) == 0

    # Sync data from device
    bo_out.sync(pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Read output buffer
    output_bytes = bo_out.read(out_volume, 0)
    output_data = np.frombuffer(output_bytes, dtype=dtype)

    # Check data
    print(ref)
    print(output_data)
    assert (ref == output_data).all()


def main(opts):
    device = pyxrt.device(0)

    # TODO: loop
    context1, kernel1, insts1 = config(device, opts.xclbin1, opts.insts1)
    run2(device, context1, kernel1, insts1, opts.size)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("-x1", "--xclbin1", type=str)
    p.add_argument("-i1", "--insts1", type=str)

    p.add_argument("-x2", "--xclbin2", type=str)
    p.add_argument("-i2", "--insts2", type=str)

    p.add_argument("-s", "--size", type=int)
    opts = p.parse_args()

    main(opts)
