#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 AMD Inc.

# Simple test for the passthrough 3D kernel

import argparse
import numpy as np
import pyxrt as xrt
import sys
import time


def test_passthrough(xclbin_path, insts_path, depth, height, width, channels):
    """Test the minimal passthrough kernel"""

    # Calculate sizes
    plane_size = height * width * channels
    tensor_size = depth * plane_size

    print(f"\n=== Passthrough 3D Test ===")
    print(f"Depth: {depth}, Height: {height}, Width: {width}, Channels: {channels}")
    print(f"Plane size: {plane_size} bytes")
    print(f"Total tensor size: {tensor_size} bytes")

    # Load device and xclbin
    print(f"\nLoading xclbin: {xclbin_path}")
    device = xrt.device(0)
    xclbin = xrt.xclbin(xclbin_path)
    device.register_xclbin(xclbin)

    # Get context
    context = xrt.hw_context(device, xclbin.get_uuid())

    # Load kernel and instructions
    kernel = xrt.kernel(context, "MLIR_AIE")

    with open(insts_path, "rb") as f:
        instr = f.read()

    print(f"Loaded {len(instr)} bytes of instructions")

    # Allocate buffers
    bo_instr = xrt.bo(device, len(instr), xrt.bo.flags.cacheable, kernel.group_id(0))
    bo_input = xrt.bo(device, tensor_size, xrt.bo.flags.host_only, kernel.group_id(2))
    bo_output = xrt.bo(device, tensor_size, xrt.bo.flags.host_only, kernel.group_id(3))

    # Create test input pattern - simple incrementing values
    input_data = np.arange(tensor_size, dtype=np.uint8)

    print(f"\nInput data range: {input_data.min()} to {input_data.max()}")
    print(f"First 16 values: {input_data[:16]}")

    # Copy data to device
    bo_instr.write(instr, 0)
    bo_input.write(input_data, 0)

    # Sync to device
    bo_instr.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
    bo_input.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

    # Run kernel
    print("\nRunning kernel...")
    start = time.perf_counter()

    h = kernel(bo_instr, len(instr), bo_input, bo_output)
    h.wait()

    elapsed = time.perf_counter() - start
    print(f"Kernel completed in {elapsed*1000:.2f} ms")

    # Sync output back
    bo_output.sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)

    # Read output
    output_data = np.zeros(tensor_size, dtype=np.uint8)
    bo_output.read(output_data, 0)

    print(f"\nOutput data range: {output_data.min()} to {output_data.max()}")
    print(f"First 16 values: {output_data[:16]}")

    # Verify passthrough
    mismatches = np.sum(input_data != output_data)

    if mismatches == 0:
        print("\n*** SUCCESS: Output matches input perfectly! ***")
        print(f"All {tensor_size} bytes passed through correctly")
        return 0
    else:
        print(f"\n*** FAILURE: {mismatches}/{tensor_size} bytes mismatched ***")

        # Show first few mismatches
        mismatch_indices = np.where(input_data != output_data)[0]
        print("\nFirst 10 mismatches:")
        for i in mismatch_indices[:10]:
            print(f"  Index {i}: expected {input_data[i]}, got {output_data[i]}")

        return 1


def main():
    parser = argparse.ArgumentParser(description="Test passthrough 3D kernel")
    parser.add_argument("-x", "--xclbin", required=True, help="Path to xclbin file")
    parser.add_argument("-i", "--insts", required=True, help="Path to instructions file")
    parser.add_argument("-d", "--depth", type=int, default=4, help="Depth dimension")
    parser.add_argument("-ht", "--height", type=int, default=4, help="Height dimension")
    parser.add_argument("-wd", "--width", type=int, default=4, help="Width dimension")
    parser.add_argument("-c", "--channels", type=int, default=8, help="Number of channels")

    args = parser.parse_args()

    try:
        return test_passthrough(
            args.xclbin,
            args.insts,
            args.depth,
            args.height,
            args.width,
            args.channels,
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
