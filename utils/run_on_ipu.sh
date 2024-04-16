#!/bin/bash

# (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Execute a command while pre-staging for AIE driver some .xclbin firmware and
# clean up the firmware stage after execution.
# This script assumes that the path of one xclbin
# appears in the argument list

# Old AIE driver
linux_driver_version=""
# Where the device driver finds the firmware to be loaded.
FIRMWARE_DIR=/lib/firmware/amdipu/1502
# The AIE device model
NPU_DEVICE=Phoenix
if [[ -d /lib/firmware/amdnpu/1502_00 ]]; then
  # New AIE driver naming and firmware location starting with Linux 6.8
  linux_driver_version="6.8"
  FIRMWARE_DIR=/lib/firmware/amdnpu/1502_00
  NPU_DEVICE=NPU1
fi

# Don't require root.
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 

XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh

# Execute the commands and its arguments
"$@"
err=$?

exit $err
