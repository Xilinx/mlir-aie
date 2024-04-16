#!/bin/bash

# (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Execute a command while pre-staging for AIE driver some .xclbin firmware and
# clean up the firmware stage after execution.
# This script assumes that the path of one xclbin
# appears in the argument list

# Don't require root.
export XRT_HACK_UNSECURE_LOADING_XCLBIN=1 

XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh

# Execute the commands and its arguments
"$@"
err=$?

exit $err
