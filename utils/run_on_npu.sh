#!/bin/bash

# (c) Copyright 2023-2024 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

XRT_DIR=${XILINX_XRT:-/opt/xilinx/xrt}
if [ ! -d "$XRT_DIR" ]; then
    XRT_DIR=/usr
fi
if [ -f "$XRT_DIR/setup.sh" ]; then
    source $XRT_DIR/setup.sh
fi

# Execute the commands and its arguments
"$@"
err=$?

exit $err
