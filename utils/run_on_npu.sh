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

# Ensure pyxrt is discoverable (Ubuntu package puts it in system dist-packages)
if ! python3 -c "import pyxrt" 2>/dev/null; then
    PYXRT_DIR=$(python3 -c "
import glob, sys, os
for p in glob.glob('/usr/lib/python3*/dist-packages/pyxrt*.so'):
    print(os.path.dirname(p)); sys.exit(0)
for p in glob.glob('/usr/lib/python3/dist-packages/pyxrt*.so'):
    print(os.path.dirname(p)); sys.exit(0)
" 2>/dev/null)
    if [ -n "$PYXRT_DIR" ]; then
        export PYTHONPATH=${PYXRT_DIR}:${PYTHONPATH}
    fi
fi

# Execute the commands and its arguments
"$@"
err=$?

exit $err
