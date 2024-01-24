#!/bin/bash

# (c) Copyright 2023 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

source /opt/xilinx/xrt/setup.sh
echo $PWD

for f in "$@"; do
  if [ -f "$f" ]; then
    filename=$(basename "$f")
    extension="${f##*.}"
    if [ "$extension" = "xclbin" ]; then
      /opt/xilinx/xrt/amdaie/setup_xclbin_firmware.sh -dev Phoenix -xclbin "$filename"
    fi
  fi
done

"$@"
err=$?

exit $err
