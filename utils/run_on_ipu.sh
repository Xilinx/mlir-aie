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

XRT_DIR=/opt/xilinx/xrt
source $XRT_DIR/setup.sh

# Get the UUID of the XCLBIN, used by the Linux kernel driver to access it
xclbin_uuid() {
  local xclbin_file_name=$1
  local uuid=$($XRT_DIR/bin/xclbinutil --info -i $xclbin_file_name \
                 | grep 'UUID (xclbin)' | awk '{print $3}')
  echo "$uuid"
}

# Install the XCLBIN file so it can be consumed by the Linux kernel driver
stage_xclbin() {
  local npu_device=$1
  local xclbin_file_name=$2
  if [[ $linux_driver_version ]]; then
    # There is a bug in the device driver which prevent loading an XCLBIN if it
    # is accessed through a symbolic link which is not owned by root. So just
    # skip the symbolic link installed by the official XDNA XRT staging script
    # and do some manual work instead
    cp -a $xclbin_file_name \
       $FIRMWARE_DIR/$(xclbin_uuid $xclbin_file_name).xclbin
  else
    # Use the official XDNA XRT staging script which requires running as root
    $XRT_DIR/amdxdna/setup_xclbin_firmware.sh \
      -dev $npu_device -xclbin $xclbin_file_name
  fi
}

# Unstage the XCLBIN file
rm_xclbin() {
  local XCLBIN_FN=$1
  if [ -f "$XCLBIN_FN" ] && [ x"${XCLBIN_FN##*.}" == x"xclbin" ]; then
    local UUID=$(xclbin_uuid $XCLBIN_FN)
    if [[ $linux_driver_version ]]; then
      # Remove the XCLBIN file
      rm $FIRMWARE_DIR/$UUID.xclbin
    else
      # Remove the XCLBIN file pointed by the link
      rm -rf "$(readlink -f $FIRMWARE_DIR/$UUID.xclbin)"
      # Remove the link
      unlink $FIRMWARE_DIR/$UUID.xclbin
    fi
  fi

  # -xtype l tests for links that are broken (it is the opposite of -type)
  find $FIRMWARE_DIR -xtype l -delete;
}

if [ x"$1" == x"--clean-xclbin" ]; then
  rm_xclbin $2
  exit 0
fi

XCLBIN_FN=""
# Analyze all the command arguments to find some .xclbin
for f in "$@"; do
  if [ -f "$f" ]; then
    filename=$(basename "$f")
    extension="${f##*.}"
    if [ x"$extension" = x"xclbin" ]; then
      XCLBIN_FN="$filename"
      # Stage the .xclbin to be loaded by the command execution later
      stage_xclbin $NPU_DEVICE $XCLBIN_FN
      break
    fi
  fi
done

# Execute the commands and its arguments
"$@"
err=$?

# TODO: use setup_xclbin_firmware.sh -clean instead
if [ x"$XCLBIN_FN" != x"" ]; then
  rm_xclbin $XCLBIN_FN
fi

exit $err
