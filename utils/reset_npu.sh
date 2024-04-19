#!/usr/bin/env bash
set -eux

NUMBER=$(lspci -D | grep "\[AMD\] Device 1502" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  sudo modprobe -r amdxdna
  sudo modprobe drm_shmem_helper
  sudo modprobe amdxdna

#  if [ -f "/opt/xilinx/xrt/test/example_noop_test" ]; then
#    /opt/xilinx/xrt/test/example_noop_test /lib/firmware/amdipu/1502/validate.xclbin
#  fi
else
  echo "couldn't find npu"
fi

