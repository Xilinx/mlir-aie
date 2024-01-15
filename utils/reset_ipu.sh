#!/usr/bin/env bash
set -eux

NUMBER=$(lspci -D | grep "\[AMD\] Device 1502" | cut -d ' ' -f1)

if [ x"$NUMBER" != x"" ]; then
  rm -f /tmp/ipu.lock
  echo "1" > /sys/bus/pci/devices/0000:c5:00.1/remove
  sleep 1
  echo "1" > /sys/bus/pci/rescan

  if [ -f "/opt/xilinx/xrt/test/shim_test" ]; then
    /opt/xilinx/xrt/test/shim_test
  fi
else
  echo "couldn't find ipu"
fi

