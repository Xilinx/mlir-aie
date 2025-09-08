# config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import subprocess

from .device import NPU1, NPU2


# Detect WSL
def is_wsl() -> bool:
    try:
        with open("/proc/sys/kernel/osrelease", "r", encoding="utf-8") as kernel:
            return "microsoft" in kernel.read().lower()
    except OSError:
        return False


# Prefer Windows xrt-smi when in WSL. Linux native otherwise.
def xrt_smi_path() -> str:
    if is_wsl():
        return "/mnt/c/Windows/System32/AMD/xrt-smi.exe"
    return "/opt/xilinx/xrt/bin/xrt-smi"


def detect_npu_device():
    """Detects the current device in the system.
       This assumes XRT and XDNA driver is installed
       and the system has NPU hardware.

    Returns:
        The current system device.
    """
    try:
        # Run `xrt-smi examine` and capture output
        xrt_smi = xrt_smi_path()
        result = subprocess.run(
            [xrt_smi, "examine"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        output = result.stdout

        # Match strings for NPU2 or NPU1
        # Sets generic "whole array" devices. Overkill.
        if any(
            keyword.lower() in output.lower()
            for keyword in [
                "NPU Strix",
                "NPU Strix Halo",
                "NPU Krackan",
                "RyzenAI-npu4",
                "RyzenAI-npu6",
            ]
        ):
            return NPU2()
        elif any(
            keyword.lower() in output.lower()
            for keyword in [
                "NPU",
                "NPU Phoenix",
                "RyzenAI-npu1",
            ]
        ):
            return NPU1()
        else:
            raise RuntimeError("No supported NPU device found.")

    except FileNotFoundError:
        if is_wsl():
            raise RuntimeError(
                "WSL detected but Windows xrt-smi.exe not found. Install AMD Ryzen AI Software."
            )
        raise RuntimeError("xrt-smi not found. Make sure XRT is installed.")
    except subprocess.CalledProcessError:
        raise RuntimeError("Failed to run xrt-smi examine.")


config = {}

config["device"] = detect_npu_device()


def set_current_device(device):
    """Sets the current device.

    Args:
        device: Device to set as the current device.

    Returns:
        The previously set device.
    """
    global config
    previous_device = config.get("device")
    config["device"] = device
    return previous_device


def get_current_device():
    """Gets the current device.

    Returns:
        The currently set device.
    """
    global config
    return config["device"]
