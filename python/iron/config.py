# config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.
import shutil
import subprocess

from .device import NPU1, NPU2

def detect_npu_device():
    """Detects the current device in the system.
       This assumes XRT and XDNA driver is installed
       and the system has NPU hardware.

    Returns:
        The current system device.
    """
    try:
        # Run `xrt-smi examine` and capture output
        xrt_smi = shutil.which("xrt-smi")
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
