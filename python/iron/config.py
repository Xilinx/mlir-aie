# config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import subprocess

from iron.device import NPU1Col4, NPU2


def detect_npu_device():
    try:
        # Run `xrt-smi examine` and capture output
        result = subprocess.run(
            ["/opt/xilinx/xrt/bin/xrt-smi", "examine"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        output = result.stdout

        # Match strings for NPU2
        # Set's generic "whole array" devices, this is overkill...
        if any(
            keyword.lower() in output.lower()
            for keyword in [
                "NPU Strix",
                "NPU Strix Halo",
                "NPU Kracken",
                "RyzenAI-npu4",
                "RyzenAI-npu6",
            ]
        ):
            return NPU2()
        else:
            return NPU1Col4()

    except FileNotFoundError:
        raise RuntimeError("xrt-smi not found. Make sure XRT is installed and set up.")
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
