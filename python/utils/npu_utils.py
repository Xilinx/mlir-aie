# npu_utils.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

import json
import os


def _load_npu_models():
    json_path = os.path.join(os.path.dirname(__file__), "npu_models.json")
    with open(json_path, "r") as f:
        return json.load(f)


# NPU Model mappings
# Maps generation name to list of model strings that may appear in xrt-smi or device info
NPU_MODELS = _load_npu_models()


def get_npu_generation(device_name: str) -> str | None:
    """
    Determine NPU generation from device name string.
    Returns 'npu1', 'npu2', or None if unknown.

    Args:
        device_name: String containing device name or model info.
    """
    # Check for NPU2 keywords first as they are more specific
    for keyword in NPU_MODELS["npu2"]:
        if keyword in device_name:
            return "npu2"

    for keyword in NPU_MODELS["npu1"]:
        if keyword in device_name:
            return "npu1"

    return None
