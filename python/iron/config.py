# config.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2025 Advanced Micro Devices, Inc.

config = {}


def set_current_device(device):
    global config
    previous_device = config.get("device")
    config["device"] = device
    return previous_device


def get_current_device():
    global config
    return config["device"]
