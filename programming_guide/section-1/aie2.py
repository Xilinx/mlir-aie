# section-1/aie2.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

from aie.dialects.aie import *  # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx  # mlir-aie context


# AI Engine structural design function
def mlir_aie_design():

    # Device declaration - aie2 device NPU
    @device(AIEDevice.npu1)
    def device_body():

        # Tile(s) declarations
        ComputeTile1 = tile(1, 3)
        ComputeTile2 = tile(2, 3)
        ComputeTile3 = tile(2, 4)


# Declares that subsequent code is in mlir-aie context
with mlir_mod_ctx() as ctx:
    mlir_aie_design()  # Call design function within the mlir-aie context
    print(ctx.module)  # Print the python-to-mlir conversion to stdout
