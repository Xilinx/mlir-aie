
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.

from aie.dialects.aie import *                     # primary mlir-aie dialect definitions
from aie.extras.context import mlir_mod_ctx        # mlir ctx wrapper 

# AI Engine structural design function
def mlir_aie_design():
    # ctx wrapper - to convert python to mlir
    with mlir_mod_ctx() as ctx:

        # Device declaration - aie2 device xcvc1902
        @device(AIEDevice.xcvc1902)
        def device_body():

            # Tile(s) declarations
            ComputeTile1 = tile(1, 3)
            ComputeTile2 = tile(2, 3)
            ComputeTile3 = tile(2, 4)

    # Print the mlir conversion
    print(ctx.module)

# Call design function to generate mlir code to stdout
mlir_aie_design()

