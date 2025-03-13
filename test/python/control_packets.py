# test.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates

# RUN: python %s | FileCheck %s

from aie.dialects.aie import *
import aie.dialects.aiex as aiex
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import generate_control_packets


def gen_cp_sequence():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            params = []

            @aiex.runtime_sequence(*params)
            def sequence(*args):
                # CHECK: 0001F000
                # CHECK: 00000002
                aiex.control_packet(address=0x0001F000, opcode=0, stream_id=0, data=[2])
                # CHECK: 09B1F020
                # CHECK: 00000003
                # CHECK: 00000004
                # CHECK: 00000005
                # CHECK: 00000006
                aiex.control_packet(
                    address=0x0001F020, opcode=2, stream_id=9, data=[3, 4, 5, 6]
                )
                # CHECK: 02700400
                aiex.control_packet(address=0x00000400, opcode=1, stream_id=2, length=4)

        return ctx.module


aie_module = gen_cp_sequence()
print(aie_module)
for i in generate_control_packets(aie_module.operation):
    print(f"{i:08X}")
