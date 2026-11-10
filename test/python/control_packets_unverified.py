# control_packets_unverified.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# generate_control_packets translates ops that were never verified, so the
# translator must reject packets the verifier would reject instead of
# encoding them.

# RUN: %python %s 2>&1 | FileCheck %s

from aie.dialects.aie import *
import aie.dialects.aiex as aiex
from aie.extras.context import mlir_mod_ctx

from aie.dialects.aie import generate_control_packets


def gen(emit):
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1)
        def device_body():
            tile(0, 0, packet_type=2, packet_id=3)

            @aiex.runtime_sequence()
            def sequence():
                emit()

        words = generate_control_packets(ctx.module.operation)
        print(f"words: {len(words)}")


# CHECK: control packet with a write opcode requires a data payload
# CHECK: words: 0
gen(lambda: aiex.control_packet(address=0x0001F000, opcode=0, stream_id=0, length=2))

# CHECK: control packet payload must be at least 1 word
# CHECK: words: 0
gen(lambda: aiex.control_packet(address=0x0001F000, opcode=0, stream_id=0, data=[]))

# CHECK: words: 3
gen(lambda: aiex.control_packet(address=0x0001F000, opcode=0, stream_id=0, data=[2]))
