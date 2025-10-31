# Copyright (C) 2023, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

from aie.iron import Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.device import NPU1

# CHECK: {stack_size = 2048 : i32}

my_worker = Worker(None, stack_size=2048, while_true=False)

rt = Runtime()
with rt.sequence():
    rt.start(my_worker)

my_program = Program(NPU1(), rt)

module = my_program.resolve_program(SequentialPlacer())

print(module)


# CHECK: {stack_size = 512 : i32}

from aie.dialects.aie import *
from aie.extras.context import mlir_mod_ctx


def mlir_aie_design():

    @device(AIEDevice.npu1)
    def device_body():

        ComputeTile1 = tile(0, 2)

        @core(ComputeTile1, stack_size=512)
        def core_body():
            pass


with mlir_mod_ctx() as ctx:
    mlir_aie_design()
    print(ctx.module)
