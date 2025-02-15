# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

# RUN: %python %s | FileCheck %s

from aie.aiert import AIERTControl
from util import construct_and_print_module
from aie.dialects.aiex import DDR_AIE_ADDR_OFFSET
from aie.dialects.aie import AIEDevice, get_target_model


# CHECK-LABEL: simple
@construct_and_print_module
def simple(module):
    tm = get_target_model(AIEDevice.npu1_4col)
    ctl = AIERTControl(tm)
    ctl.start_transaction()
    ctl.dma_update_bd_addr(0, 0, DDR_AIE_ADDR_OFFSET, 0)
    ctl.export_serialized_transaction()
