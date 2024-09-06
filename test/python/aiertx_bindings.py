# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

# RUN: %python %s | FileCheck %s

from aie.aiertx import AIERTXControl
from util import construct_and_print_module
from aie.dialects.aiex import DDR_AIE_ADDR_OFFSET


# CHECK-LABEL: simple
@construct_and_print_module
def simple(module):
    ctl = AIERTXControl(1, 4)
    ctl.start_transaction()
    ctl.dma_update_bd_addr(0, 0, DDR_AIE_ADDR_OFFSET, 0)
    ctl.export_serialized_transaction()
