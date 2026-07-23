# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Verify that ExternalFunction(inline=True) declares its kernel with a .ll
# link_with -- the LLVM-IR artifact aiecc merges into the core and inlines --
# while the default object-linked path keeps a .o.  IR-only; no hardware.

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.dialects.aie import AIEDevice, Device, tile, end
from aie.ir import Block, InsertionPoint
from aie.iron import ExternalFunction

from util import construct_and_print_module

_SRC = 'extern "C" {{ void {name}(int *a, int *b) {{ for (int i = 0; i < 16; i++) b[i] = a[i] + 1; }} }}'
_ARGS = [
    np.ndarray[(16,), np.dtype[np.int32]],
    np.ndarray[(16,), np.dtype[np.int32]],
]


# inline=True: the object file name is a .ll and the func.func declaration
# carries that .ll as its link_with (aiecc routes .ll/.bc to the IR-merge path).
# CHECK-LABEL: TEST: inline_true_declares_ll_link_with
# CHECK: func.func private @add_one_inl({{.*}}) attributes {link_with = "add_one_inl.ll"}
@construct_and_print_module
def inline_true_declares_ll_link_with():
    ExternalFunction._instances.clear()
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        ef = ExternalFunction(
            "add_one_inl",
            source_string=_SRC.format(name="add_one_inl"),
            arg_types=_ARGS,
            inline=True,
        )
        assert ef.object_file_name == "add_one_inl.ll", ef.object_file_name
        ef.resolve()
        tile(0, 2)
        end()


# Default (object-linked) path is unchanged: a .o link_with.
# CHECK-LABEL: TEST: default_declares_o_link_with
# CHECK: func.func private @add_one_obj({{.*}}) attributes {link_with = "add_one_obj.o"}
@construct_and_print_module
def default_declares_o_link_with():
    ExternalFunction._instances.clear()
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        ef = ExternalFunction(
            "add_one_obj",
            source_string=_SRC.format(name="add_one_obj"),
            arg_types=_ARGS,
        )
        assert ef.object_file_name == "add_one_obj.o", ef.object_file_name
        ef.resolve()
        tile(0, 2)
        end()
