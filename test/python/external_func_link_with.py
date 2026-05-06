# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Verify that the link_with keyword argument on external_func produces the
# expected func.func attribute in the emitted MLIR.

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.dialects.aie import AIEDevice, Device, external_func, tile, end
from aie.ir import Block, InsertionPoint

from util import construct_and_print_module


# Single external_func with link_with produces a func.func with the attribute.
# CHECK-LABEL: TEST: single_func_link_with
# CHECK: func.func private @scale({{.*}}) attributes {link_with = "scale.o"}
@construct_and_print_module
def single_func_link_with():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "scale",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
            link_with="scale.o",
        )
        tile(0, 2)
        end()


# Two external_func declarations sharing the same object file each carry
# their own link_with attribute.
# CHECK-LABEL: TEST: two_funcs_same_object_file
# CHECK-DAG: func.func private @add_one({{.*}}) attributes {link_with = "kernel.o"}
# CHECK-DAG: func.func private @scale_by_two({{.*}}) attributes {link_with = "kernel.o"}
@construct_and_print_module
def two_funcs_same_object_file():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "add_one",
            inputs=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
            ],
            link_with="kernel.o",
        )
        external_func(
            "scale_by_two",
            inputs=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
            ],
            link_with="kernel.o",
        )
        tile(0, 2)
        end()


# Two external_func declarations pointing to different object files.
# CHECK-LABEL: TEST: two_funcs_different_object_files
# CHECK-DAG: func.func private @add_one({{.*}}) attributes {link_with = "add_one.o"}
# CHECK-DAG: func.func private @scale_by_two({{.*}}) attributes {link_with = "scale_by_two.o"}
@construct_and_print_module
def two_funcs_different_object_files():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "add_one",
            inputs=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
            ],
            link_with="add_one.o",
        )
        external_func(
            "scale_by_two",
            inputs=[
                np.ndarray[(16,), np.dtype[np.int32]],
                np.ndarray[(16,), np.dtype[np.int32]],
            ],
            link_with="scale_by_two.o",
        )
        tile(0, 2)
        end()


# external_func without link_with produces no link_with attribute.
# CHECK-LABEL: TEST: func_without_link_with
# CHECK: func.func private @helper({{.*}})
# CHECK-NOT: link_with
@construct_and_print_module
def func_without_link_with():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "helper",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
        )
        tile(0, 2)
        end()
