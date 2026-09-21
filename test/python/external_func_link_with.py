# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Verify that the link_with, link_with_mode and stack_size_override keyword
# arguments on external_func produce the expected func.func attributes in the
# emitted MLIR, and that external_func enforces their contracts:
# link_with_mode needs link_with and accepts "merge" alone, and
# stack_size_override must be >= 0.

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


# link_with_mode="merge" is emitted as its own string attribute.  It is the
# only signal that routes the artifact to aiecc's llvm-link merge path -- the
# suffix (here a plain .o) does not.
# CHECK-LABEL: TEST: link_with_mode_merge
# CHECK: func.func private @merged({{.*}}) attributes {link_with = "merged.o", link_with_mode = "merge"}
@construct_and_print_module
def link_with_mode_merge():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "merged",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
            link_with="merged.o",
            link_with_mode="merge",
        )
        tile(0, 2)
        end()


# A mode with nothing to apply it to is a caller error, not a silent no-op.
# CHECK-LABEL: TEST: link_with_mode_requires_link_with
# CHECK: ValueError: external_func 'orphan': link_with_mode requires link_with to be set.
# CHECK-NOT: func.func private @orphan
@construct_and_print_module
def link_with_mode_requires_link_with():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        try:
            external_func(
                "orphan",
                inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
                link_with_mode="merge",
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print("ValueError:", e)
        tile(0, 2)
        end()


# "merge" is the only defined mode; anything else is rejected by name so a
# typo can't silently degrade to object linking.
# CHECK-LABEL: TEST: link_with_mode_rejects_unknown_value
# CHECK: ValueError: external_func 'bogus_mode': invalid link_with_mode 'inline'; the only supported value is 'merge'.
# CHECK-NOT: func.func private @bogus_mode
@construct_and_print_module
def link_with_mode_rejects_unknown_value():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        try:
            external_func(
                "bogus_mode",
                inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
                link_with="bogus.o",
                link_with_mode="inline",
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print("ValueError:", e)
        tile(0, 2)
        end()


# stack_size_override becomes its own integer attribute, independent of
# link_with. aiecc's stack analysis reads it by name from the func.func,
# whatever the link mode of the artifact.
# CHECK-LABEL: TEST: stack_size_override_emitted
# CHECK: func.func private @recursive({{.*}}) attributes {link_with = "recursive.o", stack_size_override = 4096 : i32}
@construct_and_print_module
def stack_size_override_emitted():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "recursive",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
            link_with="recursive.o",
            stack_size_override=4096,
        )
        tile(0, 2)
        end()


# 0 is a legal override: this kernel adds no stack. It differs from an absent
# attribute.
# CHECK-LABEL: TEST: stack_size_override_zero_is_legal
# CHECK: func.func private @leaf({{.*}}) attributes {link_with = "leaf.o", stack_size_override = 0 : i32}
@construct_and_print_module
def stack_size_override_zero_is_legal():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "leaf",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
            link_with="leaf.o",
            stack_size_override=0,
        )
        tile(0, 2)
        end()


# external_func without stack_size_override produces no such attribute.
# CHECK-LABEL: TEST: func_without_stack_size_override
# CHECK: func.func private @plain({{.*}})
# CHECK-NOT: stack_size_override
@construct_and_print_module
def func_without_stack_size_override():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        external_func(
            "plain",
            inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
            link_with="plain.o",
        )
        tile(0, 2)
        end()


# A negative override raises a ValueError.
# CHECK-LABEL: TEST: stack_size_override_rejects_negative
# CHECK: ValueError: external_func 'bad': stack_size_override must be >= 0, got -1.
# CHECK-NOT: func.func private @bad
@construct_and_print_module
def stack_size_override_rejects_negative():
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        try:
            external_func(
                "bad",
                inputs=[np.ndarray[(16,), np.dtype[np.int32]]],
                link_with="bad.o",
                stack_size_override=-1,
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print("ValueError:", e)
        tile(0, 2)
        end()
