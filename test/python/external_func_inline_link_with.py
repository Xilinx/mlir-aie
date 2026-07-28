# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Verify that ExternalFunction(inline=True) declares its kernel with a .ll
# link_with plus link_with_mode = "merge" -- the explicit metadata that tells
# aiecc to llvm-link the artifact into the core instead of object-linking it --
# while the default object-linked path keeps a .o and no mode.  IR-only; no
# hardware.

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


# inline=True: the artifact defaults to a .ll and the func.func declaration
# carries it as link_with, tagged link_with_mode = "merge" so aiecc merges it.
# CHECK-LABEL: TEST: inline_true_declares_ll_link_with
# CHECK: func.func private @add_one_inl({{.*}}) attributes {link_with = "add_one_inl.ll", link_with_mode = "merge"}
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
        assert ef.link_with_mode == "merge", ef.link_with_mode
        ef.resolve()
        tile(0, 2)
        end()


# Default (object-linked) path is unchanged: a .o link_with and no
# link_with_mode at all, so existing IR stays byte-identical.
# CHECK-LABEL: TEST: default_declares_o_link_with
# CHECK: func.func private @add_one_obj({{.*}}) attributes {link_with = "add_one_obj.o"}
# CHECK-NOT: link_with_mode
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
        assert ef.link_with_mode is None, ef.link_with_mode
        ef.resolve()
        tile(0, 2)
        end()


# An explicit IR filename is honored verbatim -- no renaming -- and still gets
# the merge mode.  A .bc selects bitcode instead of textual IR.
# CHECK-LABEL: TEST: inline_explicit_ir_name_preserved
# CHECK: func.func private @add_one_named({{.*}}) attributes {link_with = "custom_name.bc", link_with_mode = "merge"}
@construct_and_print_module
def inline_explicit_ir_name_preserved():
    ExternalFunction._instances.clear()
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        ef = ExternalFunction(
            "add_one_named",
            object_file_name="custom_name.bc",
            source_string=_SRC.format(name="add_one_named"),
            arg_types=_ARGS,
            inline=True,
        )
        assert ef.object_file_name == "custom_name.bc", ef.object_file_name
        ef.resolve()
        tile(0, 2)
        end()


# An explicit non-IR filename used to be silently rewritten to .ll so aiecc
# would route it by suffix.  Routing is now explicit metadata, so a name that
# cannot name LLVM IR is simply an error.
# CHECK-LABEL: TEST: inline_explicit_o_is_rejected
# CHECK: ValueError: ExternalFunction 'add_one_rejected': inline=True emits LLVM IR, so object_file_name must end in '.ll' (textual LLVM IR) or '.bc' (bitcode); got 'custom_name.o'.
@construct_and_print_module
def inline_explicit_o_is_rejected():
    ExternalFunction._instances.clear()
    dev = Device(AIEDevice.npu1_1col)
    dev_block = Block.create_at_start(dev.body_region)
    with InsertionPoint(dev_block):
        try:
            ExternalFunction(
                "add_one_rejected",
                object_file_name="custom_name.o",
                source_string=_SRC.format(name="add_one_rejected"),
                arg_types=_ARGS,
                inline=True,
            )
            raise AssertionError("expected ValueError")
        except ValueError as e:
            print("ValueError:", e)
        tile(0, 2)
        end()
