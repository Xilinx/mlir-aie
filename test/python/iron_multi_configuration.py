# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.iron import DeviceConfiguration, Program, Runtime
from aie.iron.device import NPU2Col1

Out4 = np.ndarray[(4,), np.dtype[np.int32]]
Out8 = np.ndarray[(8,), np.dtype[np.int32]]


def child_sequence(_out):
    pass


seq_a = Runtime(child_sequence, [Out4], name="seq_a")
seq_b = Runtime(child_sequence, [Out4], name="seq_b")
config_a = DeviceConfiguration("dev_a", NPU2Col1(), runtimes=[seq_a])
config_b = DeviceConfiguration("dev_b", NPU2Col1(), runtimes=[seq_b])


def coordinator(out):
    with config_a.configure():
        seq_a.call(out.window(0, (4,)))
    with config_b.configure():
        seq_b.call(out.window(4, (4,)))


entry = Runtime(coordinator, [Out8])
main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
print(Program.compose([main, config_a, config_b], entry=entry).resolve_program())

flow_a = object()
flow_b = object()
seq_a.add_flow(flow_a)
seq_b.add_flow(flow_b)
assert seq_a.flows == [flow_a]
assert seq_b.flows == [flow_b]
assert config_a.flows == [flow_a]
assert config_b.flows == [flow_b]

try:
    DeviceConfiguration("other", NPU2Col1(), runtimes=[seq_a])
except ValueError as error:
    print(f"owner: {error}")
else:
    raise AssertionError("Runtime accepted two configurations")

with mlir_mod_ctx():
    try:
        with config_b.configure():
            seq_a.call()
    except ValueError as error:
        print(f"scope: {error}")
    else:
        raise AssertionError("Runtime accepted the wrong configuration scope")

    try:
        with config_a.configure():
            with config_b.configure():
                pass
    except RuntimeError as error:
        print(f"nested: {error}")
    else:
        raise AssertionError("Configuration scopes nested")

# CHECK: module attributes {iron.configuration_count = 3 : i32, iron.entry = "main:sequence"}
# CHECK: aie.device(npu2_1col) {
# CHECK: aie.runtime_sequence(%[[OUT:.*]]: memref<8xi32>)
# CHECK: aiex.configure @dev_a
# CHECK: %[[A:.*]] = memref.reinterpret_cast
# CHECK: aiex.run @seq_a(%[[A]]) : (memref<4xi32>)
# CHECK: aiex.configure @dev_b
# CHECK: %[[B:.*]] = memref.reinterpret_cast
# CHECK: aiex.run @seq_b(%[[B]]) : (memref<4xi32>)
# CHECK: aie.device(npu2_1col) @dev_a
# CHECK: aie.runtime_sequence @seq_a(%{{.*}}: memref<4xi32>)
# CHECK: aie.device(npu2_1col) @dev_b
# CHECK: aie.runtime_sequence @seq_b(%{{.*}}: memref<4xi32>)
# CHECK: owner: Runtime 'seq_a' already belongs to configuration 'dev_a'.
# CHECK: scope: Runtime 'seq_a' belongs to configuration 'dev_a', not 'dev_b'.
# CHECK: nested: Nested configuration scopes are not supported.
