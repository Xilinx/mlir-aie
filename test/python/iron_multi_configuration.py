# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aiex import (
    npu_load_pdi,  # pyright: ignore[reportAttributeAccessIssue]
)
from aie.iron import DeviceConfiguration, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.utils.compile.jit.context import compile_context

Out4 = np.ndarray[(4,), np.dtype[np.int32]]
Out8 = np.ndarray[(8,), np.dtype[np.int32]]


def child_sequence(_out):
    pass


def child_sequence_with_scalar(_out, _value):
    pass


seq_a = Runtime(child_sequence, [Out4], name="seq_a")
seq_a_with_scalar = Runtime(
    child_sequence_with_scalar, [Out4, np.int32], name="seq_a_with_scalar"
)
seq_b = Runtime(child_sequence, [Out4], name="seq_b")
config_a = DeviceConfiguration("dev_a", NPU2Col1(), runtimes=[seq_a, seq_a_with_scalar])
config_b = DeviceConfiguration("dev_b", NPU2Col1(), runtimes=[seq_b])


# CHECK: module attributes {iron.configuration_count = 3 : i32, iron.entry = "main:sequence"}
# CHECK: aie.device(npu2_1col) {
# CHECK: aie.runtime_sequence(%[[OUT:.*]]: memref<8xi32>, %[[VALUE:.*]]: i32)
def coordinator(out, value):
    # CHECK: aiex.configure @dev_a
    # CHECK: %[[A:.*]] = memref.reinterpret_cast
    # CHECK: aiex.run @seq_a(%[[A]]) : (memref<4xi32>)
    with config_a.configure():
        seq_a.call(out.window(0, (4,)))
        # CHECK: %[[A_SCALAR:.*]] = memref.reinterpret_cast
        # CHECK: aiex.run @seq_a_with_scalar(%[[A_SCALAR]], %[[VALUE]]) : (memref<4xi32>, i32)
        seq_a_with_scalar.call(out.window(0, (4,)), value)
    # CHECK: aiex.configure @dev_b
    # CHECK: %[[B:.*]] = memref.reinterpret_cast
    # CHECK: aiex.run @seq_b(%[[B]]) : (memref<4xi32>)
    with config_b.configure():
        seq_b.call(out.window(4, (4,)))


entry = Runtime(coordinator, [Out8, np.int32])
main = DeviceConfiguration("main", NPU2Col1(), runtimes=[entry])
# CHECK: aie.device(npu2_1col) @dev_a
# CHECK: aie.runtime_sequence @seq_a(%{{.*}}: memref<4xi32>)
# CHECK: aie.runtime_sequence @seq_a_with_scalar(%{{.*}}: memref<4xi32>, %{{.*}}: i32)
# CHECK: aie.device(npu2_1col) @dev_b
# CHECK: aie.runtime_sequence @seq_b(%{{.*}}: memref<4xi32>)
print(Program.compose([main, config_a, config_b], entry=entry).resolve_program())

for mode in ("load-pdi", "expand-load-pdis", "control-packets"):
    mode_runtime = Runtime(lambda: None, [], name=f"mode_{mode}")
    mode_module = Program(
        NPU2Col1(), mode_runtime, reconfiguration_mode=mode
    ).resolve_program()
    mode_attr = mode_module.operation.attributes.get("iron.reconfiguration_mode")
    if mode == "load-pdi":
        assert mode_attr is None
    else:
        assert mode_attr.value == mode

try:
    Program(NPU2Col1(), Runtime(lambda: None, []), reconfiguration_mode="invalid")
except ValueError as error:
    assert "Unsupported reconfiguration mode 'invalid'" in str(error)
else:
    raise AssertionError("Program accepted an unsupported reconfiguration mode")

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
    # CHECK: owner: Runtime 'seq_a' already belongs to configuration 'dev_a'.
    print(f"owner: {error}")
else:
    raise AssertionError("Runtime accepted two configurations")

with mlir_mod_ctx():
    try:
        with config_b.configure():
            seq_a.call()
    except ValueError as error:
        # CHECK: scope: Runtime 'seq_a' belongs to configuration 'dev_a', not 'dev_b'.
        print(f"scope: {error}")
    else:
        raise AssertionError("Runtime accepted the wrong configuration scope")

    try:
        with config_a.configure():
            with config_b.configure():
                pass
    except RuntimeError as error:
        # CHECK: nested: Nested configuration scopes are not supported.
        print(f"nested: {error}")
    else:
        raise AssertionError("Configuration scopes nested")


def empty_sequence():
    pass


with compile_context(_iron_full_elf=True):
    implicit_runtime = Runtime(empty_sequence, [], name="implicit")
    implicit_module = str(
        Program(NPU2Col1(), implicit_runtime).resolve_program().operation
    )
    assert implicit_module.count("aiex.npu.load_pdi") == 1
    assert "device_ref = @main" in implicit_module

    target_runtime = Runtime(empty_sequence, [], name="target")
    target = DeviceConfiguration("target", NPU2Col1(), runtimes=[target_runtime])

    def explicit_sequence():
        with target.configure():
            target_runtime.call()

    explicit_runtime = Runtime(explicit_sequence, [], name="explicit")
    source = DeviceConfiguration("source", NPU2Col1(), runtimes=[explicit_runtime])
    explicit_module = str(
        Program.compose([source, target], entry=explicit_runtime)
        .resolve_program()
        .operation
    )
    assert explicit_module.count("aiex.configure") == 1
    assert "aiex.configure @target" in explicit_module
    assert "aiex.configure @source" not in explicit_module
    assert "aiex.npu.load_pdi" not in explicit_module

    def load_sequence():
        npu_load_pdi(device_ref="target")

    load_runtime = Runtime(load_sequence, [], name="load")
    load_module = str(Program(NPU2Col1(), load_runtime).resolve_program().operation)
    assert load_module.count("aiex.npu.load_pdi") == 1
    assert "device_ref = @target" in load_module

    disabled_runtime = Runtime(
        empty_sequence, [], name="disabled", implicit_configure=False
    )
    disabled_module = str(
        Program(NPU2Col1(), disabled_runtime).resolve_program().operation
    )
    assert "aiex.npu.load_pdi" not in disabled_module
