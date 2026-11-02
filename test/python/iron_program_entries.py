# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

import numpy as np
from aie.iron import DeviceConfiguration, Program, Runtime
from aie.iron.device import NPU2Col1
from aie.utils.compile.jit.context import compile_context

Out4 = np.ndarray[(4,), np.dtype[np.int32]]
Out8 = np.ndarray[(8,), np.dtype[np.int32]]


def sequence(_out):
    pass


# Several entries: the first is the default, and every one is listed.
# CHECK: module attributes {iron.entries = ["main:one", "main:two"], iron.entry = "main:one"}
# CHECK: aie.runtime_sequence @one(%{{.*}}: memref<4xi32>)
# CHECK: aie.runtime_sequence @two(%{{.*}}: memref<4xi32>)
one = Runtime(sequence, [Out4], name="one")
two = Runtime(sequence, [Out4], name="two")
print(Program(NPU2Col1(), [one, two]).resolve_program())

# One entry keeps the module as it was: no entry attributes.
# CHECK: module {
single = Runtime(sequence, [Out4], name="single")
print(Program(NPU2Col1(), single).resolve_program())

# On a full ELF each entry loads its PDI first, not only the default.
with compile_context(_iron_full_elf=True):
    first = Runtime(sequence, [Out4], name="first")
    second = Runtime(sequence, [Out4], name="second")
    module = str(Program(NPU2Col1(), [first, second]).resolve_program().operation)
    for name in ("first", "second"):
        body = module.split(f"aie.runtime_sequence @{name}(", 1)[1]
        assert body.split("}", 1)[0].count("aiex.npu.load_pdi") == 1, body

    # A composed Program's entries load their own configuration's PDI.
    left = Runtime(sequence, [Out4], name="left")
    right = Runtime(sequence, [Out4], name="right")
    config = DeviceConfiguration("dev", NPU2Col1(), runtimes=[left, right])
    composed = str(
        Program.compose([config], entry=[left, right]).resolve_program().operation
    )
    assert 'iron.entries = ["dev:left", "dev:right"]' in composed
    assert composed.count("aiex.npu.load_pdi") == 2

# Entries take the same arguments: the host buffers one build is called with.
# CHECK: mismatch: Entry runtime 'wide' takes
wide = Runtime(sequence, [Out8], name="wide")
narrow = Runtime(sequence, [Out4], name="narrow")
try:
    Program(NPU2Col1(), [narrow, wide]).resolve_program()
except ValueError as error:
    print(f"mismatch: {error}")
else:
    raise AssertionError("entries with different argument types resolved")

# CHECK: twice: Program lists an entry runtime more than once.
repeated = Runtime(sequence, [Out4], name="repeated")
holder = DeviceConfiguration("holder", NPU2Col1(), runtimes=[repeated])
try:
    Program.compose([holder], entry=[repeated, repeated]).resolve_program()
except ValueError as error:
    print(f"twice: {error}")
else:
    raise AssertionError("an entry listed twice resolved")
