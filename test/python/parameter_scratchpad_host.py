# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s %t.params | FileCheck %s

# The host scratchpad, without a device or runtime: it sizes and owns its
# table, encodes each kind as firmware reads it, and describes each slot
# (index, shift, kind, type) for a runtime that stages the table itself or
# fills a slot from elsewhere. Over a caller's buffer it encodes in place.

import argparse

import numpy as np
from aie.utils.hostruntime.parameter_scratchpad import ParameterScratchpad, slot_count

PARAMS = """\
4 3
count 0 i32 core 1 64
base 1 i32 addr - -
reps 2 i32 repeat 0 7
spare - i32 core - -
0
2
main:sequence 3 count base reps
main:sequence_1 1 count
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("params", help="where to write params.txt")
    args = parser.parse_args()
    with open(args.params, "w") as f:
        f.write(PARAMS)

    # CHECK: slots 3
    print("slots", slot_count(args.params))

    params = ParameterScratchpad(args.params, "main:sequence")
    # CHECK: names ['base', 'count', 'reps']
    print("names", sorted(params.names))
    # CHECK: count index=0 shift=2 kind=core type=i32
    # CHECK: base index=1 shift=0 kind=addr type=i32
    # CHECK: reps index=2 shift=16 kind=repeat type=i32
    for name in ("count", "base", "reps"):
        s = params.slot(name)
        print(f"{name} index={s.index} shift={s.shift} kind={s.kind} type={s.type}")

    # CHECK: table 0x14 0xdeadbeef 0x30000
    params.write("count", 5)
    params.write("base", 0xDEADBEEF)
    params.write("reps", np.int32(3))
    params.validate()
    print("table", " ".join(f"{w:#x}" for w in params.table))
    # CHECK: read count 0x14
    print(f"read count {params.read('count'):#x}")

    # CHECK: refused 65: {{.*}}outside [1, 64]{{.*}}
    # CHECK: refused spare: {{.*}}no command reads parameter 'spare'{{.*}}
    try:
        params.write("count", 65)
    except ValueError as e:
        print(f"refused 65: {e}")
    try:
        params.slot("spare")
    except RuntimeError as e:
        print(f"refused spare: {e}")

    # A caller's buffer is encoded in place.
    # CHECK: in place 0x1c 0x0 0x0
    words = np.zeros(4, dtype=np.uint32)
    later = ParameterScratchpad(args.params, "main:sequence_1", buffer=words)
    later.write("count", 7)
    print("in place", " ".join(f"{w:#x}" for w in words[:3]))

    # Commands that read no parameter have a scratchpad of no slots.
    # CHECK: no slots: 0 words
    empty = args.params + ".empty"
    with open(empty, "w") as f:
        f.write("1 0\nunused - i32 core - -\n0\n1\nmain:sequence 0\n")
    print("no slots:", ParameterScratchpad(empty, "main:sequence").table.size, "words")


main()
