# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import json
import sys


def file_loc(file, line, col):
    return {"kind": "file", "file": file, "line": line, "col": col}


def name_loc(name, line):
    return {
        "kind": "name",
        "name": name,
        "child": file_loc("user.py", line, 4),
    }


with open(sys.argv[1]) as sidecar:
    locmap = json.load(sidecar)

expected = [
    ("WRITE32", "aiex.npu.write32", 0xABC00DEF, name_loc("of_in", 42)),
    ("WRITE32", "aiex.npu.write32", 0, name_loc("zero_reg", 60)),
    ("BLOCKWRITE", "aiex.npu.blockwrite", 0x12345678, name_loc("dma_task", 50)),
    (
        "WRITE32",
        "aiex.npu.write32",
        4,
        {
            "kind": "fused",
            "children": [
                name_loc("of_in", 42),
                {
                    "kind": "callsite",
                    "callee": file_loc("kernel.py", 12, 3),
                    "caller": file_loc("user.py", 70, 8),
                },
            ],
        },
    ),
    ("WRITE32", "aiex.npu.write32", 8, {"kind": "unknown"}),
]

assert locmap["version"] == 1, locmap
operations = locmap["operations"]
assert len(operations) == len(expected), operations
previous_end = None
for entry, (opcode, source_op, address, location) in zip(operations, expected):
    assert entry["opcode"] == opcode, entry
    assert entry["source_op"] == source_op, entry
    assert int(entry["address"], 16) == address, entry
    assert entry["loc"] == location, entry
    assert entry["byte_size"] > 0 and entry["byte_size"] % 4 == 0, entry
    assert entry["byte_offset"] % 4 == 0, entry
    if previous_end is not None:
        assert entry["byte_offset"] == previous_end, entry
    previous_end = entry["byte_offset"] + entry["byte_size"]
