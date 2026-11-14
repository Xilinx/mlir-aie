# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Write a one-operation npu1_1col transaction binary built by hand.

The operation is a poll of the register at 0x21c000 for (reg & 6) == 2, laid out
as aie-rt serializes it: the v0.1 header with a 32-byte (aie-rt) or 28-byte
(TxnEncoding.h) operation, or the v1.0 header with a 16-byte operation. Any
opcode may be given, to build a transaction the parser does not accept.
"""

import argparse
import struct

ADDRESS, VALUE, MASK = 0x21C000, 2, 6

parser = argparse.ArgumentParser()
parser.add_argument("output")
parser.add_argument("--version", choices=["0.1", "1.0"], required=True)
parser.add_argument("--opcode", type=int, required=True)
parser.add_argument("--op-size", type=int, default=32, help="v0.1 operation size")
args = parser.parse_args()

if args.version == "0.1":
    # The opcode in an 8-byte operation header, then the 64-bit address, value,
    # mask and the size of the operation, padded to that size.
    op = struct.pack("<B7xQIII", args.opcode, ADDRESS, VALUE, MASK, args.op_size)
    op = op.ljust(args.op_size, b"\0")
else:
    op = struct.pack("<IIII", args.opcode, ADDRESS, VALUE, MASK)

major, minor = map(int, args.version.split("."))
# major, minor, device generation, rows, columns, memtile rows, padding, then
# the number of operations and the size of the whole transaction.
header = struct.pack("<8BII", major, minor, 3, 6, 1, 1, 0, 0, 1, 16 + len(op))
with open(args.output, "wb") as f:
    f.write(header + op)
