# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Each encoding of a mask poll that aie-codegen writes (the v0.1 header with the
# 32-byte operation aie-codegen serializes or the 28-byte one TxnEncoding.h emits,
# and the v1.0 header with its 16-byte operation) parses back to a maskpoll of
# the same register, value and mask. A busy poll has no counterpart in the IR
# and is rejected rather than read as a plain poll. A v0.1 operation whose size
# is below its header is rejected, not misparsed.
#
# The transactions are built by hand below and fed to txn2mlir.py on stdin; the
# tool path comes from lit's txn2mlir.py substitution.

# RUN: %python %s txn2mlir.py v01_aiert | FileCheck %s
# RUN: %python %s txn2mlir.py v01_28 | FileCheck %s
# RUN: %python %s txn2mlir.py v10 | FileCheck %s
# RUN: %python %s txn2mlir.py v01_busy | FileCheck %s --check-prefix=BUSY
# RUN: %python %s txn2mlir.py v10_busy | FileCheck %s --check-prefix=BUSY
# RUN: %python %s txn2mlir.py v01_short | FileCheck %s --check-prefix=SHORT

# CHECK: aie.device(npu1_1col)
# CHECK: %[[ADDR:.*]] = arith.constant 2211840 : i32
# CHECK: %[[VAL:.*]] = arith.constant 2 : i32
# CHECK: %[[MASK:.*]] = arith.constant 6 : i32
# CHECK: aiex.npu.maskpoll(%[[ADDR]], %[[VAL]], %[[MASK]]) : i32, i32, i32
# CHECK-NOT: aiex.npu

# BUSY: Unhandled opcode: 7
# BUSY-NEXT: Failed to parse binary

# SHORT: Transaction op size 20 is smaller than its 28-byte header
# SHORT-NEXT: Failed to parse binary

import struct
import subprocess
import sys

MASKPOLL, BUSYPOLL = 4, 7
ADDRESS, VALUE, MASK = 0x21C000, 2, 6


def transaction(version, opcode, op_size=32):
    """Build a one-operation npu1_1col transaction that polls ADDRESS.

    The poll waits for (reg & MASK) == VALUE. The v0.1 header carries a
    32-byte (aie-codegen) or 28-byte (TxnEncoding.h) operation, the v1.0 header a
    16-byte one. Any opcode may be given, to build a transaction the parser
    does not accept.
    """
    if version == (0, 1):
        # The opcode in an 8-byte operation header, then the 64-bit address,
        # value, mask and the size of the operation, padded to that size.
        op = struct.pack("<B7xQIII", opcode, ADDRESS, VALUE, MASK, op_size)
        op = op.ljust(op_size, b"\0")
    else:
        op = struct.pack("<IIII", opcode, ADDRESS, VALUE, MASK)
    # major, minor, device generation, rows, columns, memtile rows, padding,
    # then the number of operations and the size of the whole transaction.
    header = struct.pack("<8BII", *version, 3, 6, 1, 1, 0, 0, 1, 16 + len(op))
    return header + op


CASES = {
    "v01_aiert": transaction((0, 1), MASKPOLL, 32),
    "v01_28": transaction((0, 1), MASKPOLL, 28),
    "v10": transaction((1, 0), MASKPOLL),
    "v01_busy": transaction((0, 1), BUSYPOLL, 32),
    "v10_busy": transaction((1, 0), BUSYPOLL),
    "v01_short": transaction((0, 1), MASKPOLL, 20),
}

if __name__ == "__main__":
    tool, case = sys.argv[1:]
    result = subprocess.run(
        [sys.executable, tool, "-f", "-"],
        input=CASES[case],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    sys.stdout.buffer.write(result.stdout)
    sys.exit(result.returncode)
