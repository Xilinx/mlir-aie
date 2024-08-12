#!/usr/bin/env python3
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.

import aie
from aie.ir import *
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.dialects.ext import memref

import sys
import struct


def print_none(*args):
    pass


def print_log_(*args):
    print(*args)


print_log = print_none


def parse_txn(data, verbose=False):
    print_log = print_log_ if verbose else print_none

    header_format = "BBBBBBII"
    major, minor, dev_gen, num_rows, num_cols, num_mem_tile_rows, num_ops, txn_size = (
        struct.unpack(header_format, data[:16])
    )
    print(f"// Major: {major}")
    print(f"// Minor: {minor}")
    print(f"// DevGen: {dev_gen}")
    print(f"// NumRows: {num_rows}")
    print(f"// NumCols: {num_cols}")
    print(f"// NumMemTileRows: {num_mem_tile_rows}")
    print(f"// NumOps: {num_ops}")
    print(f"// TxnSize: {txn_size} bytes")
    operations = []
    i = 16
    # v0.1
    if major == 0 and minor == 1:
        while i < len(data):
            opc, _, _, _ = struct.unpack("BBBB", data[i : i + 4])
            print_log(f"opcode: {opc:#x}")
            if opc == 0x00:
                print_log("opcode: WRITE (0x00)")
                addr0, addr1, value, size = struct.unpack("IIII", data[i + 8 : i + 24])
                addr = addr1 << 32 | addr0
                print_log(f"addr: {addr:#x}")
                print_log(f"value: {value:#x}")
                print_log(f"size: {size}")
                operations.append((opc, addr, value))
                i = i + size
            elif opc == 0x01:
                print_log("opcode: BLOCKWRITE (0x01)")
                _, addr, size = struct.unpack("III", data[i + 4 : i + 16])
                print_log(f"addr: {addr:#x}")
                print_log(f"size: {size}")
                operations.append((opc, addr, data[i + 16 : i + size - 16]))
                i = i + size
            elif opc == 0x03:
                print_log("opcode: MASKWRITE (0x03)")
                addr0, addr1, value, mask, size = struct.unpack(
                    "IIIII", data[i + 8 : i + 28]
                )
                addr = addr1 << 32 | addr0
                print_log(f"addr: {addr:#x}")
                print_log(f"value: {value:#x}")
                print_log(f"mask: {mask:#x}")
                print_log(f"size: {size}")
                operations.append((opc, addr, value, mask))
                i = i + size
            else:
                value = struct.unpack("I", data[i : i + 4])[0]
                raise Exception(f"Unhandled header: {value:#x}")
    # v1.0
    if major == 1 and minor == 0:
        while i < len(data):
            opc, _, _, _ = struct.unpack("BBBB", data[i : i + 4])
            print_log(f"opcode: {opc:#x}")
            if opc == 0x00:
                print_log("opcode: WRITE (0x00)")
                addr, value = struct.unpack("II", data[i + 4 : i + 12])
                print_log(f"addr: {addr:#x}")
                print_log(f"value: {value:#x}")
                operations.append((opc, addr, value))
                i = i + 12
            elif opc == 0x01:
                print_log("opcode: BLOCKWRITE (0x01)")
                addr, size = struct.unpack("II", data[i + 4 : i + 12])
                print_log(f"addr: {addr:#x}")
                print_log(f"size: {size}")
                operations.append((opc, addr, data[i + 12 : i + size]))
                i = i + size
            elif opc == 0x03:
                print_log("opcode: MASKWRITE (0x03)")
                addr, value, mask = struct.unpack("III", data[i + 4 : i + 16])
                print_log(f"addr: {addr:#x}")
                print_log(f"value: {value:#x}")
                print_log(f"mask: {mask:#x}")
                operations.append((opc, addr, value, mask))
                i = i + 16
            else:
                value = struct.unpack("I", data[i : i + 4])[0]
                raise Exception(f"Unhandled header: {value:#x}")
    return num_cols, operations


def operations_to_mlir(operations, columns=5):
    with Context(), Location.unknown():
        module = Module.create()
        global_data = []
        with InsertionPoint(module.body):

            devs = {
                1: AIEDevice.npu1_1col,
                2: AIEDevice.npu1_2col,
                3: AIEDevice.npu1_3col,
                4: AIEDevice.npu1_4col,
                5: AIEDevice.npu1,
            }
            @device(devs[columns])
            def device_body():
                for op in operations:
                    if op[0] == 0x01:
                        d = np.frombuffer(op[2], dtype=np.int32)
                        blockwrite_data = memref.global_(initial_value=d)
                        global_data.append(blockwrite_data)
                    else:
                        global_data.append(None)

                @runtime_sequence()
                def sequence():
                    for op, payload in zip(operations, global_data):
                        if op[0] == 0x00:
                            addr = op[1]
                            value = op[2]
                            npu_write32(addr, value)
                        elif op[0] == 0x01:
                            addr = op[1]
                            d = memref.get_global(
                                payload.type_.value, payload.sym_name.value
                            )
                            npu_blockwrite(addr, d)
                        elif op[0] == 0x03:
                            addr = op[1]
                            value = op[2]
                            mask = op[3]
                            npu_maskwrite32(addr, value, mask)
                        else:
                            raise Exception(f"Unhandled op: {op:#x}")

    return module


if __name__ == "__main__":
    # Check if command line arguments are provided
    if len(sys.argv) == 1:
        # Read data from standard input
        data = sys.stdin.buffer.read()
        # Parse the TXN data
        columns, operations = parse_txn(data)
    else:
        # Process each file provided as command line argument
        operations = []
        for filename in sys.argv[1:]:
            # Open the file in binary mode
            with open(filename, "rb") as f:
                # Read the data from the file
                data = f.read()
                # Parse the TXN data
                columns, ops = parse_txn(data)
                operations = operations + ops

    module = operations_to_mlir(operations, columns)

    print(str(module))
