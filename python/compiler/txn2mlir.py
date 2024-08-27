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
import argparse
import aie.extras.types as T


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
                operations.append((opc, addr, data[i + 16 : i + size]))
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


def operations_to_mlir(operations, columns=5, mlir_ctrl_pkt=False):
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

                if mlir_ctrl_pkt:
                    # Runtime sequence arg0 as handle for ctrl packet raw data in host ddr
                    MAX_CTRL_PKTS_HOST_SIZE = 2048

                    @runtime_sequence(T.memref(MAX_CTRL_PKTS_HOST_SIZE, T.i32()))
                    def sequence(arg0):
                        for op, payload in zip(operations, global_data):
                            if op[0] == 0x00:
                                addr = op[1]
                                value = op[2]
                                control_packet(
                                    address=addr,
                                    opcode=0,
                                    stream_id=0,
                                    data=np.array([value]).astype(np.int32),
                                )
                            elif op[0] == 0x01:
                                addr = op[1]
                                data = np.array(payload.initial_value, dtype=np.int32)
                                # Individual access cannot cross a 128-bit boundary.
                                num_split_4s = (data.size + 3) // 4
                                data_split_4 = data
                                if num_split_4s > 1:
                                    data_split_4 = np.array_split(
                                        data[: (num_split_4s - 1) * 4], num_split_4s
                                    )
                                    data_split_4 = data_split_4.append(
                                        data[(num_split_4s - 1) * 4 :]
                                    )
                                if num_split_4s == 2:
                                    # Individual access cannot cross a 128-bit boundary.
                                    data_split_4 = [data[:4], data[4:]]
                                for d_split in data_split_4:
                                    control_packet(
                                        address=addr,
                                        opcode=0,
                                        stream_id=0,
                                        data=d_split,
                                    )
                                    addr = addr + d_split.size * 4
                            elif op[0] == 0x03:
                                addr = op[1]
                                value = op[2]
                                # mask (op[3]) is ignored, as control packet cannot do masked write
                                control_packet(
                                    address=addr,
                                    opcode=0,
                                    stream_id=0,
                                    data=np.array([value], dtype=np.int32),
                                )
                            else:
                                raise Exception(f"Unhandled op: {op:#x}")

                else:

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
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-file", "-f", type=argparse.FileType("rb"), nargs="*")
    parser.add_argument(
        "-generate-ctrl-pkt",
        dest="mlir_ctrl_pkt",
        default=False,
        action="store_true",
        help="Enable MLIR control packet op generation",
    )
    args = parser.parse_args()

    # Process each file provided as command line argument
    operations = []
    for f in args.file:
        # Read the data from the file
        data = f.read()
        # Parse the TXN data
        columns, ops = parse_txn(data)
        operations = operations + ops

    module = operations_to_mlir(operations, columns, mlir_ctrl_pkt=args.mlir_ctrl_pkt)

    print(str(module))
