//===- roundtrip_maskpoll_forms.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: rm -rf roundtrip_maskpoll_forms && mkdir -p roundtrip_maskpoll_forms
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v01_aiert.bin --version 0.1 --opcode 4 --op-size 32
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v01_28.bin --version 0.1 --opcode 4 --op-size 28
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v10.bin --version 1.0 --opcode 4
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v01_busy.bin --version 0.1 --opcode 7 --op-size 32
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v10_busy.bin --version 1.0 --opcode 7
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v01_aiert.bin | FileCheck %s
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v01_28.bin | FileCheck %s
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v10.bin | FileCheck %s
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v01_busy.bin 2>&1 | FileCheck %s --check-prefix=BUSY
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v10_busy.bin 2>&1 | FileCheck %s --check-prefix=BUSY
// RUN: %python %S/Inputs/write_maskpoll_txn.py roundtrip_maskpoll_forms/v01_short.bin --version 0.1 --opcode 4 --op-size 20
// RUN: %python txn2mlir.py -f roundtrip_maskpoll_forms/v01_short.bin 2>&1 | FileCheck %s --check-prefix=SHORT

// Each encoding of a mask poll that aie-rt writes (the v0.1 header with the
// 32-byte operation aie-rt serializes or the 28-byte one TxnEncoding.h emits,
// and the v1.0 header with its 16-byte operation) parses back to a maskpoll of
// the same register, value and mask. A busy poll has no counterpart in the IR
// and is rejected rather than read as a plain poll.
// CHECK: aie.device(npu1_1col)
// CHECK: %[[ADDR:.*]] = arith.constant 2211840 : i32
// CHECK: %[[VAL:.*]] = arith.constant 2 : i32
// CHECK: %[[MASK:.*]] = arith.constant 6 : i32
// CHECK: aiex.npu.maskpoll(%[[ADDR]], %[[VAL]], %[[MASK]]) : i32, i32, i32
// CHECK-NOT: aiex.npu

// BUSY: Unhandled opcode: 7
// BUSY-NEXT: Failed to parse binary

// A v0.1 operation whose size is below its header is rejected, not
// misparsed.
// SHORT: Transaction op size 20 is smaller than its 28-byte header
// SHORT-NEXT: Failed to parse binary
