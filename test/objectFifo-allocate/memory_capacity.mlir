// RUN: aie-opt --split-input-file --aie-objectfifo-allocate --aie-assign-buffer-addresses="alloc-scheme=basic" %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Multiple pools referencing one fixed buffer count it only once.
module @shared_fixed_exact_capacity {
  aie.device(npu2_1col) {
    %mem = aie.tile(0, 1)
    %b = aie.buffer(%mem) {sym_name = "b"} : memref<524288xi8>
    aie.objectfifo.pool @p(%mem) {depth = 1 : i32, buffers = [@b]} : memref<524288xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 524288 : i32}
    }
    aie.objectfifo.pool @q(%mem) {depth = 1 : i32, buffers = [@b]} : memref<524288xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 524288 : i32}
    }
    aie.objectfifo.dma_endpoint @reader(%mem) drains @p
    aie.objectfifo.dma_endpoint @writer(%mem) fills @q
  }
}
// CHECK-LABEL: module @shared_fixed_exact_capacity
// CHECK: aie.buffer({{.*}}) {address = 0 : i64, sym_name = "b"}
// CHECK: @reader({{.*}}) drains @p {channelIndex = 0 : i32}
// CHECK: @writer({{.*}}) fills @q {channelIndex = 0 : i32}

// -----

// Size-first ordering uses allocated bytes, not packed bits: 96xi1 precedes
// 32xi8. The two generated buffers exactly fill the remaining storage.
module @sub_byte_exact_capacity {
  aie.device(npu2_1col) {
    %mem = aie.tile(0, 1)
    %reserved = aie.buffer(%mem) {sym_name = "reserved"} : memref<524160xi8>
    aie.objectfifo.pool @small(%mem) {depth = 1 : i32} : memref<32xi8> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 32 : i32}
    }
    aie.objectfifo.pool @large(%mem) {depth = 1 : i32} : memref<96xi1> {
      aie.objectfifo.segment @s {offset = 0 : i32, size = 96 : i32}
    }
    aie.objectfifo.dma_endpoint @small_reader(%mem) drains @small
    aie.objectfifo.dma_endpoint @large_reader(%mem) drains @large
  }
}
// CHECK-LABEL: module @sub_byte_exact_capacity
// CHECK: aie.buffer({{.*}}) {{.*}}sym_name = "large_buff_0"
// CHECK: aie.buffer({{.*}}) {{.*}}sym_name = "small_buff_0"
