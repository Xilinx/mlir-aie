//===- npu_buffer_write_invalid.mlir ---------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --split-input-file --verify-diagnostics %s

// A symbolic clear must not overwrite the buffer following an eight-word acc.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    memref.global "private" constant @zeros32 : memref<32xi32> = dense<0>
    aie.runtime_sequence() {
      %z = memref.get_global @zeros32 : memref<32xi32>
      // expected-error @+1 {{write range [0, 128) exceeds buffer @acc size 32 bytes}}
      aiex.npu.blockwrite(%z) {buffer = @acc, address = 0 : ui32} : memref<32xi32>
    }
  }
}

// -----

// The offset is in words, not bytes, and contributes to the written interval.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    memref.global "private" constant @words2 : memref<2xi32> = dense<0>
    aie.runtime_sequence() {
      %z = memref.get_global @words2 : memref<2xi32>
      // expected-error @+1 {{write range [28, 36) exceeds buffer @acc size 32 bytes}}
      aiex.npu.blockwrite(%z) {buffer = @acc, address = 7 : ui32} : memref<2xi32>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence @write(%value: i32) {
      %offset = arith.constant 8 : i32
      // expected-error @+1 {{write range [32, 36) exceeds buffer @acc size 32 bytes}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence @masked(%value: i32, %mask: i32) {
      %offset = arith.constant 8 : i32
      // expected-error @+1 {{write range [32, 36) exceeds buffer @acc size 32 bytes}}
      aiex.npu.maskwrite32(%offset, %value, %mask) {buffer = @acc} : i32, i32, i32
    }
  }
}

// -----

// getLocalMemorySize() for npu2 is 65536 bytes: start at its last word.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %tail = aie.buffer(%tile) {sym_name = "tail", address = 65532 : i32, aligned = false} : memref<2xi32>
    memref.global "private" constant @words2 : memref<2xi32> = dense<0>
    aie.runtime_sequence() {
      %z = memref.get_global @words2 : memref<2xi32>
      // expected-error @+1 {{write range [65532, 65540) exceeds tile data memory size 65536 bytes}}
      aiex.npu.blockwrite(%z) {buffer = @tail, address = 0 : ui32} : memref<2xi32>
    }
  }
}

// -----

// getMemTileSize() is 524288 bytes; do not use the core tile's capacity.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 1)
    %tail = aie.buffer(%tile) {sym_name = "tail", address = 524284 : i32, aligned = false} : memref<2xi32>
    memref.global "private" constant @words2 : memref<2xi32> = dense<0>
    aie.runtime_sequence() {
      %z = memref.get_global @words2 : memref<2xi32>
      // expected-error @+1 {{write range [524284, 524292) exceeds tile data memory size 524288 bytes}}
      aiex.npu.blockwrite(%z) {buffer = @tail, address = 0 : ui32} : memref<2xi32>
    }
  }
}

// -----

// A half-word base, not word offset 2, makes a symbolic write unaligned.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc", address = 2 : i32, aligned = false} : memref<8xi32>
    memref.global "private" constant @word : memref<1xi32> = dense<0>
    aie.runtime_sequence() {
      %z = memref.get_global @word : memref<1xi32>
      // expected-error @+1 {{resolved buffer write address 2 is not 4-byte aligned}}
      aiex.npu.blockwrite(%z) {buffer = @acc, address = 0 : ui32} : memref<1xi32>
    }
  }
}

// -----

// Compare destination storage bytes, not its element count times four.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %bytes = aie.buffer(%tile) {sym_name = "bytes"} : memref<8xi8>
    aie.runtime_sequence(%data: memref<3xi32>) {
      // expected-error @+1 {{write range [0, 12) exceeds buffer @bytes size 8 bytes}}
      aiex.npu.blockwrite(%data) {buffer = @bytes, address = 0 : ui32} : memref<3xi32>
    }
  }
}

// -----

// This word offset would wrap to zero if multiplied in uint32_t.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 1073741824 : i32
      // expected-error @+1 {{write range [4294967296, 4294967300) exceeds buffer @acc size 32 bytes}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Negative i32 constants retain their unsigned address bit pattern.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant -1 : i32
      // expected-error @+1 {{write range [17179869180, 17179869184) exceeds buffer @acc size 32 bytes}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc", address = -4 : i32} : memref<8xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{referenced buffer address must be nonnegative}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Even a zero mask addresses a whole word and cannot waive its alignment.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc", address = 2 : i32, aligned = false} : memref<8xi32>
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      // expected-error @+1 {{resolved buffer write address 2 is not 4-byte aligned}}
      aiex.npu.maskwrite32(%zero, %value, %zero) {buffer = @acc} : i32, i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%offset: i32, %value: i32) {
      // expected-error @+1 {{address must be constant when buffer is present}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Arith-derived symbolic offsets must not fall through as raw EmitC addresses.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%offset: i32, %value: i32, %mask: i32) {
      %one = arith.constant 1 : i32
      %derived = arith.addi %offset, %one : i32
      // expected-error @+1 {{address must be constant when buffer is present}}
      aiex.npu.maskwrite32(%derived, %value, %mask) {buffer = @acc} : i32, i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<?xi32>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<?xi32>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<vector<[4]xi32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<vector<[4]xi32>>
    }
  }
}

// -----

// A fixed-vector element must not overflow the layout's signed element count.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<vector<4611686018427387904x4xi32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<vector<4611686018427387904x4xi32>>
    }
  }
}

// -----

// A signed element count can fit while the layout's unsigned bit product wraps.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<vector<576460752303423488xi32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<vector<576460752303423488xi32>>
    }
  }
}

// -----

// The default vector-layout query requires a nonempty innermost dimension.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<vector<i32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<vector<i32>>
    }
  }
}

// -----

// A static shape can still overflow storage arithmetic (2^62 words * 4).
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<4611686018427387904xi32>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<4611686018427387904xi32>
    }
  }
}

// -----

// The payload itself fits uint64_t bytes, but adding two word offsets does not.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<4611686018427387903xi32>) {
      // expected-error @+1 {{buffer-relative write range overflows 64-bit bytes}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 2 : ui32} : memref<4611686018427387903xi32>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<8xi32>
    aie.runtime_sequence(%data: memref<2xi16>) {
      // expected-error @+1 {{payload must have 32-bit elements when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<2xi16>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{buffer 'missing' not found in device}}
      aiex.npu.write32(%offset, %value) {buffer = @missing} : i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %lock = aie.lock(%tile, 0) {sym_name = "not_a_buffer"}
    aie.runtime_sequence(%value: i32, %mask: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{buffer 'not_a_buffer' not found in device}}
      aiex.npu.maskwrite32(%offset, %value, %mask) {buffer = @not_a_buffer} : i32, i32, i32
    }
  }
}

// -----

// No eventual nonnegative placement can fit this write into core data memory.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<16385xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 16384 : i32
      // expected-error @+1 {{write range [65536, 65540) exceeds tile data memory size 65536 bytes}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Generic syntax can bypass the buffer parser's static-shape check.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = "aie.buffer"(%tile) {sym_name = "acc"} : (index) -> memref<?xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{referenced buffer must have a static, fixed-size storage extent representable in 64 bits}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<vector<[4]xi32>>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{referenced buffer must have a static, fixed-size storage extent representable in 64 bits}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// The buffer's schema verifier diagnoses malformed forward definitions before
// the device region verifier dispatches graph-dependent named-use checks.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.buffer' op result #0 must be memref of any non-token type values, but got 'i32'}}
    %later = "aie.buffer"(%tile) {sym_name = "later"} : (index) -> i32
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.buffer' op requires one result}}
    "aie.buffer"(%tile) {sym_name = "later"} : (index) -> ()
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.buffer' op requires a single operand}}
    %later = "aie.buffer"() {sym_name = "later"} : () -> memref<1xi32>
  }
}

// -----

module {
  aie.device(npu2) {
    %not_a_tile = arith.constant 0 : i32
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.buffer' op operand #0 must be index, but got 'i32'}}
    %later = "aie.buffer"(%not_a_tile) {sym_name = "later"} : (i32) -> memref<1xi32>
  }
}

// -----

// Base attribute typing belongs to the buffer's schema verifier.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.buffer' op attribute 'address' failed to satisfy constraint: 32-bit signless integer attribute}}
    %later = "aie.buffer"(%tile) {sym_name = "later", address = 0 : i64} : (index) -> memref<1xi32>
  }
}

// -----

// Tile schema invariants likewise precede named-use verification.
module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.tile' op requires attribute 'col'}}
    %tile = "aie.tile"() {row = 2 : i32} : () -> index
    %later = "aie.buffer"(%tile) {sym_name = "later"} : (index) -> memref<1xi32>
  }
}

// -----

module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.tile' op attribute 'col' failed to satisfy constraint: 32-bit signless integer attribute whose minimum value is 0}}
    %tile = "aie.tile"() {col = 0 : i64, row = 2 : i32} : () -> index
    %later = "aie.buffer"(%tile) {sym_name = "later"} : (index) -> memref<1xi32>
  }
}

// -----

// Logical tile kind is also checked by its owner before named-use dispatch.
module {
  aie.device(npu2) {
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      aiex.npu.write32(%zero, %value) {buffer = @later} : i32, i32
    }
    // expected-error @+1 {{'aie.logical_tile' op requires attribute 'tile_type'}}
    %tile = "aie.logical_tile"() : () -> index
    %later = "aie.buffer"(%tile) {sym_name = "later"} : (index) -> memref<1xi32>
  }
}

// -----

// Nested memrefs are legal builtin elements but have no storage layout query.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
    aie.runtime_sequence(%data: memref<1xmemref<1xi32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32}
        : memref<1xmemref<1xi32>>
    }
  }
}

// -----

module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
    aie.runtime_sequence(%data: memref<1xmemref<*xi32>>) {
      // expected-error @+1 {{payload must have a static, fixed-size storage extent representable in 64 bits when buffer is present}}
      aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<1xmemref<*xi32>>
    }
  }
}

// -----

// The same layout-query guard protects the destination, not just payloads.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xmemref<1xi32>>
    aie.runtime_sequence(%value: i32) {
      %zero = arith.constant 0 : i32
      // expected-error @+1 {{referenced buffer must have a static, fixed-size storage extent representable in 64 bits}}
      aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Guard the destination shape before BufferOp::getAllocationSize could wrap.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<4611686018427387904xi32>
    aie.runtime_sequence(%value: i32) {
      %offset = arith.constant 0 : i32
      // expected-error @+1 {{referenced buffer must have a static, fixed-size storage extent representable in 64 bits}}
      aiex.npu.write32(%offset, %value) {buffer = @acc} : i32, i32
    }
  }
}

// -----

// Isolated functions are not skipped by the device's named-use walk.
module {
  aie.device(npu2) {
    %tile = aie.tile(0, 2)
    func.func @write(%value: i32) {
      %one = arith.constant 1 : i32
      // expected-error @+1 {{write range [4, 8) exceeds buffer @later size 4 bytes}}
      aiex.npu.write32(%one, %value) {buffer = @later} : i32, i32
      return
    }
    %later = aie.buffer(%tile) {sym_name = "later"} : memref<1xi32>
  }
}

// -----

// A preceding device's same-named larger buffer must not mask an overrun.
module {
  aie.device(npu2) @large {
    %tile = aie.tile(0, 2)
    %shared = aie.buffer(%tile) {sym_name = "shared"} : memref<2xi32>
    aie.runtime_sequence(%value: i32) {
      %one = arith.constant 1 : i32
      aiex.npu.write32(%one, %value) {buffer = @shared} : i32, i32
    }
  }
  aie.device(npu2) @small {
    %tile = aie.tile(0, 2)
    %shared = aie.buffer(%tile) {sym_name = "shared"} : memref<1xi32>
    aie.runtime_sequence(%value: i32) {
      %one = arith.constant 1 : i32
      // expected-error @+1 {{write range [4, 8) exceeds buffer @shared size 4 bytes}}
      aiex.npu.write32(%one, %value) {buffer = @shared} : i32, i32
    }
  }
}

// -----

// A nested device cannot use a name owned only by the outer device.
module {
  aie.device(npu2) @outer {
    %tile = aie.tile(0, 2)
    %acc = aie.buffer(%tile) {sym_name = "acc"} : memref<1xi32>
    builtin.module @nested {
      aie.device(npu2) @inner {
        aie.runtime_sequence(%value: i32) {
          %zero = arith.constant 0 : i32
          // expected-error @+1 {{buffer 'acc' not found in device}}
          aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
        }
      }
    }
  }
}

// -----

// A buffer reference still requires a device even when no region hook can run.
module {
  func.func @write(%value: i32) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{must be inside a device when buffer is present}}
    aiex.npu.write32(%zero, %value) {buffer = @acc} : i32, i32
    return
  }
}

// -----

module {
  func.func @maskwrite(%value: i32, %mask: i32) {
    %zero = arith.constant 0 : i32
    // expected-error @+1 {{must be inside a device when buffer is present}}
    aiex.npu.maskwrite32(%zero, %value, %mask) {buffer = @acc} : i32, i32, i32
    return
  }
}

// -----

module {
  func.func @blockwrite(%data: memref<1xi32>) {
    // expected-error @+1 {{must be inside a device when buffer is present}}
    aiex.npu.blockwrite(%data) {buffer = @acc, address = 0 : ui32} : memref<1xi32>
    return
  }
}
