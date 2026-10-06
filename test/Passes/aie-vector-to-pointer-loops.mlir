//===- aie-vector-to-pointer-loops.mlir ------------------------*- MLIR -*-===//
//
// Copyright (C) 2025 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-vector-to-pointer-loops %s | FileCheck %s

// Test vector-to-pointer loop transformations

// CHECK-LABEL: @test1
aie.device(xcvc1902) @test1 {
  %tile = aie.tile(1, 1)
  %buf = aie.buffer(%tile) : memref<1024xi32, 2 : i32>
  
  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    
    // CHECK: builtin.unrealized_conversion_cast
    // CHECK: ptr.to_ptr
    // CHECK: scf.for
    // CHECK-SAME: -> (!ptr.ptr<#ptr.generic_space>)
    %result = scf.for %i = %c0 to %c16 step %c1 iter_args(%idx = %c0) -> (index) {
      // CHECK: ptr.get_metadata
      // CHECK: ptr.from_ptr
      // CHECK: vector.load
      // CHECK-SAME: [%c0]
      %vec = vector.load %buf[%idx] : memref<1024xi32, 2 : i32>, vector<16xi32>
      // CHECK: vector.store
      // CHECK-SAME: [%c0]
      vector.store %vec, %buf[%idx] : memref<1024xi32, 2 : i32>, vector<16xi32>
      // CHECK: ptr.ptr_add
      %next_idx = arith.addi %idx, %c1 : index
      // CHECK: scf.yield
      scf.yield %next_idx : index
    }
    aie.end
  }
}

// A buffer in the default (attribute-less) memory space is cast to
// generic_space too; ptr.to_ptr rejects a memref whose space differs from the
// pointer's.
// CHECK-LABEL: @default_space
aie.device(npu2) @default_space {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    // CHECK: %[[CAST:.*]] = builtin.unrealized_conversion_cast %{{.*}} : memref<1024xi32> to memref<1024xi32, #ptr.generic_space>
    // CHECK: ptr.to_ptr %[[CAST]] : memref<1024xi32, #ptr.generic_space> -> <#ptr.generic_space>
    // CHECK: scf.for
    // CHECK-SAME: -> (!ptr.ptr<#ptr.generic_space>)
    %result = scf.for %i = %c0 to %c64 step %c16 iter_args(%idx = %c0) -> (index) {
      // CHECK: ptr.from_ptr
      // CHECK: vector.load
      %vec = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
      // CHECK: ptr.from_ptr
      // CHECK: vector.store
      vector.store %vec, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      // CHECK: ptr.ptr_add
      %next_idx = arith.addi %idx, %c16 : index
      scf.yield %next_idx : index
    }
    aie.end
  }
}

// Each pointer is advanced by its own memref's element size: 64 x i16 is 128
// bytes, 64 x i32 is 256.
// CHECK-LABEL: @mixed_element_sizes
aie.device(npu2) @mixed_element_sizes {
  %tile = aie.tile(0, 2)
  %in = aie.buffer(%tile) : memref<1024xi16>
  %out = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    // CHECK-DAG: %[[C256:.*]] = arith.constant 256 : index
    // CHECK-DAG: %[[C128:.*]] = arith.constant 128 : index
    // CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}, %[[Q:.*]] = %{{.*}}) -> (!ptr.ptr<#ptr.generic_space>, !ptr.ptr<#ptr.generic_space>)
    // CHECK: ptr.from_ptr %[[P]]
    // CHECK: vector.load {{.*}} : memref<1024xi16, #ptr.generic_space>, vector<64xi16>
    // CHECK: %[[NP:.*]] = ptr.ptr_add %[[P]], %[[C128]]
    // CHECK: ptr.from_ptr %[[Q]]
    // CHECK: vector.store {{.*}} : memref<1024xi32, #ptr.generic_space>, vector<64xi32>
    // CHECK: %[[NQ:.*]] = ptr.ptr_add %[[Q]], %[[C256]]
    // CHECK: scf.yield %[[NP]], %[[NQ]]
    // The loop keeps its attributes.
    // CHECK: } {loop_annotation = #{{.*}}}
    %r:2 = scf.for %i = %c0 to %c16 step %c1 iter_args(%p = %c0, %q = %c0) -> (index, index) {
      %v = vector.load %in[%p] : memref<1024xi16>, vector<64xi16>
      %e = arith.extsi %v : vector<64xi16> to vector<64xi32>
      %np = arith.addi %p, %c64 : index
      vector.store %e, %out[%q] : memref<1024xi32>, vector<64xi32>
      %nq = arith.addi %q, %c64 : index
      scf.yield %np, %nq : index, index
    } {loop_annotation = #llvm.loop_annotation<unroll = <disable = true>>}
    aie.end
  }
}

// One index walking two buffers cannot become one pointer; the loop is left
// alone.
// CHECK-LABEL: @shared_index
aie.device(npu2) @shared_index {
  %tile = aie.tile(0, 2)
  %a = aie.buffer(%tile) : memref<1024xi32>
  %b = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index

    // CHECK-NOT: ptr.
    // CHECK: scf.for {{.*}} -> (index)
    // CHECK: vector.load %{{.*}}[%{{.*}}] : memref<1024xi32>, vector<16xi32>
    // CHECK: vector.store %{{.*}}, %{{.*}}[%{{.*}}] : memref<1024xi32>, vector<16xi32>
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %r = scf.for %i = %c0 to %c1024 step %c16 iter_args(%idx = %c0) -> (index) {
      %v = vector.load %a[%idx] : memref<1024xi32>, vector<16xi32>
      vector.store %v, %b[%idx] : memref<1024xi32>, vector<16xi32>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    aie.end
  }
}

// A load whose index is not loop-carried (here a constant from outside the
// loop) stays as it is next to the converted one.
// CHECK-LABEL: @invariant_index_load
aie.device(npu2) @invariant_index_load {
  %tile = aie.tile(0, 2)
  %acc = aie.buffer(%tile) : memref<16xbf16, 2 : i32>
  %in = aie.buffer(%tile) : memref<64xbf16, 2 : i32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index

    // CHECK: scf.for {{.*}} -> (!ptr.ptr<#ptr.generic_space>)
    // CHECK: vector.load %{{.*}}[%c0] : memref<16xbf16, 2 : i32>, vector<16xbf16>
    // CHECK: ptr.from_ptr
    // CHECK: vector.load {{.*}} : memref<64xbf16, #ptr.generic_space>, vector<16xbf16>
    // CHECK: ptr.ptr_add
    %r = scf.for %i = %c16 to %c64 step %c16 iter_args(%idx = %c16) -> (index) {
      %a = vector.load %acc[%c0] : memref<16xbf16, 2 : i32>, vector<16xbf16>
      %v = vector.load %in[%idx] : memref<64xbf16, 2 : i32>, vector<16xbf16>
      %s = arith.addf %a, %v : vector<16xbf16>
      vector.store %s, %acc[%c0] : memref<16xbf16, 2 : i32>, vector<16xbf16>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    aie.end
  }
}

// The index is also read as a scalar, so it has to stay an index.
// CHECK-LABEL: @index_has_scalar_use
aie.device(npu2) @index_has_scalar_use {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>
  %lut = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index

    // CHECK-NOT: ptr.
    // CHECK: scf.for {{.*}} -> (index)
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %r = scf.for %i = %c0 to %c1024 step %c16 iter_args(%idx = %c0) -> (index) {
      %v = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %s = memref.load %lut[%idx] : memref<1024xi32>
      %b = vector.broadcast %s : i32 to vector<16xi32>
      %w = arith.addi %v, %b : vector<16xi32>
      vector.store %w, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    aie.end
  }
}

// The loop's final index is used after the loop, so it has to stay an index.
// CHECK-LABEL: @index_result_used
aie.device(npu2) @index_result_used {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c512 = arith.constant 512 : index

    // CHECK-NOT: ptr.
    // CHECK: %[[END:.*]] = scf.for {{.*}} -> (index)
    // CHECK: vector.load %{{.*}}[%[[END]]]
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %end = scf.for %i = %c0 to %c512 step %c16 iter_args(%idx = %c0) -> (index) {
      %v = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
      vector.store %v, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    %tail = vector.load %buf[%end] : memref<1024xi32>, vector<16xi32>
    vector.store %tail, %buf[%c0] : memref<1024xi32>, vector<16xi32>
    aie.end
  }
}

// The increment may name the pointer as either operand.
// CHECK-LABEL: @commuted_increment
aie.device(npu2) @commuted_increment {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index

    // CHECK-DAG: %[[C64:.*]] = arith.constant 64 : index
    // CHECK: scf.for {{.*}} iter_args(%[[P:.*]] = %{{.*}}) -> (!ptr.ptr<#ptr.generic_space>)
    // CHECK: %[[N:.*]] = ptr.ptr_add %[[P]], %[[C64]]
    // CHECK: scf.yield %[[N]]
    %r = scf.for %i = %c0 to %c1024 step %c16 iter_args(%idx = %c0) -> (index) {
      %v = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %w = arith.addi %v, %v : vector<16xi32>
      vector.store %w, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %n = arith.addi %c16, %idx : index
      scf.yield %n : index
    }
    aie.end
  }
}

// The index is used inside a nested loop, which the rewrite does not visit.
// CHECK-LABEL: @nested_use
aie.device(npu2) @nested_use {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %c1024 = arith.constant 1024 : index

    // CHECK-NOT: ptr.
    // CHECK: scf.for {{.*}} -> (index)
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %r = scf.for %i = %c0 to %c1024 step %c64 iter_args(%idx = %c0) -> (index) {
      scf.for %j = %c0 to %c4 step %c1 {
        %v = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
        vector.store %v, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      }
      %n = arith.addi %idx, %c64 : index
      scf.yield %n : index
    }
    aie.end
  }
}

// The accessed memref is created inside the loop, so no pointer to it can be
// formed before the loop.
// CHECK-LABEL: @base_defined_in_loop
aie.device(npu2) @base_defined_in_loop {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<4x256xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index

    // CHECK-NOT: ptr.
    // CHECK: scf.for {{.*}} -> (index)
    // CHECK: memref.subview
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %r = scf.for %i = %c0 to %c4 step %c1 iter_args(%idx = %c0) -> (index) {
      %row = memref.subview %buf[%i, 0] [1, 256] [1, 1] : memref<4x256xi32> to memref<256xi32, strided<[1], offset: ?>>
      %v = vector.load %row[%idx] : memref<256xi32, strided<[1], offset: ?>>, vector<16xi32>
      vector.store %v, %row[%idx] : memref<256xi32, strided<[1], offset: ?>>, vector<16xi32>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    aie.end
  }
}

// Two indices into the same buffer are combined into a third index; turning
// both into pointers would add a pointer to a pointer, so the loop is left
// alone.
// CHECK-LABEL: @summed_indices
aie.device(npu2) @summed_indices {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c256 = arith.constant 256 : index

    // CHECK-NOT: ptr.
    // CHECK: scf.for {{.*}} -> (index, index)
    // CHECK-NOT: ptr.
    // CHECK: aie.end
    %r:2 = scf.for %i = %c0 to %c256 step %c16 iter_args(%a = %c0, %b = %c0) -> (index, index) {
      %va = vector.load %buf[%a] : memref<1024xi32>, vector<16xi32>
      %vb = vector.load %buf[%b] : memref<1024xi32>, vector<16xi32>
      %s = arith.addi %a, %b : index
      %w = arith.addi %va, %vb : vector<16xi32>
      vector.store %w, %buf[%s] : memref<1024xi32>, vector<16xi32>
      %na = arith.addi %a, %c16 : index
      %nb = arith.addi %b, %c16 : index
      scf.yield %na, %nb : index, index
    }
    aie.end
  }
}

// An unsigned-compare loop stays unsigned after the rewrite.
// CHECK-LABEL: @unsigned_compare_kept
aie.device(npu2) @unsigned_compare_kept {
  %tile = aie.tile(0, 2)
  %buf = aie.buffer(%tile) : memref<1024xi32>

  %core = aie.core(%tile) {
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1024 = arith.constant 1024 : index

    // CHECK: scf.for unsigned {{.*}} -> (!ptr.ptr<#ptr.generic_space>)
    %r = scf.for unsigned %i = %c0 to %c1024 step %c16 iter_args(%idx = %c0) -> (index) {
      %v = vector.load %buf[%idx] : memref<1024xi32>, vector<16xi32>
      vector.store %v, %buf[%idx] : memref<1024xi32>, vector<16xi32>
      %n = arith.addi %idx, %c16 : index
      scf.yield %n : index
    }
    aie.end
  }
}
