// RUN: aie-opt --verify-diagnostics --split-input-file %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each initial value is one object's dense contents, in the object's shape
// and element type.

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    // expected-error@+1 {{'aie.objectfifo' op initial value of type 'memref<2xi32>' does not match the object type 'memref<4xi32>'}}
    aie.objectfifo @y(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<[0, 1]> : memref<2xi32>, dense<[2, 3]> : memref<2xi32>]
  }
}

// -----

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    // expected-error@+1 {{'aie.objectfifo' op initial value of type 'memref<4xi16>' does not match the object type 'memref<4xi32>'}}
    aie.objectfifo @y(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<1> : memref<4xi16>, dense<2> : memref<4xi16>]
  }
}

// -----

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    // expected-error@+1 {{'aie.objectfifo' op initial value dense_resource<blob> : memref<2xi16> is not a dense elements attribute}}
    aie.objectfifo @y(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<2xi16>> = [dense_resource<blob> : memref<2xi16>, dense_resource<blob> : memref<2xi16>]
  }
}
{-#
  dialect_resources: {
    builtin: {
      blob: "0x0200000001000200"
    }
  }
#-}

// -----

module {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %m = aie.tile(0, 1)
    aie.objectfifo @y(%m, {%s}, 2 : i32) : !aie.objectfifo<memref<4xi32>> = [dense<7> : memref<4xi32>, dense<[0, 1, 2, 3]> : memref<4xi32>]
  }
}
