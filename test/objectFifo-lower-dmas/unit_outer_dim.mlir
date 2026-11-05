// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A core tile's DMA steps through 3 dimensions. The consumer's layout gives 4,
// the outermost of size 1, which reads one position, so its descriptor keeps
// the inner 3.

// CHECK-LABEL: @unit_outer_dim
// CHECK: aie.mem(
// CHECK: aie.dma_bd(%{{.*}} : memref<1920xi8> offset = 0 len = 1920 sizes = [160, 3, 4] strides = [4, 640, 1])
module @unit_outer_dim {
  aie.device(npu2) {
    %s = aie.tile(0, 0)
    %t = aie.tile(0, 2)
    aie.objectfifo @in(%s, {%t dimensionsFromStream [<size = 1, stride = 0>, <size = 160, stride = 4>, <size = 3, stride = 640>, <size = 4, stride = 1>]}, 2 : i32) : !aie.objectfifo<memref<1920xi8>>
    aie.core(%t) {
      %x = aie.objectfifo.acquire @in(Consume, 1) : memref<1920xi8>
      aie.objectfifo.release @in(Consume, 1)
      aie.end
    }
  }
}
