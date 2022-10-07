// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: physical.buffer
%buffer1 = physical.buffer(): memref<1024xi32>

// CHECK: physical.lock
%lock1 = physical.lock<0>()

// CHECK: physical.buffer
%buffer2 = physical.buffer(): memref<1024xi32>

// CHECK: physical.lock
%lock2 = physical.lock<0>()

// CHECK: physical.stream
%stream:2 = physical.stream<[0, 1]>(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream_dma
physical.stream_dma(%stream#1: !physical.istream<i32>) {

  // CHECK: physical.stream_dma_connect
  %0 = physical.stream_dma_connect<0>(
      %lock1[0->1], %buffer1[0:1024]: memref<1024xi32>, %1)

  // CHECK: physical.stream_dma_connect
  %1 = physical.stream_dma_connect<1>(
      %lock2[0->1], %buffer2[0:1024]: memref<1024xi32>)

}

// CHECK: physical.stream_dma
physical.stream_dma(%stream#1: !physical.istream<i32>) {

  // CHECK: physical.stream_dma_connect
  %0 = physical.stream_dma_connect(
      %lock1[0->1], %buffer1[0:1024]: memref<1024xi32>, %0)

}
