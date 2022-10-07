// REQUIRES: aie_found
// RUN: phy-opt --convert-physical-to-aie %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @MM_2x2
module @MM_2x2 {

// CHECK:  %0 = AIE.tile(7, 0)
// CHECK:  %1 = AIE.lock(%0, 3)
// CHECK:  AIE.useLock(%1, Release, 0)
// CHECK:  %2 = AIE.lock(%0, 2)
// CHECK:  AIE.useLock(%2, Release, 0)
// CHECK:  %3 = AIE.lock(%0, 1)
// CHECK:  AIE.useLock(%3, Release, 0)
// CHECK:  %4 = AIE.lock(%0, 0)
// CHECK:  AIE.useLock(%4, Release, 0)
// CHECK:  %5 = AIE.tile(7, 4)
// CHECK:  %6 = AIE.lock(%5, 2)
// CHECK:  AIE.useLock(%6, Release, 0)
// CHECK:  %7 = AIE.lock(%5, 1)
// CHECK:  AIE.useLock(%7, Release, 0)
// CHECK:  %8 = AIE.lock(%5, 0)
// CHECK:  AIE.useLock(%8, Release, 0)
// CHECK:  %9 = AIE.tile(6, 4)
// CHECK:  %10 = AIE.lock(%9, 2)
// CHECK:  AIE.useLock(%10, Release, 0)
// CHECK:  %11 = AIE.lock(%9, 1)
// CHECK:  AIE.useLock(%11, Release, 0)
// CHECK:  %12 = AIE.lock(%9, 0)
// CHECK:  AIE.useLock(%12, Release, 0)
// CHECK:  %13 = AIE.tile(7, 3)
// CHECK:  %14 = AIE.lock(%13, 3)
// CHECK:  AIE.useLock(%14, Release, 0)
// CHECK:  %15 = AIE.lock(%13, 2)
// CHECK:  AIE.useLock(%15, Release, 1)
// CHECK:  %16 = AIE.lock(%13, 1)
// CHECK:  AIE.useLock(%16, Release, 0)
// CHECK:  %17 = AIE.lock(%13, 0)
// CHECK:  AIE.useLock(%17, Release, 0)
// CHECK:  %18 = AIE.tile(6, 3)
// CHECK:  %19 = AIE.lock(%18, 3)
// CHECK:  AIE.useLock(%19, Release, 0)
// CHECK:  %20 = AIE.lock(%18, 2)
// CHECK:  AIE.useLock(%20, Release, 1)
// CHECK:  %21 = AIE.lock(%18, 1)
// CHECK:  AIE.useLock(%21, Release, 0)
// CHECK:  %22 = AIE.lock(%18, 0)
// CHECK:  AIE.useLock(%22, Release, 0)
// CHECK:  %23 = AIE.tile(6, 0)
// CHECK:  %24 = AIE.lock(%23, 3)
// CHECK:  AIE.useLock(%24, Release, 0)
// CHECK:  %25 = AIE.lock(%23, 2)
// CHECK:  AIE.useLock(%25, Release, 0)
// CHECK:  %26 = AIE.lock(%23, 1)
// CHECK:  AIE.useLock(%26, Release, 0)
// CHECK:  %27 = AIE.lock(%23, 0)
// CHECK:  AIE.useLock(%27, Release, 0)

  %leA_0   = physical.lock<0>() { aie.tile = "6.0", aie.id = "0" }
  %leA_1   = physical.lock<0>() { aie.tile = "6.0", aie.id = "1" }
  %leB_0_0 = physical.lock<0>() { aie.tile = "6.0", aie.id = "2" }
  %leB_0_1 = physical.lock<0>() { aie.tile = "6.0", aie.id = "3" }
  %leB_1_0 = physical.lock<0>() { aie.tile = "7.0", aie.id = "0" }
  %leB_1_1 = physical.lock<0>() { aie.tile = "7.0", aie.id = "1" }
  %leC_0   = physical.lock<0>() { aie.tile = "7.0", aie.id = "2" }
  %leC_1   = physical.lock<0>() { aie.tile = "7.0", aie.id = "3" }
  %lA_0_a  = physical.lock<0>() { aie.tile = "6.3", aie.id = "0" }
  %lA_0_b  = physical.lock<0>() { aie.tile = "7.3", aie.id = "0" }
  %lA_1_a  = physical.lock<0>() { aie.tile = "6.4", aie.id = "0" }
  %lA_1_b  = physical.lock<0>() { aie.tile = "7.4", aie.id = "0" }
  %lB_0_0  = physical.lock<0>() { aie.tile = "6.3", aie.id = "1" }
  %lB_0_1  = physical.lock<0>() { aie.tile = "6.4", aie.id = "1" }
  %lB_1_0  = physical.lock<0>() { aie.tile = "7.3", aie.id = "1" }
  %lB_1_1  = physical.lock<0>() { aie.tile = "7.4", aie.id = "1" }
  %lS_0_0  = physical.lock<1>() { aie.tile = "6.3", aie.id = "2" }
  %lS_0_1  = physical.lock<0>() { aie.tile = "6.3", aie.id = "3" }
  %lS_1_0  = physical.lock<1>() { aie.tile = "7.3", aie.id = "2" }
  %lS_1_1  = physical.lock<0>() { aie.tile = "7.3", aie.id = "3" }
  %lC_0    = physical.lock<0>() { aie.tile = "6.4", aie.id = "2" }
  %lC_1    = physical.lock<0>() { aie.tile = "7.4", aie.id = "2" }

// CHECK:  %28 = AIE.external_buffer 2203318222848 : memref<1024xi32>
// CHECK:  %29 = AIE.external_buffer 2203318226944 : memref<1024xi32>
// CHECK:  %30 = AIE.external_buffer 2203318231040 : memref<1024xi32>
// CHECK:  %31 = AIE.external_buffer 2203318235136 : memref<1024xi32>
// CHECK:  %32 = AIE.external_buffer 2203318239232 : memref<1024xi32>
// CHECK:  %33 = AIE.external_buffer 2203318243328 : memref<1024xi32>
// CHECK:  %34 = AIE.external_buffer 2203318247424 : memref<1024xi32>
// CHECK:  %35 = AIE.external_buffer 2203318251520 : memref<1024xi32>
// CHECK:  %36 = AIE.buffer(%18) : memref<1024xi32>
// CHECK:  %37 = AIE.buffer(%13) : memref<1024xi32>
// CHECK:  %38 = AIE.buffer(%9) : memref<1024xi32>
// CHECK:  %39 = AIE.buffer(%5) : memref<1024xi32>
// CHECK:  %40 = AIE.buffer(%18) : memref<1024xi32>
// CHECK:  %41 = AIE.buffer(%9) : memref<1024xi32>
// CHECK:  %42 = AIE.buffer(%13) : memref<1024xi32>
// CHECK:  %43 = AIE.buffer(%5) : memref<1024xi32>
// CHECK:  %44 = AIE.buffer(%18) : memref<1024xi32>
// CHECK:  %45 = AIE.buffer(%18) : memref<1024xi32>
// CHECK:  %46 = AIE.buffer(%13) : memref<1024xi32>
// CHECK:  %47 = AIE.buffer(%13) : memref<1024xi32>
// CHECK:  %48 = AIE.buffer(%9) : memref<1024xi32>
// CHECK:  %49 = AIE.buffer(%5) : memref<1024xi32>

  %eA_0   = physical.buffer() { aie.external_address = "2203318222848" }: memref<1024xi32>
  %eA_1   = physical.buffer() { aie.external_address = "2203318226944" }: memref<1024xi32>
  %eB_0_0 = physical.buffer() { aie.external_address = "2203318231040" }: memref<1024xi32>
  %eB_0_1 = physical.buffer() { aie.external_address = "2203318235136" }: memref<1024xi32>
  %eB_1_0 = physical.buffer() { aie.external_address = "2203318239232" }: memref<1024xi32>
  %eB_1_1 = physical.buffer() { aie.external_address = "2203318243328" }: memref<1024xi32>
  %eC_0   = physical.buffer() { aie.external_address = "2203318247424" }: memref<1024xi32>
  %eC_1   = physical.buffer() { aie.external_address = "2203318251520" }: memref<1024xi32>
  %A_0_a  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
  %A_0_b  = physical.buffer() { aie.tile = "7.3" }: memref<1024xi32>
  %A_1_a  = physical.buffer() { aie.tile = "6.4" }: memref<1024xi32>
  %A_1_b  = physical.buffer() { aie.tile = "7.4" }: memref<1024xi32>
  %B_0_0  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
  %B_0_1  = physical.buffer() { aie.tile = "6.4" }: memref<1024xi32>
  %B_1_0  = physical.buffer() { aie.tile = "7.3" }: memref<1024xi32>
  %B_1_1  = physical.buffer() { aie.tile = "7.4" }: memref<1024xi32>
  %S_0_0  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
  %S_0_1  = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
  %S_1_0  = physical.buffer() { aie.tile = "7.3" }: memref<1024xi32>
  %S_1_1  = physical.buffer() { aie.tile = "7.3" }: memref<1024xi32>
  %C_0    = physical.buffer() { aie.tile = "6.4" }: memref<1024xi32>
  %C_1    = physical.buffer() { aie.tile = "7.4" }: memref<1024xi32>

  %sA:2      = physical.stream<[0, 1]>(){ aie.tile = "6.0", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_0_a:2  = physical.stream<[0]>()   { aie.tile = "6.3", aie.port = "DMA.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_0_b:2  = physical.stream<[0]>()   { aie.tile = "7.3", aie.port = "DMA.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_1_a:2  = physical.stream<[1]>()   { aie.tile = "6.4", aie.port = "DMA.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sA_1_b:2  = physical.stream<[1]>()   { aie.tile = "7.4", aie.port = "DMA.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0:2    = physical.stream<[2, 3]>(){ aie.tile = "6.0", aie.port = "DMA.O", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0_0:2  = physical.stream<[2]>()   { aie.tile = "6.3", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_0_1:2  = physical.stream<[3]>()   { aie.tile = "6.4", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1:2    = physical.stream<[4, 5]>(){ aie.tile = "7.0", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1_0:2  = physical.stream<[4]>()   { aie.tile = "7.3", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sB_1_1:2  = physical.stream<[5]>()   { aie.tile = "7.4", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sC_0:2    = physical.stream<[6]>()   { aie.tile = "6.4", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %seC_0:2   = physical.stream<[6]>()   { aie.tile = "7.0", aie.port = "DMA.I", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %sC_1:2    = physical.stream<[7]>()   { aie.tile = "7.4", aie.port = "DMA.O", aie.id = "0" }: (!physical.ostream<i32>, !physical.istream<i32>)
  %seC_1:2   = physical.stream<[7]>()   { aie.tile = "7.0", aie.port = "DMA.I", aie.id = "1" }: (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK:  AIE.broadcast_packet(%23, DMA : 0) {
// CHECK:    AIE.bp_id(0) {
// CHECK:      AIE.bp_dest<%18, DMA : 0>
// CHECK:      AIE.bp_dest<%13, DMA : 0>
// CHECK:    }
// CHECK:    AIE.bp_id(1) {
// CHECK:      AIE.bp_dest<%9, DMA : 0>
// CHECK:      AIE.bp_dest<%5, DMA : 0>
// CHECK:    }
// CHECK:  }
// CHECK:  AIE.broadcast_packet(%23, DMA : 1) {
// CHECK:    AIE.bp_id(2) {
// CHECK:      AIE.bp_dest<%18, DMA : 1>
// CHECK:    }
// CHECK:    AIE.bp_id(3) {
// CHECK:      AIE.bp_dest<%9, DMA : 1>
// CHECK:    }
// CHECK:  }
// CHECK:  AIE.broadcast_packet(%0, DMA : 0) {
// CHECK:    AIE.bp_id(4) {
// CHECK:      AIE.bp_dest<%13, DMA : 1>
// CHECK:    }
// CHECK:    AIE.bp_id(5) {
// CHECK:      AIE.bp_dest<%5, DMA : 1>
// CHECK:    }
// CHECK:  }
// CHECK:  AIE.broadcast_packet(%9, DMA : 0) {
// CHECK:    AIE.bp_id(6) {
// CHECK:      AIE.bp_dest<%0, DMA : 0>
// CHECK:    }
// CHECK:  }
// CHECK:  AIE.broadcast_packet(%5, DMA : 0) {
// CHECK:    AIE.bp_id(7) {
// CHECK:      AIE.bp_dest<%0, DMA : 1>
// CHECK:    }
// CHECK:  }

  physical.stream_hub(
          %sA#1,
        %sB_0#1,   %sB_1#1,
        %sC_0#1,   %sC_1#1,
      %sA_0_a#0, %sA_0_b#0, %sA_1_a#0, %sA_1_b#0,
      %sB_0_0#0, %sB_0_1#0, %sB_1_0#0, %sB_1_1#0,
       %seC_0#0,  %seC_1#0)
    { aie.impl = "broadcast_packet" }
    : (!physical.istream<i32>,
       !physical.istream<i32>, !physical.istream<i32>,
       !physical.istream<i32>, !physical.istream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>, !physical.ostream<i32>,
       !physical.ostream<i32>, !physical.ostream<i32>)
    -> !physical.stream_hub<i32>

// CHECK:  func.func private @extern_kernel(memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>)

  func.func private @extern_kernel(%A: memref<1024xi32>, %B: memref<1024xi32>, %acc: memref<1024xi32>, %C: memref<1024xi32>) -> ()

  func.func private @kernel(%A: memref<1024xi32>, %lA: !physical.lock,
                            %B: memref<1024xi32>, %lB: !physical.lock,
                            %acc: memref<1024xi32>, %lacc: !physical.lock,
                            %C: memref<1024xi32>, %lC: !physical.lock) {
    cf.br ^bb
^bb:
    physical.lock_acquire<1>(%lA)
    physical.lock_acquire<1>(%lB)
    physical.lock_acquire<1>(%lacc)
    physical.lock_acquire<0>(%lC)
    func.call @extern_kernel(%A, %B, %acc, %C) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
    physical.lock_release<1>(%lC)
    physical.lock_release<0>(%lacc)
    physical.lock_release<0>(%lB)
    physical.lock_release<0>(%lA)
    cf.br ^bb
  }

// CHECK:  %50 = AIE.core(%18) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%22, Acquire, 1)
// CHECK:    AIE.useLock(%21, Acquire, 1)
// CHECK:    AIE.useLock(%20, Acquire, 1)
// CHECK:    AIE.useLock(%19, Acquire, 0)
// CHECK:    func.call @extern_kernel(%36, %40, %44, %45) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
// CHECK:    AIE.useLock(%19, Release, 1)
// CHECK:    AIE.useLock(%20, Release, 0)
// CHECK:    AIE.useLock(%21, Release, 0)
// CHECK:    AIE.useLock(%22, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  }
// CHECK:  %51 = AIE.core(%9) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%12, Acquire, 1)
// CHECK:    AIE.useLock(%11, Acquire, 1)
// CHECK:    AIE.useLock(%19, Acquire, 1)
// CHECK:    AIE.useLock(%10, Acquire, 0)
// CHECK:    func.call @extern_kernel(%38, %41, %45, %48) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
// CHECK:    AIE.useLock(%10, Release, 1)
// CHECK:    AIE.useLock(%19, Release, 0)
// CHECK:    AIE.useLock(%11, Release, 0)
// CHECK:    AIE.useLock(%12, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  }
// CHECK:  %52 = AIE.core(%13) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%17, Acquire, 1)
// CHECK:    AIE.useLock(%16, Acquire, 1)
// CHECK:    AIE.useLock(%15, Acquire, 1)
// CHECK:    AIE.useLock(%14, Acquire, 0)
// CHECK:    func.call @extern_kernel(%37, %42, %46, %47) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
// CHECK:    AIE.useLock(%14, Release, 1)
// CHECK:    AIE.useLock(%15, Release, 0)
// CHECK:    AIE.useLock(%16, Release, 0)
// CHECK:    AIE.useLock(%17, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  }
// CHECK:  %53 = AIE.core(%5) {
// CHECK:    cf.br ^bb1
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%8, Acquire, 1)
// CHECK:    AIE.useLock(%7, Acquire, 1)
// CHECK:    AIE.useLock(%14, Acquire, 1)
// CHECK:    AIE.useLock(%6, Acquire, 0)
// CHECK:    func.call @extern_kernel(%39, %43, %47, %49) : (memref<1024xi32>, memref<1024xi32>, memref<1024xi32>, memref<1024xi32>) -> ()
// CHECK:    AIE.useLock(%6, Release, 1)
// CHECK:    AIE.useLock(%14, Release, 0)
// CHECK:    AIE.useLock(%7, Release, 0)
// CHECK:    AIE.useLock(%8, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  }

  physical.core @kernel(%A_0_a, %lA_0_a, %B_0_0, %lB_0_0, %S_0_0, %lS_0_0, %S_0_1, %lS_0_1)
    { aie.tile = "6.3" }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_1_a, %lA_1_a, %B_0_1, %lB_0_1, %S_0_1, %lS_0_1,   %C_0,   %lC_0)
    { aie.tile = "6.4" }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_0_b, %lA_0_b, %B_1_0, %lB_1_0, %S_1_0, %lS_1_0, %S_1_1, %lS_1_1)
    { aie.tile = "7.3" }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

  physical.core @kernel(%A_1_b, %lA_1_b, %B_1_1, %lB_1_1, %S_1_1, %lS_1_1,   %C_1,   %lC_1)
    { aie.tile = "7.4" }: (memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock, memref<1024xi32>, !physical.lock) -> !physical.core

// CHECK:  %54 = AIE.shimDMA(%23) {
// CHECK:    %60 = AIE.dmaStart(MM2S1, ^bb1, ^bb3)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb2
// CHECK:    AIE.useLock(%25, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(2, 2)
// CHECK:    AIE.dmaBd(<%30 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%25, Release, 0)
// CHECK:    cf.br ^bb2
// CHECK:  ^bb2:  // pred: ^bb1
// CHECK:    AIE.useLock(%24, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(3, 3)
// CHECK:    AIE.dmaBd(<%31 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%24, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb3:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(MM2S0, ^bb4, ^bb6)
// CHECK:  ^bb4:  // 2 preds: ^bb3, ^bb5
// CHECK:    AIE.useLock(%27, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(0, 0)
// CHECK:    AIE.dmaBd(<%28 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%27, Release, 0)
// CHECK:    cf.br ^bb5
// CHECK:  ^bb5:  // pred: ^bb4
// CHECK:    AIE.useLock(%26, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(1, 1)
// CHECK:    AIE.dmaBd(<%29 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%26, Release, 0)
// CHECK:    cf.br ^bb4
// CHECK:  ^bb6:  // pred: ^bb3
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %55 = AIE.mem(%18) {
// CHECK:    %60 = AIE.dmaStart(S2MM1, ^bb1, ^bb2)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%21, Acquire, 0)
// CHECK:    AIE.dmaBd(<%40 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%21, Release, 1)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb2:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(S2MM0, ^bb3, ^bb4)
// CHECK:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:    AIE.useLock(%22, Acquire, 0)
// CHECK:    AIE.dmaBd(<%36 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%22, Release, 1)
// CHECK:    cf.br ^bb3
// CHECK:  ^bb4:  // pred: ^bb2
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %56 = AIE.mem(%13) {
// CHECK:    %60 = AIE.dmaStart(S2MM1, ^bb1, ^bb2)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%16, Acquire, 0)
// CHECK:    AIE.dmaBd(<%42 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%16, Release, 1)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb2:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(S2MM0, ^bb3, ^bb4)
// CHECK:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:    AIE.useLock(%17, Acquire, 0)
// CHECK:    AIE.dmaBd(<%37 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%17, Release, 1)
// CHECK:    cf.br ^bb3
// CHECK:  ^bb4:  // pred: ^bb2
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %57 = AIE.mem(%9) {
// CHECK:    %60 = AIE.dmaStart(MM2S0, ^bb1, ^bb2)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%10, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(6, 6)
// CHECK:    AIE.dmaBd(<%48 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%10, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb2:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(S2MM1, ^bb3, ^bb4)
// CHECK:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:    AIE.useLock(%11, Acquire, 0)
// CHECK:    AIE.dmaBd(<%41 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%11, Release, 1)
// CHECK:    cf.br ^bb3
// CHECK:  ^bb4:  // pred: ^bb2
// CHECK:    %62 = AIE.dmaStart(S2MM0, ^bb5, ^bb6)
// CHECK:  ^bb5:  // 2 preds: ^bb4, ^bb5
// CHECK:    AIE.useLock(%12, Acquire, 0)
// CHECK:    AIE.dmaBd(<%38 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%12, Release, 1)
// CHECK:    cf.br ^bb5
// CHECK:  ^bb6:  // pred: ^bb4
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %58 = AIE.mem(%5) {
// CHECK:    %60 = AIE.dmaStart(MM2S0, ^bb1, ^bb2)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%6, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(7, 7)
// CHECK:    AIE.dmaBd(<%49 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%6, Release, 0)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb2:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(S2MM1, ^bb3, ^bb4)
// CHECK:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:    AIE.useLock(%7, Acquire, 0)
// CHECK:    AIE.dmaBd(<%43 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%7, Release, 1)
// CHECK:    cf.br ^bb3
// CHECK:  ^bb4:  // pred: ^bb2
// CHECK:    %62 = AIE.dmaStart(S2MM0, ^bb5, ^bb6)
// CHECK:  ^bb5:  // 2 preds: ^bb4, ^bb5
// CHECK:    AIE.useLock(%8, Acquire, 0)
// CHECK:    AIE.dmaBd(<%39 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%8, Release, 1)
// CHECK:    cf.br ^bb5
// CHECK:  ^bb6:  // pred: ^bb4
// CHECK:    AIE.end
// CHECK:  }
// CHECK:  %59 = AIE.shimDMA(%0) {
// CHECK:    %60 = AIE.dmaStart(S2MM1, ^bb1, ^bb2)
// CHECK:  ^bb1:  // 2 preds: ^bb0, ^bb1
// CHECK:    AIE.useLock(%1, Acquire, 0)
// CHECK:    AIE.dmaBd(<%35 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%1, Release, 1)
// CHECK:    cf.br ^bb1
// CHECK:  ^bb2:  // pred: ^bb0
// CHECK:    %61 = AIE.dmaStart(S2MM0, ^bb3, ^bb4)
// CHECK:  ^bb3:  // 2 preds: ^bb2, ^bb3
// CHECK:    AIE.useLock(%2, Acquire, 0)
// CHECK:    AIE.dmaBd(<%34 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%2, Release, 1)
// CHECK:    cf.br ^bb3
// CHECK:  ^bb4:  // pred: ^bb2
// CHECK:    %62 = AIE.dmaStart(MM2S0, ^bb5, ^bb7)
// CHECK:  ^bb5:  // 2 preds: ^bb4, ^bb6
// CHECK:    AIE.useLock(%4, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(4, 4)
// CHECK:    AIE.dmaBd(<%32 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%4, Release, 0)
// CHECK:    cf.br ^bb6
// CHECK:  ^bb6:  // pred: ^bb5
// CHECK:    AIE.useLock(%3, Acquire, 1)
// CHECK:    AIE.dmaBdPacket(5, 5)
// CHECK:    AIE.dmaBd(<%33 : memref<1024xi32>, 0, 1024>, 0)
// CHECK:    AIE.useLock(%3, Release, 0)
// CHECK:    cf.br ^bb5
// CHECK:  ^bb7:  // pred: ^bb4
// CHECK:    AIE.end
// CHECK:  }

  physical.stream_dma(%sA#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<0>(%leA_0[1->0], %eA_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<1>(%leA_1[1->0], %eA_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.0", aie.engine = "MM2S", aie.id = "0" }
  physical.stream_dma(%sA_0_a#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_0_a[0->1], %A_0_a[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.3", aie.engine = "S2MM", aie.id = "0" }
  physical.stream_dma(%sA_0_b#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_0_b[0->1], %A_0_b[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.3", aie.engine = "S2MM", aie.id = "0" }
  physical.stream_dma(%sA_1_a#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_1_a[0->1], %A_1_a[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.4", aie.engine = "S2MM", aie.id = "0" }
  physical.stream_dma(%sA_1_b#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lA_1_b[0->1], %A_1_b[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.4", aie.engine = "S2MM", aie.id = "0" }

  physical.stream_dma(%sB_0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<2>(%leB_0_0[1->0], %eB_0_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<3>(%leB_0_1[1->0], %eB_0_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.0", aie.engine = "MM2S", aie.id = "1" }
  physical.stream_dma(%sB_1#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<4>(%leB_1_0[1->0], %eB_1_0[0:1024]: memref<1024xi32>, %1)
    %1 = physical.stream_dma_connect<5>(%leB_1_1[1->0], %eB_1_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.0", aie.engine = "MM2S", aie.id = "0" }
  physical.stream_dma(%sB_0_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_0_0[0->1], %B_0_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.3", aie.engine = "S2MM", aie.id = "1" }
  physical.stream_dma(%sB_0_1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_0_1[0->1], %B_0_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.4", aie.engine = "S2MM", aie.id = "1" }
  physical.stream_dma(%sB_1_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_1_0[0->1], %B_1_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.3", aie.engine = "S2MM", aie.id = "1" }
  physical.stream_dma(%sB_1_1#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%lB_1_1[0->1], %B_1_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.4", aie.engine = "S2MM", aie.id = "1" }

  physical.stream_dma(%sC_0#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<6>(%lC_0[1->0], %C_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "6.4", aie.engine = "MM2S", aie.id = "0" }
  physical.stream_dma(%sC_1#0: !physical.ostream<i32>) {
    %0 = physical.stream_dma_connect<7>(%lC_1[1->0], %C_1[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.4", aie.engine = "MM2S", aie.id = "0" }
  physical.stream_dma(%seC_0#1: !physical.istream<i32>) {
    %0 = physical.stream_dma_connect(%leC_0[0->1], %eC_0[0:1024]: memref<1024xi32>, %0)
  } { aie.tile = "7.0", aie.engine = "S2MM", aie.id = "0" }
  physical.stream_dma(%seC_1#1: !physical.istream<i32>) {
    %1 = physical.stream_dma_connect(%leC_1[0->1], %eC_1[0:1024]: memref<1024xi32>, %1)
  } { aie.tile = "7.0", aie.engine = "S2MM", aie.id = "1" }

}
