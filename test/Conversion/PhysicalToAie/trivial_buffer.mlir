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

// CHECK: %[[Tile:.*]] = AIE.tile(6, 3)

// CHECK: AIE.external_buffer 2203318222848 : memref<1024xi32>
%0 = physical.buffer() { aie.external_address = "2203318222848" }: memref<1024xi32>

// CHECK: AIE.buffer(%[[Tile]]) : memref<1024xi32>
%1 = physical.buffer() { aie.tile = "6.3" }: memref<1024xi32>
