// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: physical.bus
%bus1 = physical.bus(): !physical.bus<i32>
// CHECK: physical.bus
%bus2 = physical.bus(): !physical.bus<i32>
// CHECK: physical.bus_cache
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i32, 1024>
