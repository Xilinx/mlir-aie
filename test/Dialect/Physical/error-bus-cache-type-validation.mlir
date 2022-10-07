// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

%bus1 = physical.bus(): !physical.bus<i32>
%bus2 = physical.bus(): !physical.bus<i32>

// CHECK: expects different type than prior uses: '!physical.bus<i16>' vs '!physical.bus<i32>'
%cache = physical.bus_cache(%bus1, %bus2) : !physical.bus_cache<i16, 1024>