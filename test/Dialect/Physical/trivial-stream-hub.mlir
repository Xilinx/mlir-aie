// RUN: phy-opt %s | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: physical.stream
%stream1:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream
%stream2:2 = physical.stream(): (!physical.ostream<i32>, !physical.istream<i32>)

// CHECK: physical.stream_hub
%hub = physical.stream_hub(%stream1#1, %stream2#0)
     : (!physical.istream<i32>, !physical.ostream<i32>)
     -> !physical.stream_hub<i32>