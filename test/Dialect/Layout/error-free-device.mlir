// RUN: (phy-opt %s 2>&1 || true) | FileCheck %s
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

// CHECK: 'layout.device' op expects parent op 'layout.platform'
layout.device<"hls"> {}
