//===- LockTypes.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_LOCK_TYPES_TD
#define MLIR_PHY_DIALECT_PHYSICAL_LOCK_TYPES_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// LockType
//===----------------------------------------------------------------------===//

def LockType : Physical_Type<"Lock", "lock"> {
  let summary = "An atomic synchronization element";
  let description = [{
    Locks are atomic storage elements, which provides a primitive synchronization
    mechanism that limits access to a resource.  A lock can be assigned a state,
    and it can be acquired by only one user at a time when the acquiring state
    matches the state stored in the lock.  The lock can be acquired again only
    when the user release it with a state.
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_LOCK_TYPES_TD
