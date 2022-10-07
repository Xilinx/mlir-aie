//===- LockOps.td ------------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_PHYSICAL_LOCK_OPS_TD
#define MLIR_PHY_DIALECT_PHYSICAL_LOCK_OPS_TD

include "phy/Dialect/Physical/PhysicalBase.td"

//===----------------------------------------------------------------------===//
// LockOp
//===----------------------------------------------------------------------===//

def LockOp: Physical_Op<"lock"> {
  let summary = "lock creation operation";
  let description = [{
    The `physical.lock` operation represents a creation of a lock.  A lock is
    an atomic unit that can be used to limit access to a resource.  This
    operation returns the created lock.  `state` specifies the initially
    released state of the lock when the system is up.

    Example:
    ```
      %lock = physical.lock<0>()
    ```
  }];
  
  let arguments = (ins I64Attr:$state);
  let results = (outs LockType:$lock);

  let assemblyFormat = [{
    `<` $state `>`  `(` `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// LockAcquireOp
//===----------------------------------------------------------------------===//

def LockAcquireOp: Physical_Op<"lock_acquire"> {
  let summary = "lock acquisition operation";
  let description = [{
    The `physical.lock_acquire` operation acquires a lock when the lock is
    released in the specified `state`.  It is blocking and returns only when
    the lock is acquired, and other users cannot acquire the lock until it
    is released.

    Example:
    ```
      physical.lock_acquire<0>(%lock)
    ```
  }];
  
  let arguments = (ins I64Attr:$state, LockType:$lock);

  let assemblyFormat = [{
    `<` $state `>`  `(` $lock `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// LockReleaseOp
//===----------------------------------------------------------------------===//

def LockReleaseOp: Physical_Op<"lock_release"> {
  let summary = "lock releasing operation";
  let description = [{
    The `physical.lock_release` operation release a lock to the specified
    `state`.  Once the lock is released, it can be acquired by other users.

    Example:
    ```
      physical.lock_release<0>(%lock)
    ```
  }];
  
  let arguments = (ins I64Attr:$state, LockType:$lock);

  let assemblyFormat = [{
    `<` $state `>`  `(` $lock `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_PHYSICAL_LOCK_OPS_TD
