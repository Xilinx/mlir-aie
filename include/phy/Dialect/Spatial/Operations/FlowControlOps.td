//===- FlowControlOps.td -----------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD
#define MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD

include "phy/Dialect/Spatial/SpatialBase.td"

//===----------------------------------------------------------------------===//
// ValidOp
//===----------------------------------------------------------------------===//

def ValidOp : Spatial_Op<"valid"> {
  let summary = "non-blocking test whether the queue is ready for dequeuing";
  let description = [{
    The `spatial.valid` operation tests if its operand queue has an element for
    dequeuing.  This operation is non-blocking and returns a boolean result,
    which is true if there is an element to dequeue.  Otherwise, the operation
    returns false.  When `spatial.valid` returns true, `spatial.front` and
    `spatial.pop` return immediately without blocking.

    Example:

    ```mlir
    %0 = spatial.valid(%q : !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FullOp
//===----------------------------------------------------------------------===//

def FullOp : Spatial_Op<"full"> {
  let summary = "non-blocking test whether the queue is full";
  let description = [{
    The `spatial.full` operation tests its operand queue if the given queue is
    ready for enqueuing elements.  This operation is non-blocking and returns a
    boolean result, which is true if it is full, or false if it is ready for
    enqueuing.  When `spatial.valid` returns true, `spatial.emplace` and
    `spatial.push` return immediately without blocking.

    Example:

    ```mlir
    %0 = spatial.full(%q : !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  let results = (outs I1:$result);

  let assemblyFormat = [{
    `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// FrontOp
//===----------------------------------------------------------------------===//

def FrontOp : Spatial_Op<"front",
    [TypesMatchWith<"'queue' data has the same type as result",
                    "result", "queue",
                    "QueueType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking retrieve the next data accessor";
  let description = [{
    The `spatial.front` operation retrieves the next, i.e., the oldest element
    from its operand queue.  This operation is blocking and returns the
    retrieved data accessor as its result.

    Example:

    ```mlir
    %queue = spatial.queue() : !spatial.queue<memref<i32>>
    %accessor = spatial.front(%queue) : memref<i32>
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $queue `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// EmplaceOp
//===----------------------------------------------------------------------===//

def EmplaceOp : Spatial_Op<"emplace",
    [TypesMatchWith<"'queue' data has the same type as result",
                    "result", "queue",
                    "QueueType::get($_self.getContext(), $_self)">]> {

  let summary = "blocking construct the last data accessor";
  let description = [{
    The `spatial.emplace` operation allocate a memory space at the last of the
    queue for preparing an element to enqueue.  When the queue has no space to
    be allocated, this operation blocks until it is available.  It returns the
    accessor to the last element for preparing the data as its result.

    Example:

    ```mlir
    %queue = spatial.queue() : !spatial.queue<memref<i32>>
    %accessor = spatial.emplace(%queue) : memref<i32>
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  let results = (outs AnyType:$result);

  let assemblyFormat = [{
    `(` $queue `)` attr-dict `:` type($result) 
  }];
}

//===----------------------------------------------------------------------===//
// PopOp
//===----------------------------------------------------------------------===//

def PopOp : Spatial_Op<"pop"> {
  let summary = "blocking remove the next element in the queue";
  let description = [{
    The `spatial.pop` operation removes the next, i.e., the oldest element in
    the queue, whose accessor can be retrieved by calling `front`.

    Example:

    ```mlir
    spatial.pop(%queue: !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  
  let assemblyFormat = [{
    `(` $queue `:` type($queue) `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// PushOp
//===----------------------------------------------------------------------===//

def PushOp : Spatial_Op<"push"> {
  let summary = "commit the last element to the queue";
  let description = [{
    The `spatial.push` operation commits the last `spatial.emplace` constructed
    element to the queue.  The enqueued element becomes the new last element in
    the queue.

    When there is no element allocated by `spatial.emplace`, this operation
    implicitly calls the corresponding `spatial.emplace` right before it, with
    the element content undefined.  In this case, the operation is blocking.
    This behavior is useful when !spatial.queue<none> is used.

    Example:

    ```mlir
    spatial.push(%queue: !spatial.queue<memref<i32>>)
    ```
  }];

  let arguments = (ins AnyQueueType:$queue);
  
  let assemblyFormat = [{
    `(` $queue `:` type($queue) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_SPATIAL_FLOW_CONTROL_OPS_TD
