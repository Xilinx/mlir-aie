//===- LayoutOps.td ----------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_PHY_DIALECT_LAYOUT_OPS_TD
#define MLIR_PHY_DIALECT_LAYOUT_OPS_TD

include "phy/Dialect/Layout/LayoutBase.td"
include "phy/Dialect/Spatial/SpatialTypes.td"

include "mlir/IR/EnumAttr.td"

//===----------------------------------------------------------------------===//
// EndOp
//===----------------------------------------------------------------------===//

def EndOp : Layout_Op<"end", [Terminator,
                              ParentOneOf<["PlatformOp", "DeviceOp"]>]> {
  let summary = "end of usage of platform or device";
  let description = [{
    'layout.end' is an implied terminator for the 'layout.platform' or
    'layout.device' regions.  You do not need to manually write it.
  }];
  let assemblyFormat = "attr-dict";
}

//===----------------------------------------------------------------------===//
// PlatformOp
//===----------------------------------------------------------------------===//

def PlatformOp : Layout_Op<"platform",
    [SingleBlockImplicitTerminator<"EndOp">,
     NoRegionArguments, ParentOneOf<["mlir::ModuleOp", "PlatformOp"]>]> {

  let summary = "usage of a platform";
  let description = [{
    An operation that uses an abstract physical platform for the enclosing region.
    The region is used to place and route vertices and edges onto the platform.
    The slots for placement, and the wires for routing, are dependent on which
    platform is used.  This operation must be enclosed by a module or another
    'layout.platform' operation.

    Example:

    ```mlir
    layout.platform<"versal"> {
      layout.device<"pl"> {}
      layout.route<["pl-aie"]>(%flow1)
    }
    ```
  }];

  let arguments = (ins StrAttr:$platform);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ `<` $platform `>` regions attr-dict }];
}

//===----------------------------------------------------------------------===//
// DeviceOp
//===----------------------------------------------------------------------===//

def DeviceOp : Layout_Op<"device",
    [SingleBlockImplicitTerminator<"EndOp">,
     NoRegionArguments, HasParent<"PlatformOp">]> {

  let summary = "usage of a device";
  let description = [{
    An operation that uses an abstract physical device for the enclosing region.
    The region is used to place and route vertices and edges onto the device.
    The slots for placement, and the wires for routing, are dependent on which
    device is used.  This operation must be enclosed by a 'layout.platform'
    operation.

    Example:

    ```mlir
    layout.platform<"versal"> {
      layout.device<"aie"> {
        layout.place<"tile/7.2/core">(%node)
        layout.route<["tile/7.2/memory-access"]>(%flow)
      }
    }
    ```
  }];

  let arguments = (ins StrAttr:$device);
  let regions = (region SizedRegion<1>:$body);

  let assemblyFormat = [{ `<` $device `>` regions attr-dict }];
}

//===----------------------------------------------------------------------===//
// PlaceOp
//===----------------------------------------------------------------------===//

def PlaceOp : Layout_Op<"place", [HasParent<"DeviceOp">]> {
  let summary = "placement of a vertex";
  let description = [{
    An operation that place the 'vertex' to the specific slot of the
    enclosing device.  This operation must be enclosed by a 'layout.device'
    operation.  The 'slot' argument is target-dependent and is
    comma-seperated.  Please check the documentation of the platform and the
    device.

    Depending on the device, a slot may be time-multiplexed by multiple nodes
    or queues.  In this case, the actual supported concurrency may be reduced.
    This sharing can be limited or required using `layout.share.*` attributes
    on the `spatial` operations.

    Example:

    ```mlir
    layout.place<["tile/7.2/core"]>(%node)
    layout.place<["tile/7.2/buffer"]>(%queue)
    ```
  }];

  let arguments = (ins StrAttr:$slot, AnyVertexType:$vertex);
  let assemblyFormat = [{
    `<` $slot `>` `(` $vertex `:` type($vertex)  `)` attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// RouteOp
//===----------------------------------------------------------------------===//

def RouteOp : Layout_Op<"route", [ParentOneOf<["PlatformOp", "DeviceOp"]>]> {
  let summary = "route of a flow connection";
  let description = [{
    An operation that route the flow, which connects the 'src' vertex to the
    'dest' vertex, to the specific wires of the enclosing device or platform.
    The 'wires' argument is a list of slots and is target-dependent.  Please
    check the documentation of the platform and the device.

    Depending on the device, a wire may be time-multiplexed by multiple flows.
    In this case, the actual supported concurrency may be reduced.  This
    sharing can be limited or required using layout.share.* attributes on the
    spatial operations.

    Depending on the device, a wire may contain buffering or caching.  Usage
    or rejection of such a wire can be specified using `layout.wire.*`
    attributes on this operation.
    
    Example:

    ```mlir
    layout.route<["tile/7.2/dma/mm2s0",
                  "tile/7.3/switchbox/7.2-8.3",
                  "tile/8.3/dma/s2mm0",
                  "tile/8.3/double-buffering",
                  "tile/8.3/memory-access"]>(
                    %n: !spatial.node
                 -> %q: !spatial.queue<memref<8xi32>>)
    ```
  }];

  let hasVerifier = 1;

  let arguments = (ins StrArrayAttr:$wires, AnyVertexType:$src, AnyVertexType:$dest);
  let assemblyFormat = [{
    `<` $wires `>` `(` $src `:` type($src) `->` $dest `:` type($dest) `)` attr-dict
  }];
}

#endif  // MLIR_PHY_DIALECT_LAYOUT_OPS_TD
