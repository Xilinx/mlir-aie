//===- AIETraceOps.td --------------------------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2025, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
// Defines operations for AIE trace configuration
//===----------------------------------------------------------------------===//

#ifndef AIE_TRACE_OPS
#define AIE_TRACE_OPS

include "aie/Dialect/AIE/IR/AIE.td"
include "aie/Dialect/AIE/IR/AIETraceAttrs.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// AIE_TraceOp - Top-Level Trace Configuration (Symbol)
//===----------------------------------------------------------------------===//

def AIE_TraceOp : AIE_Op<"trace", [
    Symbol,
    HasParent<"DeviceOp">,
    SingleBlockImplicitTerminator<"EndOp">,
    DeclareOpInterfaceMethods<OpAsmOpInterface, ["getAsmResultNames"]>
  ]> {
  let summary = "Declare a named trace configuration for a tile";
  
  let description = [{
    Declares a named trace configuration that can be invoked from a runtime
    sequence. The trace operation is a symbol that defines what events to
    monitor and how to configure the trace hardware, but does not actually
    apply the configuration. Configuration is applied via `aie.trace.start_config`
    within a `aiex.runtime_sequence` block.
    
    Example:
    ```mlir
    %tile02 = aie.tile(0, 2)
    
    aie.trace @trace_tile02(%tile02) {
      aie.trace.mode Event-Time
      aie.trace.packet id=1 type=core
      aie.trace.event<INSTR_EVENT_0>
      aie.trace.event<INSTR_VECTOR>
      aie.trace.start broadcast=15
      aie.trace.stop broadcast=14
    }
    ```
  }];
  
  let arguments = (ins
    Index:$tile,
    SymbolNameAttr:$sym_name,
    OptionalAttr<AIEI32Attr>:$buffer_size
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name `(` $tile `)` (`buffer_size` `=` $buffer_size^)? 
    $body attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceModeOp - Trace Mode Selection
//===----------------------------------------------------------------------===//

def AIE_TraceModeOp : AIE_Op<"trace.mode", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Set trace mode (Event-Time, Event-PC, or Execution)";
  
  let description = [{
    Specifies the trace mode. Default: Event-Time
    
    Example:
    ```mlir
    aie.trace.mode Event-Time
    ```
  }];
  
  let arguments = (ins TraceModeAttr:$mode);
  
  let assemblyFormat = [{
    $mode attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceEventOp - Event Slot Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceEventOp : AIE_Op<"trace.event", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Monitor a specific event in this trace unit";
  
  let description = [{
    Adds an event to be monitored by the trace unit. Each tile type
    (core/mem/memtile/shimtile) has architecture-specific events.
    
    Maximum 8 events per trace unit (verified at compile time).
    
    Example:
    ```mlir
    aie.trace.event<INSTR_EVENT_0>
    aie.trace.event<LOCK_STALL>
    ```
  }];
  
  let arguments = (ins
    TraceEventAttr:$event,
    OptionalAttr<StrAttr>:$label
  );
  
  let assemblyFormat = [{
    $event (`label` `=` $label^)? attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TracePacketOp - Packet Routing Configuration
//===----------------------------------------------------------------------===//

def AIE_TracePacketOp : AIE_Op<"trace.packet", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Enable packet-switched trace routing";
  
  let description = [{
    Enables packet routing for trace data. Assigns packet ID (1-31) and
    packet type to differentiate tile sources during parsing.
    
    Example:
    ```mlir
    aie.trace.packet id=1 type=core
    ```
  }];
  
  let arguments = (ins
    ConfinedAttr<AIEI32Attr, [IntMinValue<1>, IntMaxValue<31>]>:$id,
    TracePacketTypeAttr:$type
  );
  
  let assemblyFormat = [{
    `id` `=` $id `type` `=` $type attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceStartEventOp - Trace Start Event Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceStartEventOp : AIE_Op<"trace.start", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure trace start event";
  
  let description = [{
    Specifies the event that triggers trace capture to begin.
    For multi-tile tracing, typically uses a broadcast event (default: 15).
    
    Example:
    ```mlir
    aie.trace.start broadcast=15
    aie.trace.start event=<TRUE>
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<AIEI32Attr>:$broadcast,
    OptionalAttr<TraceEventAttr>:$event
  );
  
  let assemblyFormat = [{
    (`broadcast` `=` $broadcast^)?
    (`event` `=` $event^)?
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceStopEventOp - Trace Stop Event Configuration
//===----------------------------------------------------------------------===//

def AIE_TraceStopEventOp : AIE_Op<"trace.stop", [
    HasParent<"TraceOp">
  ]> {
  let summary = "Configure trace stop event";
  
  let description = [{
    Specifies the event that triggers trace capture to end.
    For multi-tile tracing, typically uses a broadcast event (default: 14).
    
    Example:
    ```mlir
    aie.trace.stop broadcast=14
    aie.trace.stop event=<USER_EVENT_0>
    ```
  }];
  
  let arguments = (ins
    OptionalAttr<AIEI32Attr>:$broadcast,
    OptionalAttr<TraceEventAttr>:$event
  );
  
  let assemblyFormat = [{
    (`broadcast` `=` $broadcast^)?
    (`event` `=` $event^)?
    attr-dict
  }];
  
  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// AIE_TraceConfigOp - Configuration Container (Intermediate, Symbol)
//===----------------------------------------------------------------------===//

def AIE_TraceConfigOp : AIE_Op<"trace.config", [
    Symbol,
    HasParent<"DeviceOp">,
    SingleBlockImplicitTerminator<"EndOp">
  ]> {
  let summary = "Intermediate representation of lowered trace configuration";
  
  let description = [{
    Generated by AIETraceToConfigPass. Contains calculated register
    addresses and values needed to configure trace hardware.
    
    This is an intermediate representation that makes it easier to perform
    optimizations before final lowering to NPU register writes.
    
    Example (generated, not user-written):
    ```mlir
    aie.trace.config @trace_tile02_config(%tile02) {
      aie.trace.reg register="Trace_Control0" field="Trace_Start_Event" value=15
      aie.trace.reg register="Trace_Control0" field="Mode" value="Event-Time"
    }
    ```
  }];
  
  let arguments = (ins
    Index:$tile,
    SymbolNameAttr:$sym_name
  );
  
  let regions = (region SizedRegion<1>:$body);
  
  let assemblyFormat = [{
    $sym_name `(` $tile `)` $body attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceRegOp - Logical Register/Bitfield Specification
//===----------------------------------------------------------------------===//

def AIE_TraceRegOp : AIE_Op<"trace.reg", [
    HasParent<"TraceConfigOp">
  ]> {
  let summary = "Specify a trace register field write by logical name";
  
  let description = [{
    Specifies a single register field write needed to configure trace hardware.
    Uses logical register names (e.g., "Trace_Control0"), field names
    (e.g., "Trace_Start_Event"), and values (integers or event names). If field
    is omitted, entire register is written.
    
    Example:
    ```mlir
    aie.trace.reg register="Trace_Control0" field="Trace_Start_Event" value=15
    aie.trace.reg register="Trace_Control0" field="Mode" value=0
    aie.trace.reg register="Trace_Control0" value=0x12345678
    ```
  }];
  
  let arguments = (ins
    StrAttr:$reg_name,
    StrAttr:$field,
    AnyAttr:$value,
    OptionalAttr<StrAttr>:$comment
  );
  
  let assemblyFormat = [{
    `register` `=` $reg_name `field` `=` $field `value` `=` $value
    (`comment` `=` $comment^)?
    attr-dict
  }];
}

//===----------------------------------------------------------------------===//
// AIE_TraceStartConfigOp - Runtime Configuration Invocation
//===----------------------------------------------------------------------===//

def AIE_TraceStartConfigOp : AIE_Op<"trace.start_config", []> {
  let summary = "Apply a trace configuration in runtime sequence";
  
  let description = [{
    Invokes a named trace configuration within a runtime sequence.
    During lowering, this operation is replaced by the actual register
    writes (npu_write32) that implement the trace configuration.
    
    Example:
    ```mlir
    aiex.runtime_sequence @seq {
      aie.trace.start_config @my_trace
    }
    ```
  }];
  
  let arguments = (ins
    FlatSymbolRefAttr:$trace_config
  );
  
  let assemblyFormat = [{
    $trace_config attr-dict
  }];
  
  let hasVerifier = 1;
}

#endif // AIE_TRACE_OPS
